
import os, pickle, time
import numpy as np

from multiprocessing import get_context

from .utils_data import set_para, build_struct_PC_results
from caiman.utils.utils import load_dict_from_hdf5

class detect_PC:

    def __init__(self,pathData,pathBehavior,pathResults,nbin=100,
                 nP=0,
                 plt_bool=False,sv_bool=False,suffix=''):
        '''
            initializing detect placefields class

            TODO:
                * enable providing behavior data, instead of loading/processing it automatically

        '''

        ### set global parameters and data for all processes to access
        self.para = set_para(pathData,nP=nP,nbin=nbin,plt_bool=plt_bool,sv_bool=sv_bool,suffix=suffix)
        
        self.load_behavior(pathBehavior)       ## load and process behavior
        self.load_activity(pathData)


    def run_detection(self,S=None,
          f_max=1,
          specific_n=None,dataSet='OnACID_results.hdf5',
          return_results=False,rerun=False,
          artificial=False,
          mode_info='MI',mode_activity='spikes',assignment=None):
        '''
            executes the pipeline for processing mouse behavior and neuron activity to 
            extract place fields and some related statistics 
        '''
        
        global t_start
        t_start = time.time()
        self.f_max = f_max
        self.para['modes']['info'] = mode_info
        self.para['modes']['activity'] = mode_activity
        # self.tmp = {}   ## dict to store some temporary variables in
        

        if not (specific_n is None):
            self.para['n'] = specific_n
            #self.S = S[specific_n,:]
            result = self.PC_detect(S[specific_n,:])
            return result

        idx_process = np.arange(self.nCells)
        nCells_process = len(idx_process)

        if nCells_process:

            print('run detection on %d neurons'%nCells_process)
            
            result_tmp = []
            if self.para['nP'] > 0:
                pool = get_context("spawn").Pool(self.para['nP'])
                batchSz = 500
                nBatch = nCells_process//batchSz

                for i in range(nBatch+1):
                    idx_batch = idx_process[i*batchSz:min(nCells_process,(i+1)*batchSz)]
                    result_tmp.extend(pool.starmap(self.PC_detect,zip(S[idx_batch,:],self.activity['SNR_comp'][idx_batch],self.activity['r_values'][idx_batch])))
                    print('\n\t\t\t ------ mouse %s --- session %d ------ %d / %d neurons processed\t ------ \t time passed: %7.2fs\n'%(self.para['mouse'],self.para['session'],min(nCells_process,(i+1)*batchSz),nCells_process,time.time()-t_start))
            else:
                for n0,n in enumerate(idx_process):
                    result_tmp.append(self.PC_detect(S[n,:],self.activity['SNR_comp'][n],self.activity['r_values'][n]))
                    print('\t\t\t ------ mouse %s --- session %d ------ %d / %d neurons processed\t ------ \t time passed: %7.2fs'%(self.para['mouse'],self.para['session'],n0+1,nCells_process,time.time()-t_start))
        
            results = build_struct_PC_results(self.nCells,self.para['nbin'],self.data['trials']['ct'],1+len(self.para['CI_arr']))

        return result_tmp


    def PC_detect(self,S,SNR=None,r_value=None):
    #def PC_detect(varin):
        t_start = time.time()
        result = build_struct_PC_results(1,self.para['nbin'],self.data['trials']['ct'],1+len(self.para['CI_arr']))

        if not (SNR is None):
            result['status']['SNR'] = SNR
            result['status']['r_value'] = r_value

        T = S.shape[0]
        # try:
        active, result['firingstats']['rate'] = self.get_active_Ca(S)

        # if (result['status']['SNR'] < 2) | (results['r_value'] < 0):
        #     print('component not considered to be proper neuron')
        #     return

        if result['firingstats']['rate']==0:
            print('no activity for this neuron')
            return result

        ### get trial-specific activity and overall firingmap stats
        trials_S, result['firingstats']['trial_map'] = self.get_trials_activity(active)

        ## obtain mutual information first - check if (computational cost of) finding fields is worth it at all
        t_start = time.time()
        if self.para['modes']['info']:
            MI_tmp = self.test_MI(active,trials_S)
            # print(MI_tmp)
            for key in MI_tmp.keys():
                result['status'][keyw] = MI_tmp[key]
        #print('time taken (information): %.4f'%(time.time()-t_start))

        result = self.get_correlated_trials(result,smooth=2)
        # print(result['firingstats']['trial_field'])
        # return

        firingstats_tmp = self.get_firingstats_from_trials(result['firingstats']['trial_map'])
        for key in firingstats_tmp.keys():
            result['firingstats'][key] = firingstats_tmp[key]
        # return
        # if np.any(result['firingstats']['trial_field']) and ((result['status']['SNR']>2) or np.isnan(result['status']['SNR'])):  # and (result['status']['MI_value']>0.1)     ## only do further processing, if enough trials are significantly correlated
        for t in range(5):
            trials = np.where(result['firingstats']['trial_field'][t,:])[0]
            if len(trials)<1:
                # print(f'skipping trial {t}')
                continue

            firingstats_tmp = self.get_firingstats_from_trials(result['firingstats']['trial_map'],trials,complete=False)

            #print(gauss_smooth(firingstats_tmp['map'],2))

            # if (gauss_smooth(firingstats_tmp['map'],4)>(self.para['rate_thr']/2)).sum()>self.para['width_thr']:

            ### do further tests only if there is "significant" mutual information

            for f in range(self.f_max+1):
                field = self.run_nestedSampling(result['firingstats'],firingstats_tmp['map'],f)
            ## pick most prominent peak and store into result, if bayes factor > 1/2
            if field['Bayes_factor'][0] > 0:

                dTheta = np.abs(np.mod(field['parameter'][3,0]-result['fields']['parameter'][:t,3,0]+self.para['L_track']/2,self.para['L_track'])-self.para['L_track']/2)
                if not np.any(dTheta < 10):   ## should be in cm
                    ## store results into array index "t"
                    for key in field.keys():#['parameter','p_x','posterior_mass']:
                        result['fields'][key][t,...] = field[key]
                    result['fields']['nModes'] += 1

                    ## reliability is calculated later
                    result['fields']['reliability'][t], _, _ = get_reliability(result['firingstats']['trial_map'],result['firingstats']['map'],result['fields']['parameter'],t)

                    if self.para['plt_bool']:
                        self.plt_results(result,t)

        t_process = time.time()-t_start

        #print('get spikeNr - time taken: %5.3g'%(t_end-t_start))
        print_msg = 'p-value: %.2f, value (MI/Isec): %.2f / %.2f, '%(result['status']['MI_p_value'],result['status']['MI_value'],result['status']['Isec_value'])

        if result['fields']['nModes']>0:
            print_msg += ' \t Bayes factor (reliability) :'
            for f in np.where(result['fields']['Bayes_factor']>1/2)[0]:#range(result['fields']['nModes']):
                print_msg += '\t (%d): %.2f+/-%.2f (%.2f), '%(f+1,result['fields']['Bayes_factor'][f,0],result['fields']['Bayes_factor'][f,1],result['fields']['reliability'][f])
        if not(SNR is None):
            print_msg += '\t SNR: %.2f, \t r_value: %.2f'%(SNR,r_value)
        print_msg += ' \t time passed: %.2fs'%t_process
        print(print_msg)

        #except (KeyboardInterrupt, SystemExit):
            #raise
        # except:# KeyboardInterrupt: #:# TypeError:#
        #   print('analysis failed: (-)')# p-value (MI): %.2f, \t bayes factor: %.2fg+/-%.2fg'%(result['status']['MI_p_value'],result['status']['Bayes_factor'][0,0],result['status']['Bayes_factor'][0,1]))
        #   #result['fields']['nModes'] = -1

        return result#,sampler



    def load_behavior(self,pathBehavior,T=None):
        '''
            loads behavior from specified path
            Requires file to contain a dictionary with values for each frame, aligned to imaging data:
                * time      - time in seconds
                * position  - mouse position
                * active    - boolean array defining active frames (included in analysis)

        '''
        ### load data
        with open(pathBehavior,'rb') as f:
            loadData = pickle.load(f)

        if T is None:
            T = loadData['time'].shape[0]
        
        ## first, handing over some general data
        data = {}
        for key in ['active','time','position']:
            data[key] = loadData[key]


        ## apply binning
        min_val,max_val = np.nanpercentile(data['position'],(0.1,99.9)) # this could/should be done in data aligning
        environment_length = max_val - min_val

        data['binpos'] = np.minimum((data['position'] - min_val) / environment_length * self.para['nbin'],self.para['nbin']-1).astype('int')
        data['binpos_coarse'] = np.minimum((data['position']-min_val) / environment_length * self.para['nbin_coarse'],self.para['nbin_coarse']-1).astype('int')


        ## define trials
        data['trials'] = {}
        data['trials']['start'] = np.hstack([0,np.where(np.diff(data['position'])<(-environment_length/2))[0] + 1,len(data['time'])-1])


        ## remove partial trials from data (if fraction of length < partial_threshold)
        partial_threshold = 0.5
        if not (data['binpos'][0] < self.para['nbin']*(1-partial_threshold)):
            data['active'][:max(0,data['trials']['start'][0])] = False

        if not (data['binpos'][-1] >= self.para['nbin']*partial_threshold):
            data['active'][data['trials']['start'][-1]:] = False

        data['nFrames'] = np.count_nonzero(data['active'])

        ### preparing data for active periods, only

        ## defining arrays of active time periods
        data['binpos_active'] = data['binpos'][data['active']]
        data['binpos_coarse_active'] = data['binpos_coarse'][data['active']]
        data['time_active'] = data['time'][data['active']]

        ## define start points
        data['trials']['start_active'] = np.hstack([0,np.where(np.diff(data['binpos_active'])<(-self.para['nbin']/2))[0] + 1,data['active'].sum()])
        data['trials']['start_active_t'] = data['time'][data['active']][data['trials']['start_active'][:-1]]
        data['trials']['ct'] = len(data['trials']['start_active']) - 1


        ## getting trial-specific behavior data
        data['trials']['dwelltime'] = np.zeros((data['trials']['ct'],self.para['nbin']))
        data['trials']['nFrames'] = np.zeros(data['trials']['ct'],'int')#.astype('int')

        data['trials']['trial'] = {}
        for t in range(data['trials']['ct']):
            data['trials']['trial'][t] = {}
            data['trials']['trial'][t]['binpos_active'] = data['binpos_active'][data['trials']['start_active'][t]:data['trials']['start_active'][t+1]]
            data['trials']['dwelltime'][t,:] = np.histogram(data['trials']['trial'][t]['binpos_active'],self.para['bin_array_centers'])[0]/self.para['f']
            data['trials']['nFrames'][t] = len(data['trials']['trial'][t]['binpos_active'])
        
        return data


    def load_activity(self,pathData):
        
        ## load activity data from CaImAn results
        ld = load_dict_from_hdf5(pathData)
        print(ld.keys())
        self.activity = {}
        self.activity['S'] = ld['S']
        self.activity['S'][self.activity['S']<0] = 0

        if self.activity['S'].shape[0] > 8000:
            self.activity['S'] = self.activity['S'].transpose()

        self.nCells = self.activity['S'].shape[0]

        for key in ['SNR_comp','r_values']:
            self.activity[key] = ld[key] if key in ld.keys() else np.full(self.nCells,np.NaN)
        for key in ['idx_evaluate','idx_previous']:
            self.activity[key] = ld[key].astype('bool') if key in ld.keys() else np.full(self.nCells,np.NaN)

        return self.activity
    
