import time, random, os
import numpy as np
import scipy as sp
import scipy.stats as sstats
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import rc

from caiman.utils.utils import load_dict_from_hdf5

from dynesty import DynamicNestedSampler, utils, plotting

import ultranest
from ultranest.plot import cornerplot
import ultranest.stepsampler
# implement instead !! https://dynesty.readthedocs.io/en/latest/overview.html

from .utils_data import build_struct_PC_results
from .utils import get_average, get_MI, jackknife, ecdf, compute_serial_matrix, corr0, gauss_smooth, get_reliability, get_firingrate, get_firingmap, gamma_paras, lognorm_paras, add_number

from .spike_shuffling import shuffling
from .HierarchicalBayesInference import *

class PC_detection_inference:

    def __init__(self,behavior,para):
        '''
            input:
                - behavior data
                - some parameters
        '''

        self.para = para
        self.behavior = behavior

        self.f_max = 1



    def run_detection(self,S):
        '''
            function to find place fields in neuron activity S
            relies on behavior data being loaded into the class,
            allows processing activity data of one session in parallel

            TODO
                * this should receive active S, only
                * SNR, r_value should be used outside to filter, which ones to process

        '''

        self.S = S

        #def PC_detect(varin):
        t_start = time.time()
        result = build_struct_PC_results(1,self.para['nbin'],self.behavior['trials']['ct'],1+len(self.para['CI_arr']))

        # result['status']['SNR'] = SNR
        # result['status']['r_value'] = r_value


        ### get overall as well as trial-specific activity and firingmap stats
        self.prepare_activity()
        result['firingstats']['rate'] = self.activity['firingrate']
        result['firingstats']['trial_map'] = self.activity['trials_firingmap']

        if result['firingstats']['rate']==0:
            print('no activity for this neuron')
            return result


        ## calculate mutual information 
        ## check if (computational cost of) finding fields is worth it at all
        t_start = time.time()
        if self.para['modes']['info']:
            MI_tmp = self.test_MI()
            for key in MI_tmp.keys():
                result['status'][key] = MI_tmp[key]
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
            self.tmp = {}
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

        t_process = time.time()-t_start

        #print('get spikeNr - time taken: %5.3g'%(t_end-t_start))
        print_msg = 'p-value: %.2f, value (MI/Isec): %.2f / %.2f, '%(result['status']['MI_p_value'],result['status']['MI_value'],result['status']['Isec_value'])

        if result['fields']['nModes']>0:
            print_msg += ' \t Bayes factor (reliability) :'
            for f in np.where(result['fields']['Bayes_factor']>1/2)[0]:#range(result['fields']['nModes']):
                print_msg += '\t (%d): %.2f+/-%.2f (%.2f), '%(f+1,result['fields']['Bayes_factor'][f,0],result['fields']['Bayes_factor'][f,1],result['fields']['reliability'][f])
        # if not(SNR is None):
        #     print_msg += '\t SNR: %.2f, \t r_value: %.2f'%(SNR,r_value)
        print_msg += ' \t time passed: %.2fs'%t_process
        print(print_msg)

        #except (KeyboardInterrupt, SystemExit):
            #raise
        # except:# KeyboardInterrupt: #:# TypeError:#
        #   print('analysis failed: (-)')# p-value (MI): %.2f, \t bayes factor: %.2fg+/-%.2fg'%(result['status']['MI_p_value'],result['status']['Bayes_factor'][0,0],result['status']['Bayes_factor'][0,1]))
        #   #result['fields']['nModes'] = -1

        return result#,sampler
    


    def prepare_activity(self):

        key_S = 'spikes' if self.para['modes']['activity']=='spikes' else 'S'
        
        
        activity = {}
        activity['S'] = self.S[self.behavior['active']]

        ### calculate firing rate
        activity['firingrate'], _, activity['spikes'] = get_firingrate(activity['S'],f=self.para['f'],sd_r=self.para['Ca_thr'])

        activity['s'] = activity[key_S]
        
        ## obtain quantized firing rate for MI calculation
        if self.para['modes']['info'] == 'MI' and activity['firingrate']>0:
            activity['qtl'] = sp.ndimage.gaussian_filter(activity['s'].astype('float')*self.para['f'],self.para['sigma'])
            # activity['qtl'] = activity['qtl'][self.behavior['active']]
            qtls = np.quantile(activity['qtl'][activity['qtl']>0],np.linspace(0,1,self.para['qtl_steps']+1))
            activity['qtl'] = np.count_nonzero(activity['qtl'][:,np.newaxis]>=qtls[np.newaxis,1:-1],1)
        

        ## obtain trial-specific activity
        activity['trials'] = {}
        activity['trials_firingmap'] = np.zeros((self.behavior['trials']['ct'],self.para['nbin']))    ## preallocate

        for t in range(self.behavior['trials']['ct']):
            activity['trials'][t] = {}
            activity['trials'][t]['s'] = activity['s'][self.behavior['trials']['start'][t]:self.behavior['trials']['start'][t+1]]#gauss_smooth(active['S'][self.behavior['trials']['frame'][t]:self.behavior['trials']['frame'][t+1]]*self.para['f'],self.para['f']);    ## should be quartiles?!
            
            ## prepare quantiles, if MI is to be calculated
            if self.para['modes']['info'] == 'MI':
                activity['trials'][t]['qtl'] = activity['qtl'][self.behavior['trials']['start'][t]:self.behavior['trials']['start'][t+1]];    ## should be quartiles?!

            if self.para['modes']['activity'] == 'spikes':
                activity['trials'][t]['spike_times'] = np.where(activity['trials'][t]['s'])
                activity['trials'][t]['spikes'] = activity['trials'][t]['s'][activity['trials'][t]['spike_times']]
                activity['trials'][t]['ISI'] = np.diff(activity['trials'][t]['spike_times'])

            activity['trials'][t]['rate'] = activity['trials'][t]['s'].sum()/(self.behavior['trials']['nFrames'][t]/self.para['f'])

            if activity['trials'][t]['rate'] > 0:
                print('trial ',t,activity['trials'][t]['s'])
                activity['trials_firingmap'][t,:] = get_firingmap(
                    activity['trials'][t]['s'],
                    self.behavior['trials']['binpos'][t],
                    self.behavior['trials']['dwelltime'][t,:],
                    self.para['nbin']
                )#/activity['trials'][t]['rate']
                print('done')
        self.activity = activity
        


    def test_MI(self):

        shuffle_peaks = False
        S_key = 'qtl' if self.para['modes']['info'] == 'MI' else 's'

        # initialize computation and results structures
        MI = {'MI_p_value':np.NaN,'MI_value':np.NaN,'MI_z_score':np.NaN,
            'Isec_p_value':np.NaN,'Isec_value':np.NaN,'Isec_z_score':np.NaN}
        MI_rand_distr = np.zeros(self.para['repnum'])*np.NaN
        Isec_rand_distr = np.zeros(self.para['repnum'])*np.NaN

        ### first, get actual MI value
        frate = gauss_smooth(self.activity['s']*self.para['f'],self.para['sigma'])

        MI['MI_value'] = self.get_info_value(self.activity[S_key],self.behavior['dwelltime_coarse'],mode='MI')
        MI['Isec_value'] = self.get_info_value(frate,self.behavior['dwelltime_coarse'],mode='Isec')

        ### shuffle according to specified mode
        trial_ct = self.behavior['trials']['ct']
        for L in range(self.para['repnum']):

            ## shift single trials to destroy characteristic timescale
            if self.para['modes']['shuffle'] == 'shuffle_trials':

                ## trial shuffling
                trials = np.random.permutation(trial_ct)

                shuffled_activity_qtl = np.roll(
                    np.hstack(
                        [
                        np.roll(self.activity['trials'][t][S_key],int(random.random()*self.behavior['trials']['nFrames'][t])) 
                        for t in trials
                        ]
                    ),  
                int(random.random()*self.behavior['nFrames'])
                )

            #shuffled_activity_S = np.roll(np.hstack([np.roll(trials_S[t]['S'],int(random.random()*self.behavior['trials']['T'][t])) for t in trials]),int(random.random()*self.behavior['T']))

            #shuffled_activity_S = gauss_smooth(shuffled_activity_S*self.para['f'],self.para['sigma'])

            elif self.para['modes']['shuffle'] == 'shuffle_global':
                # if self.para['modes']['activity'] == 'spikes':
                    # shuffled_activity = shuffling('dithershift',shuffle_peaks,spike_times=spike_times,spikes=spikes,T=self.behavior['nFrames'],ISI=ISI,w=2*self.para['f'])
                # else:
                shuffled_activity = shuffling('shift',shuffle_peaks,spike_train=self.activity[S_key])

            elif self.para['modes']['shuffle'] == 'randomize':
                shuffled_activity = self.activity[S_key][np.random.permutation(len(self.activity[S_key]))]

            #t_start_info = time.time()
            MI_rand_distr[L] = self.get_info_value(shuffled_activity_qtl,self.behavior['dwelltime_coarse'],mode='MI')
            #Isec_rand_distr[L] = self.get_info_value(shuffled_activity_S,norm_dwelltime_coarse,mode='Isec')

            #print('info calc: time taken: %5.3g'%(time.time()-t_start_info))
            #print('shuffle: time taken: %5.3g'%(time.time()-t_start_shuffle))

        MI_mean = np.nanmean(MI_rand_distr)
        MI_std = np.nanstd(MI_rand_distr)
        MI['MI_z_score'] = (MI['MI_value'] - MI_mean)/MI_std
        if MI['MI_value'] > MI_rand_distr.max():
            MI['MI_p_value'] = 1e-10#1/self.para['repnum']
        else:
            x,y = ecdf(MI_rand_distr)
            min_idx = np.argmin(abs(x-MI['MI_value']))
            MI['MI_p_value'] = 1 - y[min_idx]


        #Isec_mean = np.nanmean(Isec_rand_distr)
        #Isec_std = np.nanstd(Isec_rand_distr)
        #MI['Isec_z_score'] = (MI['Isec_value'] - Isec_mean)/Isec_std
        #if MI['Isec_value'] > Isec_rand_distr.max():
        #MI['Isec_p_value'] = 1e-10#1/self.para['repnum']
        #else:
        #x,y = ecdf(Isec_rand_distr)
        #min_idx = np.argmin(abs(x-MI['Isec_value']))
        #MI['Isec_p_value'] = 1 - y[min_idx]



        #Isec_mean = np.nanmean(Isec_rand_distr,0)
        #Isec_std = np.nanstd(Isec_rand_distr,0)
        #p_val = np.zeros(self.para['nbin_coarse'])
        #if ~np.any(MI['Isec_value'] > (Isec_mean+Isec_std)):
        #MI['Isec_p_value'][:] = 1#1/self.para['repnum']
        #else:
        #for i in range(self.para['nbin_coarse']):
            #if MI['Isec_value'][i] > Isec_rand_distr[:,i].max():
            #p_val[i] = 1/self.para['repnum']
            #else:
            #x,y = ecdf(Isec_rand_distr[:,i])
            #min_idx = np.argmin(abs(x-MI['Isec_value'][i]))
            #p_val[i] = 1 - y[min_idx]
        #p_val.sort()
        ##print(p_val)
        #MI['Isec_p_value'] = np.exp(np.log(p_val[:5]).mean())
        ##MI['Isec_z_score'] = np.max((MI['Isec_value'] - Isec_mean)/Isec_std)


        #plt.figure()
        #plt.subplot(211)
        #plt.plot(MI['Isec_value'])
        #plt.errorbar(np.arange(self.para['nbin_coarse']),Isec_mean,Isec_std)
        #plt.subplot(212)
        #plt.plot(MI['Isec_p_value'])
        #plt.yscale('log')
        #plt.show(block=False)


        #print('p_value: %7.5g'%MI['MI_p_value'])

        #if pl['bool']:
        #plt.figure()
        #plt.hist(rand_distr)
        #plt.plot(MI['MI_value'],0,'kx')
        #plt.show(block=True)

        return MI
  

    def get_info_value(self,activity,dwelltime,mode='MI'):

        if mode == 'MI':
            p_joint = self.get_p_joint(activity)   ## need activity trace
            return get_MI(p_joint,dwelltime/dwelltime.sum(),self.para['qtl_weight'])

        elif mode == 'Isec':
            fmap = get_firingmap(activity,self.behavior['binpos_coarse_active'],dwelltime,nbin=self.para['nbin_coarse'])
            Isec_arr = dwelltime/dwelltime.sum()*(fmap/np.nanmean(fmap))*np.log2(fmap/np.nanmean(fmap))

            #return np.nansum(Isec_arr[-self.para['nbin']//2:])
            return np.nansum(Isec_arr)


    def get_p_joint(self,activity):

        ### need as input:
        ### - activity (quantiled or something)
        ### - behavior trace
        p_joint = np.zeros((self.para['nbin_coarse'],self.para['qtl_steps']))

        for q in range(self.para['qtl_steps']):
            for (x,ct) in Counter(self.behavior['binpos_coarse_active'][activity==q]).items():
                p_joint[x,q] = ct;
        p_joint = p_joint/p_joint.sum();    ## normalize
        return p_joint
    


    def get_firingstats_from_trials(self,trials_firingmap,trials=None,complete=True):

        '''
            construct firing rate map from bootstrapping over (normalized) trial firing maps
        '''

        if trials is None:
            trials = np.arange(self.behavior['trials']['ct'])

        #trials_firingmap = trials_firingmap[trials,:]
        dwelltime = self.behavior['trials']['dwelltime'][trials,:]

        
        firingstats = {}
        firingmap_bs = np.zeros((self.para['N_bs'],self.para['nbin']))

        base_sample = np.random.randint(0,len(trials),(self.para['N_bs'],len(trials)))

        for L in range(self.para['N_bs']):
            #dwelltime = self.behavior['trials']['dwelltime'][base_sample[L,:],:].sum(0)
            firingmap_bs[L,:] = np.nanmean(trials_firingmap[trials[base_sample[L,:]],:],0)#/dwelltime
            #mask = (dwelltime==0)
            #firingmap_bs[mask,L] = 0

            #firingmap_bs[:,L] = np.nanmean(trials_firingmap[base_sample[L,:],:]/ self.behavior['trials']['dwelltime'][base_sample[L,:],:],0)
        firingstats['map'] = np.nanmean(firingmap_bs,0)

        if complete:
            ## parameters of gamma distribution can be directly inferred from mean and std
            firingstats['std'] = np.nanstd(firingmap_bs,0)
            firingstats['std'][firingstats['std']==0] = np.nanmean(firingstats['std'])
            prc = [2.5,97.5]
            firingstats['CI'] = np.nanpercentile(firingmap_bs,prc,0);   ## width of gaussian - from 1-SD confidence interval

            ### fit linear dependence of noise on amplitude (with 0 noise at fr=0)
            firingstats['parNoise'] = jackknife(firingstats['map'],firingstats['std'])
        
            if self.para['plt_theory_bool'] and self.para['plt_bool']:
                self.plt_model_selection(firingmap_bs.T,firingstats,trials_firingmap)

        firingstats['map'] = np.maximum(firingstats['map'],1/dwelltime.sum(0))#1/(self.para['nbin'])     ## set 0 firing rates to lowest possible (0 leads to problems in model, as 0 noise, thus likelihood = 0)
        firingstats['map'][dwelltime.sum(0)<0.2] = np.NaN#1/(self.para['nbin']*self.behavior['T'])
        ### estimate noise of model
        return firingstats


    def get_correlated_trials(self,result,smooth=None):

        ## check reliability
        corr = corr0(gauss_smooth(result['firingstats']['trial_map'],smooth=(0,smooth*self.para['nbin']/self.para['L_track'])))

        # corr = np.corrcoef(gauss_smooth(result['firingstats']['trial_map'],smooth=(0,smooth*self.para['nbin']/self.para['L_track'])))
        # corr = sstats.spearmanr(gauss_smooth(result['firingstats']['trial_map'],smooth=(0,smooth*self.para['nbin']/self.para['L_track'])),axis=1)[0]

        #result['firingstats']['trial_map'] = gauss_smooth(result['firingstats']['trial_map'],(0,2))
        corr[np.isnan(corr)] = 0
        ordered_corr,res_order,res_linkage = compute_serial_matrix(-(corr-1),'average')
        cluster_idx = sp.cluster.hierarchy.cut_tree(res_linkage,height=0.5)
        _, c_counts = np.unique(cluster_idx,return_counts=True)
        c_trial = np.where((c_counts>self.para['trials_min_count']) & (c_counts>(self.para['trials_min_fraction']*self.behavior['trials']['ct'])))[0]
        # print('cluster',corr)
        for (i,t) in enumerate(c_trial):
            fmap = gauss_smooth(np.nanmean(result['firingstats']['trial_map'][cluster_idx.T[0]==t,:],0),2)
            # baseline = np.percentile(fmap[fmap>0],20)
            baseline = np.nanpercentile(fmap[fmap>0],30)
            fmap2 = np.copy(fmap)
            fmap2 -= baseline
            fmap2 *= -1*(fmap2 <= 0)

            Ns_baseline = (fmap2>0).sum()
            noise = np.sqrt((fmap2**2).sum()/(Ns_baseline*(1-2/np.pi)))
            if (fmap>(baseline+4*noise)).sum()>5:
                result['firingstats']['trial_field'][i,:] = (cluster_idx.T==t)

        testing = False
        if testing and self.para['plt_bool']:
            plt.figure()
            plt.subplot(121)
            plt.pcolormesh(corr[res_order,:][:,res_order],cmap='jet')
            plt.clim([0,1])
            plt.colorbar()
            plt.subplot(122)
            corr = sstats.spearmanr(gauss_smooth(result['firingstats']['trial_map'],smooth=(0,smooth*self.para['nbin']/self.para['L_track'])),axis=1)[0]
            # print(corr)
            ordered_corr,res_order,res_linkage = compute_serial_matrix(-(corr-1),'average')
            # Z = sp.cluster.hierarchy.linkage(-(corr-1),method='average')
            # print(Z)
            plt.pcolormesh(corr[res_order,:][:,res_order],cmap='jet')
            plt.clim([0,1])
            plt.colorbar()
            plt.show(block=False)
            plt.figure()
            color_t = plt.cm.rainbow(np.linspace(0,1,self.behavior['trials']['ct']))
            for i,r in enumerate(res_order):
                if i<25:
                    col = color_t[int(res_linkage[i,3]-2)]
                    plt.subplot(5,5,i+1)
                    plt.plot(np.linspace(0,self.para['L_track'],self.para['nbin']),gauss_smooth(result['firingstats']['trial_map'][r,:],smooth*self.para['nbin']/self.para['L_track']),color=col)
                    plt.ylim([0,20])
                    plt.title('trial # %d'%r)
            plt.show(block=False)
        return result


    def run_nestedSampling(self,firingstats,firingmap,f):

        hbm = HierarchicalBayesModel(firingmap,self.para['bin_array'],firingstats['parNoise'],f)

        ### test models with 0 vs 1 fields
        paramnames = [self.para['names'][0]]
        for ff in range(f):
            paramnames.extend(self.para['names'][1:]*f)

        ## hand over functions for sampler
        my_prior_transform = hbm.transform_p
        my_likelihood = hbm.set_logl_func()


        # sampler = DynamicNestedSampler(my_likelihood,my_prior_transform,4)
        # print(sampler)

        sampler = ultranest.ReactiveNestedSampler(paramnames, my_likelihood, my_prior_transform,wrapped_params=hbm.pTC['wrap'],vectorized=True,num_bootstraps=20)#,log_dir='/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Programs/PC_analysis/test_ultra')   ## set up sampler...
        num_samples = 400
        if f>1:
            sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=3)#, adaptive_nsteps='move-distance')
            num_samples = 200

        sampling_result = sampler.run(min_num_live_points=num_samples,max_iters=10000,cluster_num_live_points=20,max_num_improvement_loops=3,show_status=False,viz_callback=False)  ## ... and run it #max_ncalls=500000,(f+1)*100,
        #t_end = time.time()
        #print('nested sampler done, time: %5.3g'%(t_end-t_start))

        Z = [sampling_result['logz'],sampling_result['logzerr']]    ## store evidences
        field = {'Bayes_factor':np.zeros(2)*np.NaN}
        if f > 0:

            fields_tmp = self.detect_modes_from_posterior(sampler)
            if len(fields_tmp)>0:

                for key in fields_tmp.keys():
                    field[key] = fields_tmp[key]
                field['Bayes_factor'][0] = Z[0]-self.tmp['Z'][0]
                field['Bayes_factor'][1] = np.sqrt(Z[1]**2 + self.tmp['Z'][1]**2)
            else:
                field['Bayes_factor'] = np.zeros(2)*np.NaN
        self.tmp['Z'] = Z
        #if f==2:
            ##try:
            #if np.any(~np.isnan(fields_tmp['posterior_mass'])):

            #if np.any(~np.isnan(result['fields']['posterior_mass'])):
                #f_major = np.nanargmax(result['fields']['posterior_mass'])
                #theta_major = result['fields']['parameter'][f_major,3,0]
                #dTheta = np.abs(np.mod(theta_major-fields_tmp['parameter'][:,3,0]+self.para['nbin']/2,self.para['nbin'])-self.para['nbin']/2)
                #if result['fields']['Bayes_factor'][f-1,0] > 0:
                #for key in fields_tmp.keys():
                    #result['fields'][key] = fields_tmp[key]
                #result['fields']['major'] = np.nanargmin(dTheta)
            #else:
                #if result['fields']['Bayes_factor'][f-1,0] > 0:
                #for key in fields_tmp.keys():
                    #result['fields'][key] = fields_tmp[key]
                #result['fields']['major'] = np.NaN

            #print('peaks to compare:')
            #print(result['fields']['parameter'][:,3,0])
            #print(fields_tmp['parameter'][:,3,0])
            #print(dTheta)
            #if np.any(dTheta<(self.para['nbin']/10)):

            #else:
                #print('peak detection for 2-field model was not in line with 1-field model')
                #print(result['fields']['parameter'][:,3,0])
                #print(fields_tmp['parameter'][:,3,0])
                #result['status']['Bayes_factor'][-1,:] = np.NaN
            #return result, sampler
            #except:
            #pass
            #return result, sampler
        #else:

        #if result['status']['Bayes_factor'][f-1,0]<=0:
            #break_it = True

        return field


    def detect_modes_from_posterior(self,sampler,plt_bool=False):
        ### handover of sampled points
        data_tmp = ultranest.netiter.logz_sequence(sampler.root,sampler.pointpile)[0]
        logp_prior = np.log(-0.5*(np.diff(np.exp(data_tmp['logvol'][1:]))+np.diff(np.exp(data_tmp['logvol'][:-1])))) ## calculate prior probabilities (phasespace-slice volume from change in prior-volume (trapezoidal form)

        data = {}
        data['logX'] = np.array(data_tmp['logvol'][1:-1])
        data['logl'] = np.array(data_tmp['logl'][1:-1])
        data['logz'] = np.array(data_tmp['logz'][1:-1])
        data['logp_posterior'] = logp_prior + data['logl'] - data['logz'][-1]   ## normalized posterior weight
        data['samples'] = data_tmp['samples'][1:-1,:]

        if self.para['plt_bool']:
            plt.figure(figsize=(2.5,1.5),dpi=300)
            ## plot weight
            ax1 = plt.subplot(111)
            dZ = np.diff(np.exp(data['logz']))
            ax1.fill_between(data['logX'][1:],dZ/dZ.max(),color=[0.5,0.5,0.5],zorder=0,label='$\Delta Z$')

            w = np.exp(logp_prior)
            ax1.plot(data['logX'],w/w.max(),'r',zorder=5,label='$w$')

            L = np.exp(data['logl'])
            ax1.plot(data['logX'],L/L.max(),'k',zorder=10,label='$\mathcal{L}$')

            ax1.set_yticks([])
            ax1.set_xlabel('ln X')
            ax1.legend(fontsize=8,loc='lower left')
            plt.tight_layout()
            plt.show(block=False)

            if self.para['plt_sv']:
                pathSv = os.path.join(self.para['pathFigs'],'PC_analysis_NS_contributions.png')
                plt.savefig(pathSv)
                print('Figure saved @ %s'%pathSv)

            print('add colorbar to other plot')

        nPars = data_tmp['samples'].shape[-1]
        nf = int((nPars - 1)/3)

        testing = True
        bins = 2*self.para['nbin']
        offset = self.para['nbin']

        print('data samples:', data['samples'])

        fields = {}
        for f in range(nf):

            #fields[f] = {}
            #fields[f]['nModes'] = 0
            #fields[f]['posterior_mass'] = np.zeros(3)*np.NaN
            #fields[f]['parameter'] = np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN
            #fields[f]['p_x'] = np.zeros((3,self.para['nbin']))*np.NaN

            data['pos_samples'] = np.array(data['samples'][:,3+3*f])
            logp = np.exp(data['logp_posterior'])   ## even though its not logp, but p!!

            ### search for baseline (where whole prior space is sampled)
            x_space = np.linspace(0,self.para['nbin'],11)
            logX_top = -(data['logX'].min())
            logX_bottom = -(data['logX'].max())
            for i in range(10):
                logX_base = (logX_top + logX_bottom)/2
                mask_logX = -data['logX']>logX_base
                cluster_hist = np.histogram(data['pos_samples'][mask_logX],bins=x_space)[0]>0
                if np.mean(cluster_hist) > 0.9:
                    logX_bottom = logX_base
                else:
                    logX_top = logX_base
                i+=1

            post,post_bin = np.histogram(data['pos_samples'],bins=np.linspace(0,self.para['nbin'],bins+1),weights=logp*(np.random.rand(len(logp))<(logp/logp.max())))
            post /= post.sum()

            # construct wrapped and smoothed histogram
            post_cat = np.concatenate([post[-offset:],post,post[:offset]])
            post_smooth = sp.ndimage.gaussian_filter(post,2,mode='wrap')
            post_smooth = np.concatenate([post_smooth[-offset:],post_smooth,post_smooth[:offset]])

            ## find peaks and troughs
            mode_pos, prop = sp.signal.find_peaks(post_smooth,distance=self.para['nbin']/5,height=post_smooth.max()/3)
            mode_pos = mode_pos[(mode_pos>offset) & (mode_pos<(bins+offset))]
            trough_pos, prop = sp.signal.find_peaks(-post_smooth,distance=self.para['nbin']/5)


            if testing and self.para['plt_bool']:
                plt.figure()
                ax = plt.subplot(211)
                #bin_arr = np.linspace(-25,125,bins+2*offset)
                bin_arr = np.linspace(0,bins+2*offset,bins+2*offset)
                ax.bar(bin_arr,post_smooth)
                ax.plot(bin_arr[mode_pos],post_smooth[mode_pos],'ro')
                ax2 = plt.subplot(212)
                ax2.bar(post_bin[:-1],post,width=0.5,facecolor='b',alpha=0.5)
                ax2.plot(post_bin[np.mod(mode_pos-offset,bins)],post[np.mod(mode_pos-offset,bins)],'ro')
                ax2.plot(post_bin[np.mod(trough_pos-offset,bins)],post[np.mod(trough_pos-offset,bins)],'bo')
                plt.show(block=False)

            modes = {}
            #c_ct = 0
            p_mass = np.zeros(len(mode_pos))
            for (i,p) in enumerate(mode_pos):
                try:
                    ## find neighbouring troughs
                    dp = trough_pos-p
                    t_right = p+dp[dp>0].min()
                    t_left = p+dp[dp<0].max()
                except:
                    try:
                        t_left = np.where(post_smooth[:p]<(post_smooth[p]*0.01))[0][-1]
                    except:
                        t_left = 0
                    try:
                        t_right = p+np.where(post_smooth[p:]<(post_smooth[p]*0.01))[0][0]
                    except:
                        pass#nbin+2*offset

                p_mass[i] = post_cat[t_left:t_right].sum()    # obtain probability mass between troughs
                if p_mass[i] > 0.05:
                    modes[i] = {}
                    modes[i]['p_mass'] = p_mass[i]
                    modes[i]['peak'] = post_bin[p-offset]
                    modes[i]['left'] = post_bin[np.mod(t_left-offset,bins)]
                    modes[i]['right'] = post_bin[np.mod(t_right-offset,bins)]
                    #c_ct += 1

                if testing and self.para['plt_bool']:
                    print('peak @x=%.1f'%post_bin[p-offset])
                    print('\ttroughs: [%.1f, %.1f]'%(post_bin[np.mod(t_left-offset,bins)],post_bin[np.mod(t_right-offset,bins)]))
                    print('\tposterior mass: %5.3g'%p_mass[i])

            nsamples = len(logp)
            if testing and self.para['plt_bool']:
                plt.figure()
                plt.subplot(311)
                plt.scatter(data['pos_samples'],-data['logX'],c=np.exp(data['logp_posterior']),marker='.',label='samples')
                plt.plot([0,self.para['nbin']],[logX_base,logX_base],'k--')
                plt.xlabel('field position $\\theta$')
                plt.ylabel('-ln(X)')
                plt.legend(loc='lower right')
                plt.show(block=False)
            
            if len(p_mass)<1:
                return {}

            if np.max(p_mass)>0.05:
                p = np.argmax(p_mass)
                m = modes[p]
                #for (p,m) in enumerate(modes.values()):
                if m['p_mass'] > 0.3 and ((m['p_mass']<p_mass).sum()<3):

                    field = self.define_field(data,logX_base,modes,p,f)
                    field['posterior_mass'] = m['p_mass']
                else:
                    field = {}
            else:
                field = {}
            #plt.show(block=False)

            #print('val: %5.3g, \t (%5.3g,%5.3g)'%(val[c,i],CI[c,i,0],CI[c,i,1]))
        #print('time took (post-process posterior): %5.3g'%(time.time()-t_start))
        #print(fields[f]['parameter'])
        if self.para['plt_bool'] or plt_bool:
            #plt.figure()
            #### plot nsamples
            #### plot likelihood
            #plt.subplot(313)
            #plt.plot(-data['logX'],np.exp(data['logl']))
            #plt.ylabel('likelihood')
            #### plot importance weight
            #plt.subplot(312)
            #plt.plot(-data['logX'],np.exp(data['logp_posterior']))
            #plt.ylabel('posterior weight')
            #### plot evidence
            #plt.subplot(311)
            #plt.plot(-data['logX'],np.exp(data['logz']))
            #plt.ylabel('evidence')
            #plt.show(block=False)

            col_arr = ['tab:blue','tab:orange','tab:green']

            fig = plt.figure(figsize=(7,4),dpi=300)
            ax_NS = plt.axes([0.1,0.11,0.2,0.85])
            #ax_prob = plt.subplot(position=[0.6,0.675,0.35,0.275])
            #ax_center = plt.subplot(position=[0.6,0.375,0.35,0.275])
            ax_phase_1 = plt.axes([0.4,0.11,0.125,0.2])
            ax_phase_2 = plt.axes([0.55,0.11,0.125,0.2])
            ax_phase_3 = plt.axes([0.4,0.335,0.125,0.2])
            ax_hist_1 = plt.axes([0.7,0.11,0.1,0.2])
            ax_hist_2 = plt.axes([0.55,0.335,0.125,0.15])
            ax_hist_3 = plt.axes([0.4,0.56,0.125,0.15])


            ax_NS.scatter(data['pos_samples'],-data['logX'],c=np.exp(data['logp_posterior']),marker='.',label='samples')
            ax_NS.plot([0,self.para['nbin']],[logX_base,logX_base],'k--')
            ax_NS.set_xlabel('field position $\\theta$')
            ax_NS.set_ylabel('-ln(X)')
            ax_NS.legend(loc='lower right')


            if False:
                for c in range(fields[f]['nModes']):
                    #if fields[f]['posterior_mass'][c] > 0.05:
                    #ax_center.plot(logX_arr,blob_center[:,c],color=col_arr[c])
                    #ax_center.fill_between(logX_arr,blob_center_CI[:,0,c],blob_center_CI[:,1,c],facecolor=col_arr[c],alpha=0.5)

                    ax_phase_1.plot(data['samples'][clusters[c_arr[c]]['mask'],2+3*f],data['samples'][clusters[c_arr[c]]['mask'],3+3*f],'k.',markeredgewidth=0,markersize=1)
                    ax_phase_2.plot(data['samples'][clusters[c_arr[c]]['mask'],1+3*f],data['samples'][clusters[c_arr[c]]['mask'],3+3*f],'k.',markeredgewidth=0,markersize=1)
                    ax_phase_3.plot(data['samples'][clusters[c_arr[c]]['mask'],2+3*f],data['samples'][clusters[c_arr[c]]['mask'],1+3*f],'k.',markeredgewidth=0,markersize=1)

                    ax_hist_1.hist(data['samples'][clusters[c_arr[c]]['mask'],3+3*f],np.linspace(0,self.para['nbin'],50),facecolor='k',orientation='horizontal')
                    ax_hist_2.hist(data['samples'][clusters[c_arr[c]]['mask'],1+3*f],np.linspace(0,10,20),facecolor='k')
                    ax_hist_3.hist(data['samples'][clusters[c_arr[c]]['mask'],2+3*f],np.linspace(0,5,20),facecolor='k')

                    #ax_phase.plot(logX_arr,blob_phase_space[:,c],color=col_arr[c],label='mode #%d'%(c+1))
                    #ax_prob.plot(logX_arr,blob_probability_mass[:,c],color=col_arr[c])

                    #if c < 3:
                            #ax_NS.annotate('',(fields[f]['parameter'][c,3,0],logX_top),xycoords='data',xytext=(fields[f]['parameter'][c,3,0]+5,logX_top+2),arrowprops=dict(facecolor=ax_center.lines[-1].get_color(),shrink=0.05))

            nsteps = 5
            logX_arr = np.linspace(logX_top,logX_base,nsteps)
            for (logX,i) in zip(logX_arr,range(nsteps)):
                ax_NS.plot([0,self.para['nbin']],[logX,logX],'--',color=[1,i/(2*nsteps),i/(2*nsteps)],linewidth=0.5)

            #ax_center.set_xticks([])
            #ax_center.set_xlim([logX_base,logX_top])
            #ax_prob.set_xlim([logX_base,logX_top])
            #ax_center.set_ylim([0,self.para['nbin']])
            #ax_center.set_ylabel('$\\theta$')
            #ax_prob.set_ylim([0,1])
            #ax_prob.set_xlabel('-ln(X)')
            #ax_prob.set_ylabel('posterior')

            ax_phase_1.set_xlabel('$\\sigma$')
            ax_phase_1.set_ylabel('$\\theta$')
            ax_phase_2.set_xlabel('$A$')
            ax_phase_3.set_ylabel('$A$')
            ax_phase_2.set_yticks([])
            ax_phase_3.set_xticks([])

            for axx in [ax_hist_1,ax_hist_2,ax_hist_3]:
                plt.setp(axx,xticks=[],yticks=[])
                axx.spines[['top','right','bottom']].set_visible(False)

            #ax_phase_1.set_xticks([])
            #ax_phase.set_xlim([logX_base,logX_top])
            #ax_phase.set_ylim([0,1])
            #ax_phase.set_ylabel('% phase space')
            #ax_phase.legend(loc='upper right')

            if self.para['plt_sv']:
                pathSv = os.path.join(self.para['pathFigs'],'PC_analysis_NS_results.png')
                plt.savefig(pathSv)
                print('Figure saved @ %s'%pathSv)
            plt.show(block=False)


        #if nf > 1:

        #if testing and self.para['plt_bool']:
            #print('detected from nested sampling:')
            #print(fields[0]['parameter'][:,3,0])
            #print(fields[0]['posterior_mass'])
            #print(fields[1]['parameter'][:,3,0])
            #print(fields[1]['posterior_mass'])

        #fields_return = {}
        #fields_return['nModes'] = 0
        #fields_return['posterior_mass'] = np.zeros(3)*np.NaN
        #fields_return['parameter'] = np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN
        #fields_return['p_x'] = np.zeros((3,self.para['nbin']))*np.NaN

        #for f in range(fields[0]['nModes']):
            #p_cluster = fields[0]['posterior_mass'][f]
            #dTheta = np.abs(np.mod(fields[0]['parameter'][f,3,0]-fields[1]['parameter'][:,3,0]+self.para['L_track']/2,self.para['L_track'])-self.para['L_track']/2)
            #if np.any(dTheta<5):  ## take field with larger probability mass to have better sampling
            #f2 = np.nanargmin(dTheta)
            #if fields[0]['posterior_mass'][f] > fields[1]['posterior_mass'][f2]:
                #handover_f = 0
                #f2 = f
            #else:
                #handover_f = 1
                #p_cluster = fields[1]['posterior_mass'][f2]
            #else:
            #handover_f = 0
            #f2 = f

            #if p_cluster>0.3:
            #fields_return['parameter'][fields_return['nModes'],...] = fields[handover_f]['parameter'][f2,...]
            #fields_return['p_x'][fields_return['nModes'],...] = fields[handover_f]['p_x'][f2,...]
            #fields_return['posterior_mass'][fields_return['nModes']] = fields[handover_f]['posterior_mass'][f2]
            #fields_return['nModes'] += 1
            #if fields_return['nModes']>=3:
                #break

        #for f in range(fields[1]['nModes']):
            #if fields_return['nModes']>=3:
            #break

            #dTheta = np.abs(np.mod(fields[1]['parameter'][f,3,0]-fields[0]['parameter'][:,3,0]+self.para['L_track']/2,self.para['L_track'])-self.para['L_track']/2)

            #dTheta2 = np.abs(np.mod(fields[1]['parameter'][f,3,0]-fields_return['parameter'][:,3,0]+self.para['L_track']/2,self.para['L_track'])-self.para['L_track']/2)


            #if (not np.any(dTheta<5)) and (not np.any(dTheta2<5)) and (fields[1]['posterior_mass'][f]>0.3):  ## take field with larger probability mass to have better sampling
            #fields_return['parameter'][fields_return['nModes'],...] = fields[1]['parameter'][f,...]
            #fields_return['p_x'][fields_return['nModes'],...] = fields[1]['p_x'][f,...]
            #fields_return['posterior_mass'][fields_return['nModes']] = fields[1]['posterior_mass'][f]
            #fields_return['nModes'] += 1
        #if testing and self.para['plt_bool']:
            #print(fields_return['parameter'][:,3,0])
            #print(fields_return['posterior_mass'])
        #else:
        return field


    def define_field(self,data,logX_base,modes,p,f):

        field = {}
        field['posterior_mass'] = np.NaN
        field['parameter'] = np.zeros((4,1+len(self.para['CI_arr'])))*np.NaN
        field['p_x'] = np.zeros(self.para['nbin'])*np.NaN
        #fields[f]['posterior_mass'][fields[f]['nModes']]

        logp = np.exp(data['logp_posterior'])
        nsamples = len(logp)
        samples = data['samples']
        mask_mode = np.ones(nsamples,'bool')

        ## calculate further statistics
        for (p2,m2) in enumerate(modes.values()):
            if not (p==p2):
                # obtain samples first
                if m2['left']<m2['right']:
                    mask_mode[(samples[:,3]>m2['left']) & (samples[:,3]<m2['right']) & (-data['logX']>logX_base)] = False
                else:
                    mask_mode[((samples[:,3]>m2['left']) | (samples[:,3]<m2['right'])) & (-data['logX']>logX_base)] = False

        mode_logp = logp[mask_mode]#/posterior_mass
        mode_logp /= mode_logp.sum()#logp.sum()#

        #if testing and self.para['plt_bool']:
        ##plt.figure()
        #plt.subplot(312)
        #plt.scatter(data['pos_samples'][mask_mode],-data['logX'][mask_mode],c=np.exp(data['logp_posterior'][mask_mode]),marker='.',label='samples')
        #plt.plot([0,self.para['nbin']],[logX_base,logX_base],'k--')
        #plt.xlabel('field position $\\theta$')
        #plt.ylabel('-ln(X)')
        #plt.legend(loc='lower right')

        ## obtain parameters
        field['parameter'][0,0] = get_average(samples[mask_mode,0],mode_logp)
        field['parameter'][1,0] = get_average(samples[mask_mode,1+3*f],mode_logp)
        field['parameter'][2,0] = get_average(samples[mask_mode,2+3*f],mode_logp)
        field['parameter'][3,0] = get_average(samples[mask_mode,3+3*f],mode_logp,True,[0,self.para['nbin']])
        #print(field['parameter'][2,0])
        for i in range(4):
            ### get confidence intervals from cdf
            if i==0:
                samples_tmp = samples[mask_mode,0]
            elif i==3:
                samples_tmp = (samples[mask_mode,3+3*f]+self.para['nbin']/2-field['parameter'][3,0])%self.para['nbin']-self.para['nbin']/2        ## shift whole axis such, that peak is in the center, to get proper errorbars
            else:
                samples_tmp = samples[mask_mode,i+3*f]

        x_cdf_posterior, y_cdf_posterior = ecdf(samples_tmp,mode_logp)
        for j in range(len(self.para['CI_arr'])):
            field['parameter'][i,1+j] = x_cdf_posterior[np.where(y_cdf_posterior>=self.para['CI_arr'][j])[0][0]]

        field['p_x'],_ = np.histogram(samples[mask_mode,3],bins=np.linspace(0,self.para['nbin'],self.para['nbin']+1),weights=mode_logp*(np.random.rand(len(mode_logp))<(mode_logp/mode_logp.max())),density=True)
        field['p_x'][field['p_x']<(0.001*field['p_x'].max())] = 0

        field['parameter'][3,0] = field['parameter'][3,0] % self.para['nbin']
        field['parameter'][3,1:] = (field['parameter'][3,0] + field['parameter'][3,1:]) % self.para['nbin']

        ## rescaling to length 100
        field['parameter'][2,:] *= self.para['L_track']/self.para['nbin']
        field['parameter'][3,:] *= self.para['L_track']/self.para['nbin']

        return field
    



    def plt_model_selection(self,fmap_bs,firingstats,trials_fmap):
        print('plot model selection')
        rc('font',size=10)
        rc('axes',labelsize=12)
        rc('xtick',labelsize=8)
        rc('ytick',labelsize=8)

        prc = [15.8,84.2]

        fr_mu = firingstats['map']#gauss_smooth(np.nanmean(fmap_bs,1),2)
        fr_CI = firingstats['CI']
        fr_std = firingstats['std']

        fig = plt.figure(figsize=(7,5),dpi=150)

        ## get data
        # pathDat = os.path.join(self.para['pathSession'],'results_redetect.mat')
        pathDat = self.para['pathSession']
        ld = load_dict_from_hdf5(pathDat)
        # ld = loadmat(pathDat,variable_names=['S','C'])

        C = ld['C'][self.para['n'],:]

        #S_raw = self.S
        S_raw = ld['S'][self.para['n'],:]
        _,S_thr,_ = get_firingrate(S_raw[self.behavior['active']],f=self.para['f'],sd_r=self.para['Ca_thr'])
        if self.para['modes']['activity'] == 'spikes':
            S = S_raw>S_thr
        else:
            S = S_raw

        t_start = 0#
        t_end = 600#
        n_trial = 6

        ax_Ca = plt.axes([0.1,0.75,0.5,0.175])
        add_number(fig,ax_Ca,order=1)
        ax_loc = plt.axes([0.1,0.5,0.5,0.25])
        ax1 = plt.axes([0.6,0.5,0.35,0.25])
        ax2 = plt.axes([0.6,0.26,0.35,0.125])
        add_number(fig,ax2,order=4,offset=[-75,10])
        ax3 = plt.axes([0.1,0.08,0.35,0.225])
        add_number(fig,ax3,order=3)
        ax4 = plt.axes([0.6,0.08,0.35,0.125])
        add_number(fig,ax4,order=5,offset=[-75,10])
        ax_acorr = plt.axes([0.7,0.85,0.2,0.1])
        add_number(fig,ax_acorr,order=2,offset=[-50,10])

        idx_longrun = self.behavior['active']
        t_longrun = self.behavior['time'][idx_longrun]
        t_stop = self.behavior['time'][~idx_longrun]
        ax_Ca.bar(t_stop,np.ones(len(t_stop))*1.2*S_raw.max(),color=[0.9,0.9,0.9],zorder=0)

        ax_Ca.fill_between([self.behavior['trials']['start_t'][n_trial],self.behavior['trials']['start_t'][n_trial+1]],[0,0],[1.2*S_raw.max(),1.2*S_raw.max()],color=[0,0,1,0.2],zorder=1)

        ax_Ca.plot(self.behavior['time'],C,'k',linewidth=0.2)
        ax_Ca.plot(self.behavior['time'],S_raw,'r',linewidth=1)
        ax_Ca.plot([0,self.behavior['time'][-1]],[S_thr,S_thr])
        ax_Ca.set_ylim([0,1.2*S_raw.max()])
        ax_Ca.set_xlim([t_start,t_end])
        ax_Ca.set_xticks([])
        ax_Ca.set_ylabel('Ca$^{2+}$')
        ax_Ca.set_yticks([])


        ax_loc.plot(self.behavior['time'],self.behavior['binpos'],'.',color='k',zorder=5,markeredgewidth=0,markersize=1.5)
        idx_active = (S>0) & self.behavior['active']
        idx_inactive = (S>0) & ~self.behavior['active']

        t_active = self.behavior['time'][idx_active]
        pos_active = self.behavior['binpos'][idx_active]
        S_active = S[idx_active]

        t_inactive = self.behavior['time'][idx_inactive]
        pos_inactive = self.behavior['binpos'][idx_inactive]
        S_inactive = S[idx_inactive]
        if self.para['modes']['activity'] == 'spikes':
            ax_loc.scatter(t_active,pos_active,s=3,color='r',zorder=10)
            ax_loc.scatter(t_inactive,pos_inactive,s=3,color='k',zorder=10)
        else:
            ax_loc.scatter(t_active,pos_active,s=(S_active/S.max())**2*10+0.1,color='r',zorder=10)
            ax_loc.scatter(t_inactive,pos_inactive,s=(S_inactive/S.max())**2*10+0.1,color='k',zorder=10)
        ax_loc.bar(t_stop,np.ones(len(t_stop))*self.para['L_track'],width=1/15,color=[0.9,0.9,0.9],zorder=0)
        ax_loc.fill_between([self.behavior['trials']['start_t'][n_trial],self.behavior['trials']['start_t'][n_trial+1]],[0,0],[self.para['L_track'],self.para['L_track']],color=[0,0,1,0.2],zorder=1)

        ax_loc.set_ylim([0,self.para['L_track']])
        ax_loc.set_xlim([t_start,t_end])
        ax_loc.set_xlabel('time [s]')
        ax_loc.set_ylabel('position [bins]')

        nC,T = ld['C'].shape
        n_arr = np.random.randint(0,nC,20)
        lags = 300
        t=np.linspace(0,lags/15,lags+1)
        for n in n_arr:
            acorr = np.zeros(lags+1)
            acorr[0] = 1
            for i in range(1,lags+1):
                acorr[i] = np.corrcoef(ld['C'][n,:-i],ld['C'][n,i:])[0,1]
        #acorr = np.correlate(ld['S'][n,:],ld['S'][n,:],mode='full')[T-1:T+lags]
        #ax_acorr.plot(t,acorr/acorr[0])
        ax_acorr.plot(t,acorr,linewidth=0.5)
        for T in self.behavior['trials']['nFrames']:
            ax_acorr.annotate(xy=(T/self.para['f'],0.5),xytext=(T/self.para['f'],0.9),text='',arrowprops=dict(arrowstyle='->',color='k'))
        ax_acorr.set_xlabel('$\Delta t$ [s]')
        ax_acorr.set_ylabel('corr.')
        ax_acorr.spines['right'].set_visible(False)
        ax_acorr.spines['top'].set_visible(False)

        # i = random.randint(0,self.para['nbin']-1)
        i=72
        #ax1 = plt.subplot(211)
        #for i in range(3):
        #trials = np.where(self.result['firingstats']['trial_field'][i,:])[0]
        #if len(trials)>0:
            #fr_mu_trial = gauss_smooth(np.nanmean(self.result['firingstats']['trial_map'][trials,:],0),2)
            #ax1.barh(self.para['bin_array'],fr_mu_trial,alpha=0.5,height=1,label='$\\bar{\\nu}$')

        ax1.barh(self.para['bin_array'],fr_mu,facecolor='b',alpha=0.2,height=1,label='$\\bar{\\nu}$')
        ax1.barh(self.para['bin_array'][i],fr_mu[i],facecolor='b',height=1)

        # ax1.errorbar(fr_mu,self.para['bin_array'],xerr=fr_CI,ecolor='r',linewidth=0.2,linestyle='',fmt='',label='1 SD confidence')
        Y = trials_fmap/self.behavior['trials']['dwelltime']
        mask = ~np.isnan(Y)
        Y = [y[m] for y, m in zip(Y.T, mask.T)]

        #flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5)
        #h_bp = ax1.boxplot(Y,flierprops=flierprops)#,positions=self.para['bin_array'])
        ax1.set_yticks([])#np.linspace(0,100,6))
        #ax1.set_yticklabels(np.linspace(0,100,6).astype('int'))
        ax1.set_ylim([0,self.para['nbin']])

        ax1.set_xlim([0,np.nanmax(fr_mu[np.isfinite(fr_mu)])*1.2])
        ax1.set_xticks([])
        #ax1.set_xlabel('Ca$^{2+}$-event rate $\\nu$')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        #ax1.set_ylabel('Position on track')
        ax1.legend(title='# trials = %d'%self.behavior['trials']['ct'],loc='lower left',bbox_to_anchor=[0.55,0.025],fontsize=8)#[h_bp['boxes'][0]],['trial data'],

        #ax2 = plt.subplot(426)
        ax2.plot(np.linspace(0,5,101),firingstats['parNoise'][1]+firingstats['parNoise'][0]*np.linspace(0,5,101),'--',color=[0.5,0.5,0.5],label='lq-fit')
        ax2.plot(fr_mu,fr_std,'r.',markersize=1)#,label='$\\sigma(\\nu)$')
        ax2.set_xlim([0,np.nanmax(fr_mu[np.isfinite(fr_mu)])*1.2])
        #ax2.set_xlabel('firing rate $\\nu$')
        ax2.set_xticks([])
        ax2.set_ylim([0,np.nanmax(fr_std)*1.2])
        ax2.set_ylabel('$\\sigma_{\\nu}$')
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax1.bar(self.para['bin_array'][i],fr_mu[i],color='b')

        x_arr = np.linspace(0,fmap_bs[i,:].max()*1.2,self.para['nbin']+1)
        offset = (x_arr[1]-x_arr[0])/2
        act_hist = np.histogram(fmap_bs[i,:],x_arr,density=True)
        ax3.bar(act_hist[1][:-1],act_hist[0],width=x_arr[1]-x_arr[0],color='b',alpha=0.2,label='data (bin %d)'%i)

        alpha, beta = gamma_paras(fr_mu[i],fr_std[i])
        mu, shape = lognorm_paras(fr_mu[i],fr_std[i])

        def gamma_fun(x,alpha,beta):
            return beta**alpha * x**(alpha-1) * np.exp(-beta*x) / sp.special.gamma(alpha)

        ax3.plot(x_arr,sstats.gamma.pdf(x_arr,alpha,0,1/beta),'r',label='fit: $\\Gamma(\\alpha,\\beta)$')
        #ax3.plot(x_arr,gamma_fun(x_arr,alpha,beta),'r',label='fit: $\\Gamma(\\alpha,\\beta)$')

        #D,p = sstats.kstest(fmap_bs[i,:1000],'gamma',args=(alpha,0,1/beta))

        #sstats.gamma.rvs()
        #ax3.plot(x_arr,sstats.lognorm.pdf(x_arr,s=shape,loc=0,scale=np.exp(mu)),'b',label='fit: $lognorm(\\alpha,\\beta)$')
        #ax3.plot(x_arr,sstats.truncnorm.pdf(x_arr,(0-fr_mu[i])/fr_std[i],np.inf,loc=fr_mu[i],scale=fr_std[i]),'g',label='fit: $gauss(\\mu,\\sigma)$')

        ax3.set_xlabel('$\\nu$')
        ax3.set_ylabel('$p_{bs}(\\nu)$')
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)


        ax2.legend(fontsize=8)
        #ax2.set_title("estimating noise")

        ax3.legend(fontsize=8)

        D_KL_gamma = np.zeros(self.para['nbin'])
        D_KL_gauss = np.zeros(self.para['nbin'])
        D_KL_lognorm = np.zeros(self.para['nbin'])

        D_KS_stats = np.zeros(self.para['nbin'])
        p_KS_stats = np.zeros(self.para['nbin'])

        for i in range(self.para['nbin']):
            x_arr = np.linspace(0,fmap_bs[i,:].max()*1.2,self.para['nbin']+1)
            offset = (x_arr[1]-x_arr[0])/2
            act_hist = np.histogram(fmap_bs[i,:],x_arr,density=True)
            alpha, beta = gamma_paras(fr_mu[i],fr_std[i])
            mu, shape = lognorm_paras(fr_mu[i],fr_std[i])

            D_KS_stats[i],p_KS_stats[i] = sstats.kstest(fmap_bs[i,:],'gamma',args=(alpha,offset,1/beta))

        ax4.plot(fr_mu,D_KS_stats,'k.',markersize=1)
        ax4.set_xlim([0,np.nanmax(fr_mu[np.isfinite(fr_mu)])*1.2])
        ax4.set_xlabel('$\\bar{\\nu}$')
        ax4.set_ylabel('$D_{KS}$')
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)

        plt.tight_layout()
        plt.show(block=False)
        if self.para['plt_sv']:
            pathSv = os.path.join(self.para['pathFigs'],'PC_analysis_HBM.png')
            plt.savefig(pathSv)
            print('Figure saved @ %s'%pathSv)