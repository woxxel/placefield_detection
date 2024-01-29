''' contains several functions, defining data and parameters used for analysis

  set_para

'''
import os
import numpy as np

class detection_parameters:
    
    params = {}

    def __init__(self,nP=0,nbin=100,plt_bool=False,sv_bool=False):

        coarse_factor = int(nbin/20)
        qtl_steps = 4

        # fact = 1 ## factor from path length to bin number

        self.params = self.params | {
            'nbin':nbin,'f':15,
            'bin_array':np.linspace(0,nbin-1,nbin),
            'bin_array_centers':np.linspace(0,nbin,nbin+1)-0.5,
            'coarse_factor':coarse_factor,
            'nbin_coarse':int(nbin/coarse_factor),
            'pxtomu':536/512,
            'L_track':120,

            'SNR_thr': 2,
            'r_value_thr': 0.5,

            'rate_thr':4,
            'width_thr':5,

            'trials_min_count':3,
            'trials_min_fraction':0.2,

            'Ca_thr':0,

            # 't_measures': get_t_measures(mouse),

            'nP':nP,
            'N_bs':10000,
            'repnum':1000,
            'qtl_steps':qtl_steps,'sigma':5,
            'qtl_weight':np.ones(qtl_steps)/qtl_steps,
            'names':['A_0','A','SD','theta'],
            #'CI_arr':[0.001,0.025,0.05,0.159,0.5,0.841,0.95,0.975,0.999],
            'CI_arr':[0.025,0.05,0.95,0.975],

            'plt_bool': plt_bool & (nP==0),
            'plt_theory_bool': True & (nP==0),
            # 'plt_theory_bool':False,
            'plt_sv': sv_bool & (nP==0),

            ### modes, how to perform PC detection
            'modes':{'activity':'calcium',#'spikes',#          ## data provided: 'calcium' or 'spikes'
                    'info':'MI',                   ## information calculated: 'MI', 'Isec' (/second), 'Ispike' (/spike)
                    'shuffle':'shuffle_trials'     ## how to shuffle: 'shuffle_trials', 'shuffle_global', 'randomize'
                    },
        }

    
    def set_paths(self,pathData,pathResults,suffix=''):
        
        self.params = self.params | {
            
            'pathData': pathData,

            'pathResults': pathResults,
            'pathFigures': os.path.join(pathResults,f'figures'),#'/home/wollex/Data/Science/PhD/Thesis/pics/Methods',
            
            ### provide names for distinct result files (needed?)
            'pathResults':              os.path.join(pathResults,'placefields{suffix}.pkl'),
            # 'pathResults_status':       os.path.join(pathResults,'PC_fields%s_status.pkl'%suffix),
            # 'pathResults_fields':       os.path.join(pathResults,'PC_fields%s_para.pkl'%suffix),
            # 'pathResults_firingstats':  os.path.join(pathResults,'PC_fields%s_firingstats.pkl'%suffix),
        }


      

## -----------------------------------------------------------------------------------------------------------------------

  #if nargin == 3:

    #para.t_s = get_t_measures(mouse);
    #para.nSes = length(para.t_s);

    #time_real = false;
  #if time_real
    #t_measures = get_t_measures(mouse);
    #t_mask_m = false(1,t_measures(nSes));
    #for s = 1:nSes-1
      #for sm = s+1:nSes
        #dt = t_measures(sm)-t_measures(s);
        #t_mask_m(dt) = true;
      #end
    #end
    #t_data_m = find(t_mask_m);
    #t_ses = t_measures;
    #t_mask = t_mask_m;
    #t_data = t_data_m;
    #nT = length(t_data);
  #else
    #t_measures = get_t_measures(mouse);
    #t_measures = t_measures(s_offset:s_offset+nSes-1);
##      t_measures = 1:nSes;    ## remove!
##      t_measures
    #t_ses = linspace(1,nSes,nSes);
    #t_data = linspace(1,nSes,nSes);
    #t_mask = true(nSes,1);
    #nT = nSes;
  #end

    # return para


def build_struct_PC_results(nCells,nbin,trial_ct,nStats=5):
    results = {}
    results['status'] = {}
    
    for key in ['MI_value','MI_p_value','MI_z_score',
            'Isec_value','Isec_p_value','Isec_z_score',
            'SNR','r_value']:
        results['status'][key] = np.full(nCells,np.NaN)

    results['fields'] = {
        'parameter':        np.full((nCells,5,4,nStats),np.NaN),          ### (mean,std,CI_low,CI_top)
        'p_x':              np.zeros((nCells,5,nbin)),##sp.sparse.COO((nCells,3,nbin)),#
        'posterior_mass':   np.zeros((nCells,5))*np.NaN,
        'reliability':      np.zeros((nCells,5))*np.NaN,
        'Bayes_factor':     np.zeros((nCells,5,2))*np.NaN,
        'nModes':           np.zeros(nCells).astype('int')
    }

    results['firingstats'] = {
        'rate':         np.full(nCells,np.NaN),
        'map':          np.zeros((nCells,nbin))*np.NaN,
        'std':          np.zeros((nCells,nbin))*np.NaN,
        'CI':           np.zeros((nCells,2,nbin))*np.NaN,
        'trial_map':    np.zeros((nCells,trial_ct,nbin))*np.NaN,
        'trial_field':  np.zeros((nCells,5,trial_ct),'bool'),
        'parNoise':     np.zeros((nCells,2))*np.NaN
    }

    ## if method is called for nCells = 1, collapse data from first dimension
    for field in results.keys():
        for key in results[field].keys():
            results[field][key] = np.squeeze(results[field][key])
    
    return results