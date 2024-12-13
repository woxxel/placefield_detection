import numpy as np
from matplotlib import pyplot as plt

# from scipy.ndimage import gaussian_filter as gauss_filter
from utils import gauss_smooth as gauss_filter

from HierarchicalBayesInference import HierarchicalBayesModel

class HierarchicalBayesInference_analysis(HierarchicalBayesModel):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        print('ready for analysis!')

    def display_results(self,results=None,groundtruth_fields=None,groundtruth_activation=None,use_dynesty=False):

        # if results is None:
        #     results = self.inference_results
        if use_dynesty:
            mean = np.full((1,self.nParams),np.nan)
            for i,key in enumerate(self.paramNames):
                mean[0,i] = posterior[key]['mean']
        else:
            posterior = self.build_posterior(results,use_dynesty=use_dynesty)
            mean = np.array(results['posterior']['mean'])[np.newaxis,:]

        my_logp = self.set_logp_func()

        # key = 'weighted_samples'
        # # key = 'samples'
        # if key == 'weighted_samples':
        #     idx = np.argmax(my_logp(results['weighted_samples']['points']))
        #     logp_best = my_logp(results['weighted_samples']['points'][idx])
        # else:
        #     idx = np.argmax(my_logp(results['samples']))
        #     logp_best = my_logp(results['samples'][idx])

        # print(logp_best)
        # sample = results[key]['points'][idx]

        # mean = sample[np.newaxis,:]

        params = self.from_p_to_params(mean)
        active_model = my_logp(mean,get_active_model=True)


        if self.N_f>1:
            active_model[1,active_model[-1,...]] = True
            active_model[2,active_model[-1,...]] = True
        # print(active_model)
        if self.N_f > 0:
            theta_vals = {
                'CI': np.zeros((self.N_f,2,self.nSamples)),
                'mean': np.zeros((self.N_f,self.nSamples))
            }
            A_vals = {
                'CI': np.zeros((self.N_f,2,self.nSamples)),
                'mean': np.zeros((self.N_f,self.nSamples))
            }
            for key in self.paramNames:
                if key.startswith('PF'):
                    f = int(key[2])
                    if 'theta' in key:
                        suffix = key.split('__')[-1]
                        if not suffix.isdigit():
                            continue
                        trial = int(key.split('__')[-1])
                        theta_vals['mean'][f-1,trial] = posterior[key]['mean']
                        theta_vals['CI'][f-1,:,trial] = posterior[key]['CI'][[0,-1]]
                    elif 'A' in key:
                        suffix = key.split('__')[-1]
                        if not suffix.isdigit():
                            continue
                        trial = int(key.split('__')[-1])
                        A_vals['mean'][f-1,trial] = posterior[key]['mean']
                        A_vals['CI'][f-1,:,trial] = posterior[key]['CI'][[0,-1]]
        


        nbin = 40
        field_match = np.full(self.N_f,-1,'int')
        if groundtruth_fields:
            for f in range(self.N_f):
                theta = mean[0,self.priors[f'PF{f+1}_theta__mean']['idx']]
                
                for f_truth,field in enumerate(groundtruth_fields['PF']):

                    dTheta = abs(np.mod(theta - field['theta'] + nbin/2.,nbin)-nbin/2.)
                    if dTheta <= 5.:
                        print('match!',dTheta,theta,field['theta'])
                        field_match[f] = f_truth
        print(field_match)
            # 'PF1_theta__mean'
            # for theta in theta_vals['mean']:
            #     dTheta = np.mod(theta - theta_true + nbin/2.,nbin)-nbin/2.
            #     print(dTheta)
        trial_activation__true_positive = np.zeros((self.N_f,self.nSamples),'bool')
        trial_activation__true_negative = np.zeros((self.N_f,self.nSamples),'bool')
        trial_activation__false_positive = np.zeros((self.N_f,self.nSamples),'bool')
        trial_activation__false_negative = np.zeros((self.N_f,self.nSamples),'bool')
        for f in range(self.N_f):
            if field_match[f] >= 0:
                trial_activation__true_positive[f,:] = groundtruth_activation[field_match[f],:] & active_model[f+1,0,:]
                trial_activation__true_negative[f,:] = ~groundtruth_activation[field_match[f],:] & ~active_model[f+1,0,:]
                trial_activation__false_positive[f,:] = ~groundtruth_activation[field_match[f],:] & active_model[f+1,0,:]
                trial_activation__false_negative[f,:] = groundtruth_activation[field_match[f],:] & ~active_model[f+1,0,:]
        
        sensitivity = trial_activation__true_positive.sum(axis=1)/(trial_activation__true_positive.sum(axis=1) + trial_activation__false_negative.sum(axis=1))
        specficity = trial_activation__true_negative.sum(axis=1)/(trial_activation__true_negative.sum(axis=1) + trial_activation__false_positive.sum(axis=1))

        print(f'{sensitivity=}, {specficity=}')


        print(f"A0: {groundtruth_fields['A0']} vs {posterior['A0']['mean']}")
        for f in range(self.N_f):
            if field_match[f] >= 0: 
                for key in ['A','sigma']:
                    print(f"PF{f+1}_{key}: {groundtruth_fields['PF'][field_match[f]][key]} vs {posterior[f'PF{f+1}_{key}']['mean']}")


        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax_theta = fig.add_subplot(222)
        ax_theta_inactive = fig.add_subplot(224)

        # ax_A = fig.add_subplot(224)

        for field in groundtruth_fields['PF']:
            ax.axhline(field['theta'],linestyle='--',color='tab:green')


        col = ['r','g']
        if self.N_f > 0:
            for f,PF in enumerate(params['PF']):
                ax.axhline(mean[0,self.priors[f'PF{f+1}_theta__mean']['idx']],color='k',linestyle='--')

                ax.plot(theta_vals['mean'][f,:],color='r',linestyle='-',linewidth=0.5)
                ax.errorbar(
                    np.where(active_model[f+1,0,...])[0],
                    theta_vals['mean'][f,active_model[f+1,0,...]],
                    np.abs(theta_vals['mean'][f,active_model[f+1,0,...]] - theta_vals['CI'][f,:,active_model[f+1,0,...]].T),
                    color='k')#,marker='o')

                idx_true_positives = np.where(trial_activation__true_positive[f,...])[0]
                # print(idx_true_positives)
                # print(theta_vals['mean'][f,idx_true_positives])
                ax.scatter(idx_true_positives,theta_vals['mean'][f,idx_true_positives],marker='o',c='tab:green',s=40)

                idx_false_negatives = np.where(trial_activation__false_negative[f,...])[0]
                ax.scatter(idx_false_negatives,theta_vals['mean'][f,idx_false_negatives],marker='o',c='tab:orange',s=40,alpha=0.6)

                idx_false_positives = np.where(trial_activation__false_positive[f,...])[0]
                ax.scatter(idx_false_positives,theta_vals['mean'][f,idx_false_positives],marker='o',c='tab:red',s=40)


                ax_theta.plot(posterior[f'PF{f+1}_theta__mean']['x'],posterior[f'PF{f+1}_theta__mean']['p_x'],color=col[f])
                for i in np.where(active_model[f+1,0,...])[0]:
                    ax_theta.plot(np.mod(posterior[f'PF{f+1}_theta__{i}']['x'],self.Nx),posterior[f'PF{f+1}_theta__{i}']['p_x'],color=col[f],linewidth=0.2)
                
                for i in np.where(~active_model[f+1,0,...])[0]:
                    ax_theta_inactive.plot(np.mod(posterior[f'PF{f+1}_theta__{i}']['x'],self.Nx),posterior[f'PF{f+1}_theta__{i}']['p_x'],color=col[f],linewidth=0.2)
                
                # ax_A.plot(posterior[f'PF{f+1}_A__mean']['x'],posterior[f'PF{f+1}_A__mean']['p_x'],color=col[f])
                # for i in np.where(active_model[f+1,0,...])[0]:
                # 	ax_A.plot(posterior[f'PF{f+1}_A__{i}']['x'],posterior[f'PF{f+1}_A__{i}']['p_x'],color=col[f],linewidth=0.2)
        plt.setp(ax,ylim=[0-5,self.Nx+5])
        plt.setp(ax_theta,xlim=[0,self.Nx])
        plt.setp(ax_theta_inactive,xlim=[0,self.Nx])
        plt.show(block=False)


        lw = 1
        fig = plt.figure(figsize=(12,8))
        for trial in range(self.nSamples):
            ax = fig.add_subplot(5,5,trial+1)
            ax.plot(self.N[0,trial,:]/self.dwelltime[0,trial,:],'k:')
            ax.plot(gauss_filter(self.N[0,trial,:]/self.dwelltime[0,trial,:],1),'k')

            if self.N_f > 0:
                isfield = active_model[1,0,trial]
                if self.N_f>1:
                    isfield_2 = active_model[2,0,trial]
                else:
                    isfield_2 = False

                if not isfield and not isfield_2:
                    ax.plot(self.x_arr[0,0,:],self.model_of_tuning_curve(params,None)[0,trial,:],'r--',linewidth=lw)
                if isfield and isfield_2:
                    ax.plot(self.x_arr[0,0,:],self.model_of_tuning_curve(params)[0,trial,:],'r--',linewidth=lw)
                
                if isfield and not isfield_2:
                    ax.plot(self.x_arr[0,0,:],self.model_of_tuning_curve(params,0)[0,trial,:],'r--',linewidth=lw)
                if not isfield and isfield_2:
                    ax.plot(self.x_arr[0,0,:],self.model_of_tuning_curve(params,1)[0,trial,:],'r--',linewidth=lw)
            else:
                ax.plot(self.x_arr[0,0,:],self.model_of_tuning_curve(params,None)[0,trial,:],'r--',linewidth=lw)

            
            ax.set_ylim([0,30])

        plt.show(block=False)
        