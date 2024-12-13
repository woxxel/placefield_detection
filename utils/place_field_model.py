import numpy as np


def model_of_tuning_curve(x, parameter, 
    n_x, n_trials, fields: int | str | None = 'all',
    stacked=False):

    ## build tuning-curve model
    shift = n_x/2.
    
    N_in = parameter['A0'].shape[0]
    
    if not (fields is None) and 'PF' in parameter:
        fields = parameter['PF'] if fields=='all' else [parameter['PF'][fields]]
        n_fields = len(fields)

        mean_model = np.zeros((n_fields+1,N_in,n_trials,x.shape[-1]))
        mean_model[0,...] = parameter['A0'][...,np.newaxis]
                
        for f,field in enumerate(fields):

            mean_model[f+1,...] = field['A'][...,np.newaxis]*np.exp(
                -(np.mod(x - field['theta'][...,np.newaxis] + shift, n_x)-shift)**2/(2*field['sigma'][...,np.newaxis]**2)
            )
        
                
    else:
        mean_model = np.zeros((1,N_in,n_trials,x.shape[-1]))
        mean_model[0,...] = parameter['A0'][...,np.newaxis]
    
    if stacked:
        return mean_model
    else:
        return mean_model.sum(axis=0)


def intensity_model_from_position(x,parameter,n_x,fields=None):
    '''
        function to build tuning-curve model
    '''
    
    shift = n_x/2.
    intensity_model = np.full(len(x),parameter['A0'])

    if not (fields is None) and 'PF' in parameter:
        # fields = parameter['PF'] if fields=='all' else [parameter['PF'][fields]]
        
        
        for f in fields:
            field = parameter['PF'][f]

            intensity_model += field['A']*np.exp(
                -(np.mod(x - field['theta'] + shift, n_x)-shift)**2/(2*field['sigma']**2)
            )
    return intensity_model