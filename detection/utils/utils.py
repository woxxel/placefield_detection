''' contains various useful program snippets:

  get_nFolder   get number of folders in path
  pathcat       attach strings to create proper paths
  calculate_hsm          half sampling mode to obtain baseline


'''

import os, cmath
import scipy as sp
import scipy.stats as sstats
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform


def get_nPaths(path,pathStr):

  paths = []
  nF = 0
  for file in os.listdir(path):
    if file.startswith(pathStr):
      paths.append(os.path.join(path,file))
      nF+=1

  return nF, paths


def find_modes(data,axis=None,sort_it=True):

  if axis is not None:
    def fnc(x): return find_modes(x,sort_it=sort_it)
    dataMode = np.apply_along_axis(fnc, axis, data)
  else:
    data = data[np.isfinite(data)]
    if sort_it:
      data = np.sort(data)

    dataMode = calculate_hsm(data)

  return dataMode



def calculate_hsm(data,sort_it=True):
  ### adapted from caiman
  ### Robust estimator of the mode of a data set using the half-sample mode.
  ### versionadded: 1.0.3

  ### Create the function that we can use for the half-sample mode
  ### sorting done as first step, if not specified else

  data = data[np.isfinite(data)]
  if data.size == 0:
    return np.NaN
  if np.all(data == data[0]):
    return data[0]
  
  if sort_it:
    data = np.sort(data)

  if data.size == 1:
    return data[0]
  elif data.size == 2:
    return data.mean()
  elif data.size == 3:
    i1 = data[1] - data[0]
    i2 = data[2] - data[1]
    if i1 < i2:
      return data[:2].mean()
    elif i2 > i1:
      return data[1:].mean()
    else:
      return data[1]
  else:
    wMin = np.inf
    N = data.size//2 + data.size % 2
    for i in range(N):
      w = data[i + N - 1] - data[i]
      if w < wMin:
        wMin = w
        j = i
    return calculate_hsm(data[j:j + N])


def bootstrap_data(fun,data,N_bs):
  ## data:    data to be bootstrapped over
  ## fun:     function to be applied to data "f(data)". needs to return calculated parameters in first return statement
  ## N_bs:    number of bootstrap-samples
  N_data = data.shape[0]
  single=False
  try:
      pars,_ = fun(data)
      par = np.zeros(np.append(N_bs,np.array(pars).shape))*np.NaN
  except:
      pars = fun(data)
      single = True
      par = np.zeros((N_bs,1))*np.NaN

  samples = np.random.randint(0,N_data,(N_bs,N_data))

  for i in range(N_bs):
    data_bs = data[samples[i,:],...]    ### get bootstrap sample
    if single:
        par[i,...] = fun(data_bs)     ### obtain parameters from function "fun"
    else:
        par[i,...],p_cov = fun(data_bs)     ### obtain parameters from function "fun"

  return par.mean(0), par.std(0)



def ecdf(x,p=None):

  if type(p)==np.ndarray:
    #assert abs(1-p.sum()) < 10**(-2), 'probability is not normalized, sum(p) = %5.3g'%p.sum()
    #if abs(1-p.sum()) < 10**(-2):
    p /= p.sum()
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = np.cumsum(p[sort_idx])
  else:
    x = np.sort(x)
    y = np.cumsum(np.ones(x.shape)/x.shape)

  return x,y


def fdr_control(x,alpha):

  if alpha < 1: 
    x[x==0.001] = 10**(-10)
    x_mask = ~np.isnan(x)
    N = x_mask.sum()
    FDR_thr = range(1,N+1)/N*alpha
    x_masked = x[x_mask]
    idxes = np.where(x_mask)[0];
    idx_sorted = np.argsort(x_masked);
    x_sorted = x_masked[idx_sorted]
    FDR_arr = x_sorted<FDR_thr;
    idx_cut = np.where(FDR_arr==False)[0][0]
    FDR_arr[idx_cut:] = False

    classified = np.zeros(len(x)).astype('bool')
    classified[idxes[idx_sorted[FDR_arr]]] = True
  else:
    classified = np.ones(len(x)).astype('bool')
  return classified


def KS_test(dat1,dat2):

  ## p1 & p2 are probability distributions defined on the same kernel

  ### normalize distributions
  #p1 /= p1.sum()
  #p2 /= p2.sum()

  ## generate cumulative density functions from data:
  N1 = dat1.shape[0]
  dat1.sort()

  N2 = dat2.shape[0]
  dat2.sort()

  all_values = np.concatenate((dat1,dat2))
  all_values.sort()

  d1 = np.zeros(N1+N2)
  d2 = np.zeros(N1+N2)

  d1[all_values.searchsorted(dat1)] = 1/N1
  d2[all_values.searchsorted(dat2)] = 1/N2

  plt.figure()
  plt.plot(d1,'k')
  plt.plot(d2,'r')
  plt.show(block=False)

  ## add 0 entries at other data points
  return np.abs(d1-d2).max()


def occupation_measure(data,x_ext,y_ext,nA=[10,10]):

  ## nA:      number zones per row / column (2 entries)

  N = data.shape[0]
  NA_exp = N/(nA[0]*nA[1])
  print(NA_exp)
  A = np.histogram2d(data[:,0],data[:,1],[np.linspace(x_ext[0],x_ext[1],nA[0]+1),np.linspace(y_ext[0],y_ext[1],nA[1]+1)])[0]
  rA = A/NA_exp

  return 1-np.sqrt(np.sum((rA-1)**2))/(nA[0]*nA[1])


def fun_wrapper(fun,x,p):
  if np.isscalar(p):
    return fun(x,p)
  if p.shape[-1] == 2:
    return fun(x,p[...,0],p[...,1])
  if p.shape[-1] == 3:
    return fun(x,p[...,0],p[...,1],p[...,2])
  if p.shape[-1] == 4:
    return fun(x,p[...,0],p[...,1],p[...,2],p[...,3])
  if p.shape[-1] == 5:
    return fun(x,p[...,0],p[...,1],p[...,2],p[...,3],p[...,4])
  if p.shape[-1] == 6:
    return fun(x,p[...,0],p[...,1],p[...,2],p[...,3],p[...,4],p[...,5])
  if p.shape[-1] == 7:
    return fun(x,p[...,0],p[...,1],p[...,2],p[...,3],p[...,4],p[...,5],p[...,6])


def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = np.maximum(0,squareform(dist_mat,checks=False))

    #res_linkage = linkage(flat_dist_mat, method=method,preserve_input=False)
    res_linkage = sp.cluster.hierarchy.linkage(flat_dist_mat,method=method,optimal_ordering=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]

    return seriated_dist, res_order, res_linkage


def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))


def gauss_smooth(X,smooth=None,mode='wrap'):
  if (smooth is None) or not np.any(np.array(smooth)>0):
    return X
  else:
    V = X.copy()
    V[np.isnan(X)] = 0
    VV = sp.ndimage.gaussian_filter(V,smooth,mode=mode)

    W = 0*X.copy()+1
    W[np.isnan(X)] = 0
    WW = sp.ndimage.gaussian_filter(W,smooth,mode=mode)

  return VV/WW


def get_reliability(trial_map,map,field,t):

    ### need: obtain reliability for single (or batch of?) neuron
    ### ability to test / plot

    ## trial_map:       firing maps of single trials in (t,bin)-format
    ## map:             overall firing map of session
    ## field:           parameters of place field

    ## obtain noise level + threshold
    sd_fmap = 2
    nbin = map.shape[0]
    base = np.nanmedian(map)

    if np.all(np.isnan(field[t,...])):
        return

    ## find threshold value of firing map as significant fluctuations over baseline
    fmap_noise = map - np.nanmedian(map)
    fmap_noise = -1*fmap_noise*(fmap_noise<0)
    N_noise = (fmap_noise>0).sum()
    noise = np.sqrt((fmap_noise**2).sum()/(N_noise*(1-2/np.pi)))
    fmap_thr = np.maximum(4,base+sd_fmap*noise)

    ## find bins belonging to place field
    field_bin = int(field[t,3,0])
    field_bin_l = int(field[t,3,0]-field[t,2,0]) % nbin
    field_bin_r = int(field[t,3,0]+field[t,2,0]+1) % nbin
    ## obtain average firing rate within field (slight smoothing)
    fmap = gauss_smooth(trial_map,(0,4))
    if field_bin_l < field_bin_r:
        field_rate_bins = fmap[:,field_bin_l:field_bin_r]
    else:
        field_rate_bins = np.hstack([fmap[:,field_bin_l:],fmap[:,:field_bin_r]])
    # field_rate = np.mean(field_rate_bins,1)
    field_rate = np.max(field_rate_bins,1)

    ## field_trials are such, where average field firing rate is above threshold
    trial_field = field_rate > fmap_thr
    field_max = np.mean(np.max(field_rate_bins[trial_field],1))
    rel = (field_rate > fmap_thr).mean()

    testing = False
    if testing:
        print('max fr: %.2g'%field_max)
        print('reliability: %.2f'%rel)

        plt.figure()
        plt.subplot(211)

        plt.plot(map,'k')
        plt.plot(gauss_smooth(trial_map[-1,:],0),'b--')
        plt.plot(gauss_smooth(trial_map[-1,:],2),'b')

        plt.plot(field[t,3,0],5,'rx')

        plt.plot([0,nbin],[fmap_thr,fmap_thr],'r--')
        # print(field_rate)
        plt.subplot(212)
        plt.plot(field_rate)
        plt.plot([0,field_rate.shape[0]],[fmap_thr,fmap_thr],'r--')
        plt.legend()
        plt.show(block=False)

        plt.figure()
        for tt in range(trial_map.shape[0]):
            plt.subplot(5,6,tt+1)
            plt.plot([field_bin_l,field_bin_l],[0,10],'k--')
            plt.plot([field_bin_r,field_bin_r],[0,10],'k--')
            for i in [1,3]:
                plt.plot(gauss_smooth(trial_map[tt,:],i))
            plt.plot([0,nbin],[fmap_thr,fmap_thr],'r:')
            plt.ylim([0,20])
        plt.show(block=False)

    return rel, field_max, trial_field

def get_firingrate(S,f=15,sd_r=1,Ns_thr=10):
    '''
        calculates the firing rate from a an array of spike probabilities ('S' from CaImAn) 
        by thresholding data according to multiples sd_r of estimated variance

        Ns_thr:
          - minimum number of non-zero entries in S

        
        returns
            - firing rate (in spikes/sec, depending on provided framerate f)
            - calculated spiking threshold
            - array with number of spikes per frame

    '''

    S[S<S.max()*10**(-4)]=0
    Ns = (S>0).sum()
    if Ns<Ns_thr:
        return 0,np.NaN,np.zeros_like(S)
    else:
        # estimate noise level by using median value (assuming most entries are not actual spikes)
        # and obtain values below median to estimate variance from negative half-gaussian distribution
        trace = S[S>0]
        baseline = np.median(trace)
        trace -= baseline
        trace *= -1*(trace <= 0)

        # calculate variance
        Ns_baseline = (trace>0).sum()
        noise = np.sqrt((trace**2).sum()/(Ns_baseline*(1-2/np.pi)))

        # either use provided sd_r, or calculate multiples of variance to ensure 
        # p-value of 10% (including correction for multiple comparisons)
        sd_r = sstats.norm.ppf((1-0.1)**(1/Ns)) if (sd_r==-1) else sd_r
        firing_threshold_adapt = baseline + sd_r*noise

        # number of spikes in each bin is the value multiple above calculated threshold
        N_spikes = np.floor(S / firing_threshold_adapt).sum()

        return N_spikes/(S.shape[0]/f),firing_threshold_adapt,np.floor(S / firing_threshold_adapt)#S > firing_threshold_adapt#


def get_firingmap(S,binpos,dwelltime=None,nbin=None):

    if not nbin:
        nbin = np.max(binpos)+1
    
    ### calculates the firing map
    spike_times = np.where(S)
    spikes = S[spike_times]
    binpos = binpos[spike_times]#.astype('int')

    firingmap = np.zeros(nbin)
    for (p,s) in zip(binpos,spikes):#range(len(binpos)):
        firingmap[p] = firingmap[p]+s

    if not (dwelltime is None):
        firingmap = firingmap/dwelltime
        firingmap[dwelltime==0] = np.NaN

    return firingmap


def get_MI(p_joint,p_x,p_f):

    ### - joint distribution
    ### - behavior distribution
    ### - firing rate distribution
    ### - all normalized, such that sum(p) = 1

    p_tot = p_joint * np.log2(p_joint/(p_x[:,np.newaxis]*p_f[np.newaxis,:]))
    return np.nansum(p_tot)


def get_recurr(status,status_dep):

    nC,nSes = status.shape
    recurr = np.zeros((nSes,nSes))*np.NaN
    for s in range(nSes):#np.where(cluster.sessions['bool'])[0]:
        overlap = status[status[:,s],:].sum(0).astype('float')
        N_ref = status_dep[status[:,s],:].sum(0)
        recurr[s,1:nSes-s] = (overlap/N_ref)[s+1:]

    return recurr


def get_status_arr(cluster,SD=1):

    nSes = cluster.meta['nSes']
    nC = cluster.meta['nC']
    nbin = cluster.para['nbin']
    sig_theta = cluster.stability['all']['mean'][0,2]

    status_arr = ['act','code','stable']

    ds_max = nSes
    status = {}
    status['stable'] = np.zeros((nC,nSes,nSes),'bool')
    for c in np.where(cluster.stats['cluster_bool'])[0]:
        for s in np.where(cluster.sessions['bool'])[0]:
            if cluster.status[c,s,2]:
                for f in np.where(cluster.status_fields[c,s,:])[0]:

                    loc_compare = cluster.fields['location'][c,:s,:,0]
                    loc_compare[~cluster.status_fields[c,:s,:]] = np.NaN
                    dLoc = np.abs(np.mod(cluster.fields['location'][c,s,f,0] - loc_compare +nbin/2,nbin)-nbin/2)

                    stable_s = np.where(dLoc<(SD*sig_theta))[0]
                    if len(stable_s)>0:
                        ds = s - stable_s[-1]
                        status['stable'][c,s,np.unique(s-stable_s)] = True
                        # status['stable'][c,s] = ds

    status['act'] = np.pad(cluster.status[...,1][...,np.newaxis],((0,0),(0,0),(0,nSes-1)),mode='edge')
    status['code'] = np.pad(cluster.status[...,2][...,np.newaxis],((0,0),(0,0),(0,nSes-1)),mode='edge')
    # status['stable'] = status['stable']
    # status['stable'] = status['stable']==1

    status_dep = {}
    status_dep['act'] = np.ones((nC,nSes),'bool')
    status_dep['act'][:,~cluster.sessions['bool']] = False
    status_dep['code'] = np.copy(status['act'][...,0])
    status_dep['stable'] = np.copy(status['code'][...,0])

    return status,status_dep


def get_CI(p,X,Y,alpha=0.05):
    ## what's that for?
    n,k = X.shape

    sigma2 = np.sum((Y-np.dot(X,p))**2) / (n-k)
    C = sigma2 * np.linalg.inv(np.dot(X.T,X))
    se = np.sqrt(np.diag(C))

    sT = sstats.distributions.t.ppf(1.0 - alpha/2.0, n-k)
    CI = sT * se
    return CI




def jackknife(X,Y,W=None,rank=1):

  ## jackknifing a linear fit (with possible weights)
  ## W_i = weights of value-tuples (X_i,Y_i)

    if type(W) == np.ndarray:
        print('weights given (not working)')
        W = np.ones(Y.shape)
        Xw = X * np.sqrt(W[:,np.newaxis])
        Yw = Y * np.sqrt(W)
    else:
        if rank==1:
            Xw = X
        elif rank==2:
            Xw = np.vstack([X,np.ones(len(X))]).T
        Yw = Y

    if len(Xw.shape) < 2:
        Xw = Xw[:,np.newaxis]

    N_data = len(Y);

    fit_jk = np.zeros((N_data,2));
    mask_all = (~np.isnan(Y)) & (~np.isnan(X))

    for i in range(N_data):
        mask = np.copy(mask_all)
        mask[i] = False;
        try:
            if rank==1:
                    fit_jk[i,0] = np.linalg.lstsq(Xw[mask,:],Yw[mask])[0]
            elif rank==2:
                    fit_jk[i,:] = np.linalg.lstsq(Xw[mask,:],Yw[mask])[0]
            #fit_jk[i,1] = 0
        except:
            fit_jk[i,:] = np.NaN

    return np.nanmean(fit_jk,0)




### -------------- lognorm distribution ---------------------
def lognorm_paras(mean,sd):
    shape = np.sqrt(np.log(sd/mean**2+1))
    mu = np.log(mean/np.sqrt(sd/mean**2 + 1))
    return mu, shape

### -------------- Gamma distribution -----------------------
def gamma_paras(mean,SD):
    alpha = (mean/SD)**2
    beta = mean/SD**2
    return alpha, beta


def get_mean_SD(SDs):
    ## isn't that just np.nanstd()?
    mask = np.isfinite(SDs)
    n = mask.sum()
    vars = SDs[mask]**2
    return np.sqrt(1/n**2 * np.sum(vars))


def get_average(x,p,periodic=False,bounds=None):

  #assert abs(1-p.sum()) < 10**(-2), 'probability not normalized, sum(p) = %5.3g'%p.sum()
  if not np.isscalar(p) | (abs(1-np.sum(p)) < 10**(-2)):
      p /= p.sum()
  if periodic:
    assert bounds, 'bounds not specified'
    L = bounds[1]-bounds[0]
    scale = L/(2*np.pi)
    avg = (cmath.phase((p*np.exp(+complex(0,1)*(x-bounds[0])/scale)).sum())*scale) % L + bounds[0]
    # avg = (cmath.phase((p*np.exp(+complex(0,1)*(x-bounds[0])/scale)).sum())*scale + bounds[0]) % L
    # avg = (cmath.phase((p*periodic_to_complex(x,bounds)).sum())*scale) % L + bounds[0]
  else:
    avg = (x*p).sum()
  return avg


def corr0(X,Y=None):

  Y = X if Y is None else Y

  X -= np.nanpercentile(X,20)
  Y -= np.nanpercentile(X,20)

  c_xy = np.zeros((len(X),len(X)))
  for i,x in enumerate(X):
    for j,y in enumerate(Y):
      c_xy[i,j] = (x*y).sum()/np.sqrt((x**2).sum()*(y**2).sum())

  return c_xy


## ------------------ PLOTTING ------------------- ##
def add_number(fig,ax,order=1,offset=None):

    # offset = [-175,50] if offset is None else offset
    offset = [-75,25] if offset is None else offset
    pos = fig.transFigure.transform(plt.get(ax,'position'))
    x = pos[0,0]+offset[0]
    y = pos[1,1]+offset[1]
    ax.text(x=x,y=y,s='%s)'%chr(96+order),ha='center',va='center',transform=None,weight='bold',fontsize=14)

