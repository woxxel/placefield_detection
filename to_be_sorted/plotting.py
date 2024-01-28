import numpy as np
import scipy.stats as sstats
import matplotlib.pyplot as plt

from .utils import gamma_paras
from .HierarchicalBayesInference import *


def plot_theory():
    nbin = 100
    x_arr = np.linspace(0,nbin,1001)

    parNoise = [0.4,0.]
    A_0 = 0.01
    A = 0.03
    std = 6.
    theta = 63.
    
    hbm = HierarchicalBayesModel(np.random.rand(1001),x_arr,parNoise,1)
    TC = hbm.TC(np.array([A_0,A,std,theta]))

    plt.figure(figsize=(3,2),dpi=150)
    ax = plt.axes([0.15,0.25,0.8,0.7])
    # plt.bar(np.linspace(0,self.para['nbin'],self.para['nbin']),fr_mu,color='b',alpha=0.2,width=1,label='$\\bar{\\nu}$')
    ax.plot(x_arr,TC,'k',label='tuning curve model TC($x$;$A_0$,A,$\\sigma$,$\\theta$)')

    y_arr = np.linspace(0,0.1,1001)
    x_offset = 10
    alpha, beta = gamma_paras(A_0,A_0/2)
    x1 = sstats.gamma.pdf(y_arr,alpha,0,1/beta)
    x1 = -10*x1/x1.max()+x_offset
    x2 = x_offset*np.ones(1001)
    ax.plot(x_offset,A_0,'ko')
    ax.fill_betweenx(y_arr,x1,x2,facecolor='b',alpha=0.2,edgecolor=None)

    idx = 550
    x_offset = x_arr[idx]
    plt.plot(x_offset,TC[idx],'ko')

    alpha, beta = gamma_paras(TC[idx],TC[idx]/2)
    x1 = sstats.gamma.pdf(y_arr,alpha,0,1/beta)
    x1 = -10*x1/x1.max()+x_offset
    x2 = x_offset*np.ones(1001)
    ax.fill_betweenx(y_arr,x1,x2,facecolor='b',alpha=0.2,edgecolor=None,label='assumed noise')

    ### add text to show parameters
    plt_text = True
    if plt_text:
        ax.annotate("", xy=(theta+3*std, A_0), xytext=(theta+3*std, A_0+A),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
        ax.text(theta+3*std+2,A_0+A/2,'A')

        ax.annotate("", xy=(theta, A_0*0.9), xytext=(theta+2*std, A_0*0.9),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
        ax.text(theta+2,A_0*0.3,'$2\\sigma$')

        ax.annotate("", xy=(90, 0), xytext=(90, A_0),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
        ax.text(92,A_0/3,'$A_0$')

        ax.annotate("", xy=(theta, 0), xytext=(theta,A_0+A),
            arrowprops=dict(arrowstyle="-"))
        ax.text(theta,A_0+A*1.1,'$\\theta$')

    ax.set_xlabel('Position $x$ [bins]')
    ax.set_ylabel('Ca$^{2+}$ event rate')
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.legend(loc='upper left')
    ax.set_ylim([0,0.055])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show(block=False)

    # if self.para['plt_sv']:
    #     pathSv = os.path.join(self.para['pathFigs'],'PC_analysis_HBM_model.png')
    #     plt.savefig(pathSv)
    #     print('Figure saved @ %s'%pathSv)