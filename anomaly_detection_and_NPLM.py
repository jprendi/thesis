'''
scripts to run the NPLM analysis in combo with the anomaly detection algorithm
(for dijet invariant mass)
'''

from scripts.get_2Dhistogrids import create_reweight_maps as reweight
import matplotlib.pyplot as plt
import numpy as np
from scripts.dataset import inject_signal
from model_data_sculpting import sculpt_study
from scripts.physics import invariant_mass
from scripts.NPLM_notebooky import run_NPLM

def anomaly_based_NPLM(sigkey, sig_injection, inject, t_obs_throws, plot1, reconstruct_plot, loss_plot):
    '''
    this function gives us the resulting test statistics of a given NPLM test!

    input:
    - sigkey (string): the type of BSM signal we are interested in probing
    - sig_injection (float): how much signal we want to inject to the ZeroBias dataset
    - t_obs_throws (int): how many times we want to run NPLM
    - plot1 (bool): whether we want to plot the distributions before applying NPLM on it
    - reconstruct_plot (bool): whether we want to plot the reconstruction plot that comes from NPLM
    - loss_plot (bool): whether we want to plot the loss of NPLM

    output:
    - throws (float, list): returns the observed values of the test statistics
    '''

    RW = reweight(sigkey)
    NGT = RW.top_NGT
    ZBT = RW.top_ZBT
    NGT_weight = RW.NGT_weights
    ZB_all = RW.sculpts.ZeroBias_data
    sig_all = RW.sculpts.Signal_data

    sig_filt = sig_all[(sig_all[:, 63] > 30) & (sig_all[:, 66] > 30)]
    ZB_filt = ZB_all[(ZB_all[:, 63] > 30) & (ZB_all[:, 66] > 30)]

    ZB_score_filt = RW.sculpts.ZeroBias_anomaly_score[(ZB_all[:, 63] > 30) & (ZB_all[:, 66] > 30)]
    sig_score_filt = RW.sculpts.Signal_anomaly_score[(sig_all[:, 63] > 30) & (sig_all[:, 66] > 30)]

    if inject:
        new_dataset, labels, dataset_indv, sizes_indv, n1, n2 = inject_signal(ZB_filt, sig_filt, percentage=sig_injection)
        new_dataset_anom_score = np.concatenate((ZB_score_filt[n1], sig_score_filt[n2]))
        inj_signal = sculpt_study.select_anomalous_datapoints(new_dataset_anom_score, new_dataset, area='top', percentage=0.05)
    else: 
        inj_signal = sculpt_study.select_anomalous_datapoints(RW.sculpts.ZeroBias_anomaly_score, RW.sculpts.ZeroBias_data, area='top', percentage=0.05)

    inj_sig_invmass = invariant_mass(inj_signal, type='jet')

    throws = []
    for i in range(0,t_obs_throws):
        np.random.seed(i)
        featureData =  np.random.choice(inj_sig_invmass, 20000, replace=False)
        NP_try = run_NPLM(NGT=NGT, ZBT=featureData.reshape((-1,1)), NG_weights=NGT_weight, plot=plot1, plot_reconstructions=reconstruct_plot,
                 plot_loss=loss_plot)
        throws.append(NP_try)

    return throws