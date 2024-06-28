'''
all from this with *very* small adaptions:

https://github.com/GaiaGrosso/NPLM_package/blob/v0.0.6/example_1D/1D_tutorial.ipynb
'''


import sys, os, time, datetime, h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import norm, expon, chi2, uniform, chisquare

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
from tensorflow import linalg as la

from NPLM.NNutils import *
from NPLM.PLOTutils import *
from NPLM.ANALYSISutils import *

from scripts.physics import invariant_mass


def run_NPLM(ZBT, NGT,  NG_weights, bins_code, ymax_code, xlabel_code, seed=5839, 
                total_epochs_tau = 20000,
                plot=False, plot_reconstructions=False,
                plot_loss=False):
    np.random.seed(seed)
    total_epochs_tau = total_epochs_tau
    patience_tau = 1000
    inputsize          = 1
    latentsize         = 4
    n_layers           = 1
    BSMweight_clipping = 9
    layers             = [inputsize]
    for _ in range(n_layers):
        layers.append(latentsize)
    layers.append(1)
    # print(layers)
    hidden_layers      = layers[1:-1]
    BSMarchitecture    = layers
    BSMdf              = compute_df(input_size=BSMarchitecture[0], hidden_layers=BSMarchitecture[1:-1])

    # featureReff, ind = invariant_mass(NGT, type='jet', return_indices = True)
    featureReff = NGT
    featureData = ZBT
    # np.array(invariant_mass(ZBT, type='jet')).reshape((-1,1))

    featureRef = np.array(featureReff).reshape((-1,1))
    feature    = np.concatenate((featureData, featureRef), axis=0)

    # # target                                                                                                                                                         
    targetData  = np.ones_like(featureData)
    targetRef   = np.zeros_like(featureRef)
    weightsData = np.ones_like(featureData)#*ratiooooo

    # weightsRef_1 = np.array(NG_weights)[ind].reshape((-1,1))

    # counts_D, bin_edges_D= np.histogram(featureData,  weights=weightsData, bins=100, range=(0,4000))
    # counts_R, bin_edges_R = np.histogram(featureRef,  weights=weightsRef_1, bins=100, range=(0,4000))

    # integral_D= np.sum(counts_D * np.diff(bin_edges_D))
    # integral_R= np.sum(counts_R * np.diff(bin_edges_R))

    # ratiooo = integral_D/integral_R


    # weightsRef = weightsRef_1*ratiooo#
    N_D = len(featureData)
    N_R = len(featureRef)
    weightsRef  = np.ones_like(featureRef)*N_D*1./N_R


    target_0    = np.concatenate((targetData, targetRef), axis=0)
    weights     = np.concatenate((weightsData, weightsRef), axis=0)
    target      = np.concatenate((target_0, weights), axis=1)

    batch_size  = feature.shape[0]
    inputsize   = feature.shape[1]

    Scale   = 0
    Norm    = 0
    sigma_s = 0.15
    sigma_n = 0.15
    NU0_S     = np.random.normal(loc=Scale, scale=sigma_s, size=1)[0]
    NU0_N     = np.random.normal(loc=Norm, scale=sigma_n, size=1)[0]
    NUR_S     = np.array([0. ])
    NUR_N     = 0
    NU_S      = np.array([0. ])
    NU_N      = 0
    SIGMA_S   = np.array([sigma_s])
    SIGMA_N   = sigma_n 
                                                                                                                                                        
    input_shape = (None, inputsize)                

    weights_file  = './LINEAR_Parametric_EXPO1D_batches_ref40000_bkg40000_sigmaS0.1_-1.0_-0.5_0.5_1.0_patience300_epochs30000_layers1_4_1_actrelu_model_weights9300.h5'

    sigma = weights_file.split('sigmaS', 1)[1]                                                                                                                 
    sigma = float(sigma.split('_', 1)[0])                                                                                                                        
    scale_list=weights_file.split('sigma', 1)[1]                                                                                   
    scale_list=scale_list.split('_patience', 1)[0]                                                                                                               
    scale_list=np.array([float(s) for s in scale_list.split('_')[1:]])*sigma  
    # print(scale_list)
    shape_std = np.std(scale_list)
    activation= weights_file.split('act', 1)[1]
    activation=activation.split('_', 1)[0]
    wclip=None
    if 'wclip' in weights_file:
        wclip= weights_file.split('wclip', 1)[1]
        wclip = float(wclip.split('/', 1)[0])

    layers=weights_file.split('layers', 1)[1]
    layers= layers.split('_act', 1)[0]
    architecture = [int(l) for l in layers.split('_')]

    poly_degree = 0
    if 'LINEAR' in weights_file:
        poly_degree = 1
    elif 'QUADRATIC' in weights_file:
        poly_degree = 2
    elif 'CUBIC' in weights_file:
        poly_degree = 3
    else:
        print('Unrecognized number of degree for polynomial parametric net in file: \n%s'%(weights_file))
        poly_degree = None
        
    scale_parNN = {
        'poly_degree'   : poly_degree,
        'architectures' : [architecture for i in range(poly_degree)],
        'wclips'        : [wclip for i in range(poly_degree)],
        'activation'    : activation,
        'shape_std'     : shape_std,
        'weights_file'  : weights_file
        }

    parNN_list = { 
        'scale': scale_parNN,
        }

    tau = imperfect_model(input_shape=input_shape,
                NU_S=NUR_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S, 
                NU_N=NUR_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,
                correction='SHAPE', shape_dictionary_list=[parNN_list['scale']],
                BSMarchitecture=BSMarchitecture, BSMweight_clipping=BSMweight_clipping, train_f=True, train_nu=True)
    # print(tau.summary())

    tau.compile(loss=imperfect_loss,  optimizer='adam')

    hist_tau = tau.fit(feature, target, batch_size=batch_size, epochs=total_epochs_tau, verbose=False)

    loss_tau  = np.array(hist_tau.history['loss'])
    # test statistic                                                                                                                                                                                                                                                                                                               
    final_loss = loss_tau[-1]
    tau_OBS    = -2*final_loss
    print('tau_OBS: %f'%(tau_OBS))

    if plot_loss == True:
        fig = plt.figure(figsize=(9,6))                                                                                                                                            
        fig.patch.set_facecolor('white') 
        plt.plot(loss_tau, label='tau')
        font=font_manager.FontProperties(family='serif', size=18)
        plt.legend(prop=font)
        plt.ylabel('Loss', fontsize=18, fontname='serif')
        plt.xlabel('Epochs', fontsize=18, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.yticks(fontsize=16, fontname='serif')
        # plt.ylim(-3600,-2500)
        plt.grid()
        plt.show()
        plt.close()


    if plot==True:
        # bins_code = {                                                                                                                                                                                                                                                                         
        #     'mass': np.arange(0, 4000, 50
                            
        #                     )                                                                                                                             
        #     }  
        # ymax_code = {                                                                                                                                                                                                                                                                       
        #     'mass': 5                                                                                                                          
        #     }  
        # xlabel_code = {                                                                                                                                                                                                                                                                        
        #     'mass': r'$m_{jj}$',                                                                                                                            
        #     }  
        feature_labels = list(bins_code.keys())

        REF    = feature[target[:, 0]==0]
        DATA   = feature[target[:, 0]==1]
        weight = target[:, 1]
        weight_REF       = weight[target[:, 0]==0]
        weight_DATA      = weight[target[:, 0]==1]
        plot_training_data(data=DATA, weight_data=weight_DATA, ref=REF, weight_ref=weight_REF, 
                        feature_labels=feature_labels, bins_code=bins_code, xlabel_code=xlabel_code, ymax_code=ymax_code)
                        #    save=True, save_path='thesis', file_name='no_reweight')

    if plot_reconstructions == True:

        REF    = feature[target[:, 0]==0]
        DATA   = feature[target[:, 0]==1]
        weight = target[:, 1]

        weight_REF       = weight[target[:, 0]==0]
        weight_DATA      = weight[target[:, 0]==1]
        output_tau_ref   = tau.predict(REF)


        # bins_code = {                                                                                                                                                                                                                                                                         
        #     'mass': np.arange(0, 4000,40)                                                                                                                             
        #     }  
        # ymax_code = {                                                                                                                                                                                                                                                                       
        #     'mass': 5                                                                                                                        
        #     }  
        # xlabel_code = {                                                                                                                                                                                                                                                                        
        #     'mass': r'$m_{jj}$',                                                                                                                            
        #     }  
        feature_labels = list(bins_code.keys())

        plot_reconstruction(df=BSMdf, data=DATA, weight_data=weight_DATA, ref=REF, weight_ref=weight_REF, 
                            tau_OBS=tau_OBS, output_tau_ref=output_tau_ref,  
                            feature_labels=feature_labels, bins_code=bins_code, xlabel_code=xlabel_code, ymax_code=ymax_code)#,
                            # delta_OBS=delta_OBS, output_delta_ref=output_delta_ref, 
                            # save=True, save_path='thesis', file_name='no_reweight_reco')

    return tau_OBS