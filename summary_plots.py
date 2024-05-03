'''
the script that just produced my summary plots for ROC and PR and directly saves them.. nothing fancy
'''

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pickle


ibm_color_blind_palette = ["#648fff", "#ffb000", "#785ef0", "#dc267f", "#fe6100", "#8c564b"]
FCF = {'ndim':2, 'sample_size':256, 'max_depth': None, 'ntrees':200, 'missing_action':"fail", 'coefs':"normal", 'ntry':1, 'prob_pick_pooled_gain':1}


sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]


for sig_key in sigkeys:
    plt.figure(figsize=(10,10))
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')
    plt.xlim([10**-(6), 1.0])
    plt.ylim([10**-(6), 1.05])
    plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
    plt.xlabel('False Positive Rate',fontsize=15)
    plt.ylabel('True Positive Rate',fontsize=15)
    plt.semilogx()
    plt.semilogy()

    results = f"results/isotree/kfold_models/{sig_key}_ntrees_100__scoring_metric_depth__5"
    with open(results, 'rb') as f:
        results_dict_iso = pickle.load(f)
    plt.plot(results_dict_iso["ROCAUC"]["base"], results_dict_iso["ROCAUC"]["mean_curve"], color= ibm_color_blind_palette[0], lw=2, label = f'iForest depth (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["ROCAUC"]["auc_mean"] * 100, results_dict_iso["ROCAUC"]['auc_unc'] * 100))
    
    results = f"results/isotree/kfold_models/{sig_key}_ntrees_100__scoring_metric_boxed_ratio__5"
    with open(results, 'rb') as f:
        results_dict_iso = pickle.load(f)
    plt.plot(results_dict_iso["ROCAUC"]["base"], results_dict_iso["ROCAUC"]["mean_curve"], color= ibm_color_blind_palette[1], lw=2, label = f'iForest boxed_ratio (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["ROCAUC"]["auc_mean"] * 100, results_dict_iso["ROCAUC"]['auc_unc'] * 100))
    
    results = f'results/isotree/other_models/{sig_key}_ndim_2__sample_size_256__max_depth_None__ntrees_200__missing_action_fail__coefs_normal__ntry_1__prob_pick_pooled_gain_1__5'
    with open(results, 'rb') as f:
        results_dict_iso = pickle.load(f)
    plt.plot(results_dict_iso['ROCAUC']["base"], results_dict_iso['ROCAUC']["mean_curve"], lw=2, color=ibm_color_blind_palette[3], label = f'RRCF (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["ROCAUC"]["auc_mean"] * 100, results_dict_iso["ROCAUC"]['auc_unc'] * 100))


    results = f'results/CAE/CAE{sig_key}'
    with open(results, 'rb') as f:
        results_dict_cae = pickle.load(f)
    plt.plot(results_dict_cae['ROCAUC']["FPR_AUC"], results_dict_cae['ROCAUC']["TPR_AUC"], lw=2, color=ibm_color_blind_palette[4], label = f'CAE (AUC = %.01f%%)' % (results_dict_cae["ROCAUC"]["AUC_AUC"] * 100))
        
    results = f'results/supervised_classifier_performance/supervised_{sig_key}.pkl'
    with open(results, 'rb') as f:
        results_dict_sup = pickle.load(f)
    plt.plot(results_dict_sup["FPR"], results_dict_sup["TPR"], lw=2, color=ibm_color_blind_palette[2], label=f'supervised classifier (AUC = %.01f%%)' % (results_dict_sup["AUC"]* 100))


    plt.title(f'ROC {sig_key}')
    plt.legend(loc='lower right',fontsize=15)
    plt.savefig(f"{sig_key}_summary_plots_CAE")
    plt.show()
    

for sig_key in sigkeys:
    plt.figure(figsize=(10,10))
    plt.xlabel('Recall',fontsize=15)
    plt.ylabel('Precision',fontsize=15)

    results = f"results/isotree/kfold_models/{sig_key}_ntrees_100__scoring_metric_depth__5"
    with open(results, 'rb') as f:
        results_dict_iso = pickle.load(f)
    plt.plot(results_dict_iso["PRAUC"]["interp_recall"], results_dict_iso["PRAUC"]["mean_precision"], color= ibm_color_blind_palette[0], lw=2, label = f'iForest depth (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["PRAUC"]["pr_auc_mean"] * 100, results_dict_iso["PRAUC"]['pr_auc_unc'] * 100))
    
    results = f"results/isotree/kfold_models/{sig_key}_ntrees_100__scoring_metric_boxed_ratio__5"
    with open(results, 'rb') as f:
        results_dict_iso = pickle.load(f)
    plt.plot(results_dict_iso["PRAUC"]["interp_recall"], results_dict_iso["PRAUC"]["mean_precision"], color= ibm_color_blind_palette[1], lw=2, label = f'iForest boxed_ratio (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["PRAUC"]["pr_auc_mean"] * 100, results_dict_iso["PRAUC"]['pr_auc_unc'] * 100))
    
    results = f'results/isotree/other_models/{sig_key}_ndim_2__sample_size_256__max_depth_None__ntrees_200__missing_action_fail__coefs_normal__ntry_1__prob_pick_pooled_gain_1__5'
    with open(results, 'rb') as f:
        results_dict_iso = pickle.load(f)
    plt.plot(results_dict_iso['PRAUC']["interp_recall"], results_dict_iso['PRAUC']["mean_precision"], lw=2, color=ibm_color_blind_palette[3], label = f'RRCF (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["PRAUC"]["pr_auc_mean"] * 100, results_dict_iso["PRAUC"]['pr_auc_unc'] * 100))


    results = f'results/CAE/CAE{sig_key}'
    with open(results, 'rb') as f:
        results_dict_cae = pickle.load(f)
    plt.plot(results_dict_cae['PRAUC']["recall"], results_dict_cae['PRAUC']["precision"], lw=2, color=ibm_color_blind_palette[4], label = f'CAE (AUC = %.01f%%)' % (results_dict_cae["PRAUC"]["PR_AUC"] * 100))
        
    results = f'results/supervised_classifier_performance/supervised_{sig_key}.pkl'
    with open(results, 'rb') as f:
        results_dict_sup = pickle.load(f)
    plt.plot(results_dict_sup["recall"], results_dict_sup["precision"], lw=2, color=ibm_color_blind_palette[2], label=f'supervised classifier (AUC = %.01f%%)' % (results_dict_sup["PR_AUC"]* 100))

    plt.title(f'Precision Recall {sig_key}')
    plt.legend(loc='best',fontsize=15)
    plt.savefig(f"{sig_key}_summary_plots_CAE_PR")
    plt.show()