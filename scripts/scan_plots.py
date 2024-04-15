import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_single_funcs(sig_key, scoring_metric, baseline_pr_sup, baseline_if_sup):

    results = f'results/supervised_classifier_performance/supervised_{sig_key}.pkl'
    with open(results, 'rb') as f:
        results_dict_sup = pickle.load(f)

    plt.figure(figsize=(10,10))
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')
    plt.xlim([10**-(6), 1.0])
    plt.ylim([10**-(6), 1.05])
    plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
    plt.xlabel('False Positive Rate',fontsize=13)
    plt.ylabel('True Positive Rate',fontsize=13)
    plt.semilogx()
    plt.semilogy()
    plt.plot(results_dict_sup["FPR"], results_dict_sup["TPR"], lw=2, label=f'supervised classifier {sig_key} (AUC = %.01f%%)' % (results_dict_sup["AUC"]* 100))
    plt.legend(loc='lower right',fontsize=13)
    plt.tight_layout()
    plt.savefig(f"scans/supervised_classifier/{sig_key}_classifier_auroc")
    plt.show()

    plt.figure(figsize=(10,10))
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.axhline(baseline_pr_sup, color='red', linestyle='dashed', linewidth=1, label="baseline") # threshold value for measuring anomaly detection efficiency
    plt.xlabel('Recall',fontsize=13)
    plt.ylabel('Precision',fontsize=13)
    plt.plot(results_dict_sup["recall"], results_dict_sup["precision"], lw=2, label=f'supervised classifier {sig_key} (AUC = %.01f%%)' % (results_dict_sup["PR_AUC"]* 100))
    plt.legend(loc='lower right',fontsize=13)
    plt.tight_layout()
    plt.savefig(f"scans/supervised_classifier/{sig_key}_classifier_prauc")
    plt.show()


    results = f"results/isotree/kfold_models/{sig_key}_ntrees_100__scoring_metric_{scoring_metric}__5"
    with open(results, 'rb') as f:
        results_dict_iso = pickle.load(f)


    plt.figure(figsize=(10,10))
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')

    plt.xlim([10**-(6), 1.0])
    plt.ylim([10**-(6), 1.05])
    plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
    plt.xlabel('False Positive Rate',fontsize=13)
    plt.ylabel('True Positive Rate',fontsize=13)
    plt.semilogx()
    plt.semilogy()
    plt.plot(results_dict_iso["ROCAUC"]["base"], results_dict_iso["ROCAUC"]["mean_curve"], lw=2, label = f'iForest {sig_key} {scoring_metric} (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["ROCAUC"]["auc_mean"] * 100, results_dict_iso["ROCAUC"]['auc_unc'] * 100))

    plt.fill_between(results_dict_iso["ROCAUC"]["base"],
                    results_dict_iso["ROCAUC"]["mean_curve"] - results_dict_iso["ROCAUC"]["error_curve"]
    ,
                    results_dict_iso["ROCAUC"]["mean_curve"] + results_dict_iso["ROCAUC"]["error_curve"]
    ,
                    alpha=0.5,
                label = 'five fold uncertainty')

    plt.legend(loc='lower right',fontsize=13)
    plt.tight_layout()
    plt.savefig(f"scans/kfold_scoring_metric/{sig_key}_{scoring_metric}_iforest_auroc")
    plt.show()


    plt.figure(figsize=(10,10))
    plt.xlabel('Recall',fontsize=13)
    plt.ylabel('Precision',fontsize=13)


    plt.axhline(baseline_if_sup, color='red', linestyle='dashed', linewidth=1, label="baseline") # threshold value for measuring anomaly detection efficiency


    plt.plot(results_dict_iso["PRAUC"]["interp_recall"], results_dict_iso["PRAUC"]["mean_precision"], lw=1, label = f'iForest {sig_key} {scoring_metric} (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["PRAUC"]["pr_auc_mean"] * 100, results_dict_iso["PRAUC"]['pr_auc_unc'] * 100))

    plt.fill_between(results_dict_iso["PRAUC"]["interp_recall"],
                    results_dict_iso["PRAUC"]["mean_precision"] - results_dict_iso["PRAUC"]["error_precision"]
    ,
                    results_dict_iso["PRAUC"]["mean_precision"] + results_dict_iso["PRAUC"]["error_precision"]
    ,
                    alpha=0.5,
                label = 'five fold uncertainty')

    plt.legend(loc='center',fontsize=13)
    plt.tight_layout()
    plt.savefig(f"scans/kfold_scoring_metric/{sig_key}_{scoring_metric}_iforest_prauc")
    plt.show()


sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]
scoring_metrics = ["depth", "density", "adj_depth", "adj_density", "boxed_ratio", "boxed_density2", "boxed_density"]
baseline_if_sup = [0.3405, 0.0573, 0.1681, 0.0081, 0.0397, 0.4463, 0.0188, 0.0101]
baseline_pr_sup = [0.2052, 0.0295, 0.0918, 0.0040, 0.0202, 0.2873, 0.0095, 0.0051]


for idx, sig_key in enumerate(sigkeys):
    for scoring_metric in scoring_metrics:
        plot_single_funcs(sig_key, scoring_metric, baseline_pr_sup[idx], baseline_if_sup[idx])



# now we do not go through through boxed_density due to its big uncertainties.
scoring_metrics = ["depth", "density", "adj_depth", "adj_density", "boxed_ratio", "boxed_density2"]
ibm_color_blind_palette = ["#648fff", "#ffb000", "#785ef0", "#dc267f", "#fe6100", "#8c564b"]

for sig_key in sigkeys:
    plt.figure(figsize=(10,10))
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')
    plt.xlim([10**-(6), 1.0])
    plt.ylim([10**-(6), 1.05])
    plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
    plt.xlabel('False Positive Rate',fontsize=13)
    plt.ylabel('True Positive Rate',fontsize=13)
    plt.semilogx()
    plt.semilogy()

    for idx, scoring_metric in enumerate(scoring_metrics):

        results = f"results/isotree/kfold_models/{sig_key}_ntrees_100__scoring_metric_{scoring_metric}__5"
        with open(results, 'rb') as f:
            results_dict_iso = pickle.load(f)
            plt.plot(results_dict_iso["ROCAUC"]["base"], results_dict_iso["ROCAUC"]["mean_curve"], color= ibm_color_blind_palette[idx], lw=2, label = f'iForest {sig_key} {scoring_metric} (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["ROCAUC"]["auc_mean"] * 100, results_dict_iso["ROCAUC"]['auc_unc'] * 100))
    plt.legend(loc='lower right',fontsize=13)
    plt.savefig(f"scans/kfold_scoring_metric/{sig_key}_summary_iforest_auroc")
    plt.show()
    













# import numpy as np
# from scripts import dataset, metrics
# from isotree import IsolationForest
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import pickle
# import os
# import glob
# import math




# def scoring_metric(signal, name):
#     x_train, x_test = dataset.create_xtrain_xtest()
#     signal = dataset.load_dataset('BSM_preprocessed.h5', signal)

#     scoring_metrics = ['depth', 'density', 'adj_depth', 'adj_density', 'boxed_ratio', 'boxed_density2', 'boxed_density']

#     plt.figure(figsize=(10, 10))
#     plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')
#     plt.xlim([10**-(6), 1.0])
#     plt.ylim([10**-(6), 1.05])
#     plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
#     plt.xlabel('False Positive Rate',fontsize=20)
#     plt.ylabel('True Positive Rate',fontsize=20)
#     plt.semilogx()
#     plt.semilogy()

#     for metric in scoring_metrics:
#         print(f"working on {metric}")
#         model = IsolationForest(ndim=1,ntrees=100, scoring_metric=metric).fit(x_train) # this is scoring_metric = "depth"
#         score = model.predict(x_test, output="score")
#         score_bsm = model.predict(signal, output='score')
#         fpr, tpr, auc = metrics.get_roc_auc(score_0=score, score_1=score_bsm) 
#         plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = %.1f%%)' % (auc * 100))
#         print(f"{metric} done")

#     plt.legend(loc='lower right',fontsize=15)
#     plt.tight_layout()
#     plt.savefig(name)
#     plt.show()





# def plot_roc(FPR, TPR, AUC):
#     plt.figure(figsize=(5, 4))
#     plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')
#     plt.xlim([10**-(6), 1.0])
#     plt.ylim([10**-(6), 1.05])
#     plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
#     plt.xlabel('False Positive Rate',fontsize=20)
#     plt.ylabel('True Positive Rate',fontsize=20)
#     plt.semilogx()
#     plt.semilogy()
#     plt.plot(FPR, TPR, lw=2, label='BB (AUC = %.1f%%)' % (AUC * 100))
#     plt.legend(loc='lower right',fontsize=15)
#     plt.tight_layout()

# def get_roc_auc(score_0, score_1):
#     labels = np.concatenate((np.ones(len(score_1)), np.zeros(len(score_0))))
#     all = np.concatenate((score_1, score_0))
#     FPR, TPR, _ = roc_curve(labels, all)
#     AUC = auc(FPR, TPR)
#     return FPR, TPR, AUC

# def train_and_save_model(kwargs, x_train):
#   # kwargs argument is a dictionary
#   model_string = "trained_models/model__" + "__".join([f"{key}_{value}" for key, value in kwargs.items()]) # this line is thanks to chatgpt
#   if os.path.isfile(model_string)==True:
#     print("This model already exists!")
#     return model_string
#   else:
#     model = IsolationForest(**kwargs).fit(x_train)
#     pickle.dump(model, open(model_string, 'wb'))
#     return model_string  

# def predict_value(model, x_test, signal, output1="score"):
#     loaded_model = pickle.load(open(model, 'rb'))
#     score_x_test = loaded_model.predict(x_test, output=output1)
#     score_signal = loaded_model.predict(signal, output=output1)
#     fpr, tpr, auc = get_roc_auc(score_x_test, score_signal)
#     return fpr, tpr, auc

# # for 1 thing (for now?:))

# def get_seed_uncertainty(kwargs, x_test, signal):
#     # here we assume that we have already trained the models :-)
#     model_string = "trained_models/model__" + "__".join([f"{key}_{value}" for key, value in kwargs.items()])+"*"
#     model_names = glob.glob(model_string)
#     fprs = {}
#     tprs = {}
#     aucs = {}
#     for i in range(len(model_names)):
#         fprs[i], tprs[i], aucs[i] = predict_value(model=model_names[i], x_test=x_test, signal=signal)

#     # interpolate

#     interp_tpr={}
#     base = np.exp(np.linspace(math.log(0.00000005), 0., 1000000)) # bc we are interested in the loglog plot later! :)
#     seeds = [0,1,2,3,4]
#     for i in seeds:
#         interp_tpr[i] = np.interp(base, fprs[i], tprs[i])

#     mean_curve = np.mean(list(interp_tpr.values()), axis=0)
#     error_curve = np.std(list(interp_tpr.values()), axis=0)
#     auc_mean = np.mean(list(aucs.values()))
#     auc_unc = np.std(list(aucs.values()))

#     return mean_curve, error_curve, base, auc_mean, auc_unc

# def plot_auroc_unc_(mean_curve, error_curve, base, auc_mean, auc_unc): 
#     plt.plot(base,mean_curve, linewidth=0.5,  label='BB (AUC = %.1f%% $\pm$ %.1f%%)' % (auc_mean * 100, auc_unc * 100))
#     plt.semilogx()
#     plt.semilogy()
#     plt.fill_between(base,
#                 mean_curve - error_curve,
#                 mean_curve + error_curve,
#                 alpha=0.5,
#             label = 'depth')
#     plt.legend(loc='lower right',fontsize=15)


# ## work in progress ::
    
    
# # prob_pick_pooled_gain=0.0, 
#     #prob_pick_avg_gain=0.0, 
#     #prob_pick_full_gain=0.0, 
#     #prob_pick_dens=0.0, 
#     #prob_pick_col_by_range=0.0,
#     #prob_pick_col_by_var=0.0, 
#     #prob_pick_col_by_kurt=0.0


# # def prob_pick_col(signal, name):
# #     x_train, x_test = dataset.create_xtrain_xtest()
# #     signal = dataset.load_dataset('BSM_preprocessed.h5', signal)


# #     plt.figure(figsize=(10, 10))
# #     plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')
# #     plt.xlim([10**-(6), 1.0])
# #     plt.ylim([10**-(6), 1.05])
# #     plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
# #     plt.xlabel('False Positive Rate',fontsize=20)
# #     plt.ylabel('True Positive Rate',fontsize=20)
# #     plt.semilogx()
# #     plt.semilogy()

# #     for metric in scoring_metrics:
# #         print(f"working on {metric}")
# #         model = IsolationForest(ndim=1,ntrees=100, scoring_metric=).fit(x_train) # this is scoring_metric = "depth"
# #         score = model.predict(x_test, output="score")
# #         score_bsm = model.predict(signal, output='score')
# #         fpr, tpr, auc = metrics.get_roc_auc(score_0=score, score_1=score_bsm) 
# #         plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = %.1f%%)' % (auc * 100))
# #         print(f"{metric} done")

# #     plt.legend(loc='lower right',fontsize=15)
# #     plt.tight_layout()
# #     plt.savefig(name)
# #     plt.show()