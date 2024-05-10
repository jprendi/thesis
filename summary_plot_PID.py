import matplotlib.pyplot as plt
import numpy as np
import pickle

ibm_color_blind_palette = ["#648fff", "#ffb000", "#785ef0", "#dc267f", "#fe6100", "#8c564b"]
trees = [10, 50, 100]
sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]

fig, axs = plt.subplots(4, 2, figsize=(14, 24))
axs = axs.flatten()

for idx, sig_key in enumerate(sigkeys):
    ax = axs[idx]
    ax.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')
    ax.set_xlim([10**-(6), 1.0])
    ax.set_ylim([10**-(6), 1.05])
    ax.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)  # threshold value for measuring anomaly detection efficiency
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate', fontsize=15)
    ax.semilogx()
    ax.semilogy()

    for t_idx, tree in enumerate(trees):
        results = f'results/isotree/other_models/{sig_key}___PIDForest__{tree}'
        with open(results, 'rb') as f:
            results_dict_iso = pickle.load(f)
        ax.plot(results_dict_iso['ROCAUC']["base"], results_dict_iso['ROCAUC']["mean_curve"], lw=2, color=ibm_color_blind_palette[t_idx],
                label=f'{tree} trees (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["ROCAUC"]["auc_mean"] * 100, results_dict_iso["ROCAUC"]['auc_unc'] * 100))

    ax.set_title(f'ROC PIDForest {sig_key}')
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig("scans/PIDForest/ROC_ntrees_combined.png")
plt.show()


# Baselines for each signal
baselines = [0.3405, 0.0573, 0.1681, 0.0081, 0.0397, 0.4486, 0.0188, 0.0101]  # Adjust these values accordingly

fig, axs = plt.subplots(4, 2, figsize=(14, 24))
axs = axs.flatten()

for idx, sig_key in enumerate(sigkeys):
    ax = axs[idx]
    ax.set_xlabel('Recall', fontsize=15)
    ax.set_ylabel('Precision', fontsize=15)

    # Plotting baseline
    ax.plot([0, 1], [baselines[idx], baselines[idx]], '--', color='gray', label=f'Baseline = {baselines[idx]:.0%}')

    for t_idx, tree in enumerate(trees):
        results = f'results/isotree/other_models/{sig_key}___PIDForest__{tree}'
        with open(results, 'rb') as f:
            results_dict_iso = pickle.load(f)
        ax.plot(results_dict_iso['PRAUC']["interp_recall"], results_dict_iso['PRAUC']["mean_precision"], lw=2, color=ibm_color_blind_palette[t_idx],
        label=f'{tree} trees (AUC = %.01f%% ± %.01f%%)' % (results_dict_iso["PRAUC"]["pr_auc_mean"] * 100, results_dict_iso["PRAUC"]['pr_auc_unc'] * 100))

    ax.set_title(f'Precision Recall {sig_key}')
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig("scans/PIDForest/PR_ntrees_combined_with_baseline.png")
plt.show()
