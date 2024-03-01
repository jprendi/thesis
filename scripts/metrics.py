
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc(FPR, TPR, AUC):
    plt.figure(figsize=(5, 4))
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')
    plt.xlim([10**-(6), 1.0])
    plt.ylim([10**-(6), 1.05])
    plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.semilogx()
    plt.semilogy()

    plt.plot(FPR, TPR, lw=2, label='BB (AUC = %.1f%%)' % (AUC * 100))

    plt.legend(loc='lower right',fontsize=15)
    plt.tight_layout()

def get_roc_auc(score_0, score_1):
    labels = np.concatenate((np.ones(len(score_1)), np.zeros(len(score_0))))
    all = np.concatenate((score_1, score_0))
    FPR, TPR, _ = roc_curve(labels, all)
    AUC = auc(FPR, TPR)
    return FPR, TPR, AUC