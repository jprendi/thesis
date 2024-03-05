
import numpy as np
from scripts import dataset, metrics
from isotree import IsolationForest
import matplotlib.pyplot as plt

def scoring_metric(signal, name):
    x_train, x_test = dataset.create_xtrain_xtest()
    signal = dataset.load_dataset('BSM_preprocessed.h5', signal)

    scoring_metrics = ['depth', 'density', 'adj_depth', 'adj_density', 'boxed_ratio', 'boxed_density2', 'boxed_density']

    plt.figure(figsize=(10, 10))
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='0.75')
    plt.xlim([10**-(6), 1.0])
    plt.ylim([10**-(6), 1.05])
    plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.semilogx()
    plt.semilogy()

    for metric in scoring_metrics:
        print(f"working on {metric}")
        model = IsolationForest(ndim=1,ntrees=100, scoring_metric=metric).fit(x_train) # this is scoring_metric = "depth"
        score = model.predict(x_test, output="score")
        score_bsm = model.predict(signal, output='score')
        fpr, tpr, auc = metrics.get_roc_auc(score_0=score, score_1=score_bsm) 
        plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = %.1f%%)' % (auc * 100))
        print(f"{metric} done")

    plt.legend(loc='lower right',fontsize=15)
    plt.tight_layout()
    plt.savefig(name)
    plt.show()


## work in progress ::
    
    
# prob_pick_pooled_gain=0.0, 
    #prob_pick_avg_gain=0.0, 
    #prob_pick_full_gain=0.0, 
    #prob_pick_dens=0.0, 
    #prob_pick_col_by_range=0.0,
    #prob_pick_col_by_var=0.0, 
    #prob_pick_col_by_kurt=0.0


# def prob_pick_col(signal, name):
#     x_train, x_test = dataset.create_xtrain_xtest()
#     signal = dataset.load_dataset('BSM_preprocessed.h5', signal)


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
#         model = IsolationForest(ndim=1,ntrees=100, scoring_metric=).fit(x_train) # this is scoring_metric = "depth"
#         score = model.predict(x_test, output="score")
#         score_bsm = model.predict(signal, output='score')
#         fpr, tpr, auc = metrics.get_roc_auc(score_0=score, score_1=score_bsm) 
#         plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = %.1f%%)' % (auc * 100))
#         print(f"{metric} done")

#     plt.legend(loc='lower right',fontsize=15)
#     plt.tight_layout()
#     plt.savefig(name)
#     plt.show()