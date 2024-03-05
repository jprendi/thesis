
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pickle
import os

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

def train_and_save_model(kwargs, x_train):
  # kwargs argument is a dictionary

  model_string = "trained_models/model__" + "__".join([f"{key}_{value}" for key, value in kwargs.items()]) # this line is thanks to chatgpt
  if os.path.isfile(model_string)==True:
    print("This model already exists!")
    return model_string
  else:
    model = IsolationForest(**kwargs).fit(x_train)
    pickle.dump(model, open(model_string, 'wb'))
    return model_string
  
def predict_value(model, x_test, signal, output1="score"):
    loaded_model = pickle.load(open(model, 'rb'))
    score_x_test = loaded_model.predict(x_test, output=output1)
    score_signal = loaded_model.predict(signal, output=output1)
    fpr, tpr, auc = get_roc_auc(score_x_test, score_signal)
    return fpr, tpr, auc
