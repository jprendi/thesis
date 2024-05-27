"""
this was taken from this page:

https://codimd.web.cern.ch/e_kkhesGRZiiu__85_fpnQ#Predictions-from-Keras


loading this script will 
        1) load the AXOL1TL architecture
        2) get predictions on different BSM signatures
        3) save the performance assessement in terms of ROC/PR in the end as csv file for easy overview! :)
                (and other data like fpr, tpr, precision, recall and the AUC respectively of course)

                
"""
from scripts.dataset import load_dataset, remove_columns
from qkeras.utils import _add_supported_quantized_objects
from qkeras import quantized_bits
co = {}; _add_supported_quantized_objects(co)
import tensorflow as tf
import h5py
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import numpy as np 
from scripts.dataset import load_axo_dataset
import pickle
import pandas as pd

# import os
# import tensorflow as tf # type: ignore
# import tensorflow.keras as keras  # type: ignore
# from tensorflow.keras.models import Model  # type: ignore
# from tensorflow.keras.layers import Input, Dense,ZeroPadding2D, BatchNormalization, Activation, Layer, ReLU, LeakyReLU,Conv2D,AveragePooling2D,UpSampling2D,Reshape,Flatten  # type: ignore
# from tensorflow.keras import backend as K  # type: ignore
# from scripts import dataset
# from scripts.func import load_model, save_model, mse_loss
# from sklearn.metrics import roc_curve, auc, precision_recall_curve  # type: ignore
# import pickle
# import pandas as pd  # type: ignore

class AXOL1TL:
    '''
    this class loads AXO and helps the prediction process
    '''
    def __init__(self):
        """
        ☆★☆ init of the class!
        
        what happens here is that we load our bkg dataset and model.
        it also directly determines the score the bkg has with said model.
        """
        self.model = self.AXO_model()
        self.bkg =  load_axo_dataset('NuGun_preprocessed.h5')
        self.bkg_score = self.AXO_predictions(self.bkg)

    def AXO_model(self):
        """
        this function loads the AXO architecture, taken from:
        https://gitlab.cern.ch/cms-l1-ad/
    
        Returns: 
        - tensorflow.keras.Model : our loaded model

        """
        co = {}; _add_supported_quantized_objects(co)
        return tf.keras.models.load_model('encoder_trimmed.h5', custom_objects=co)


    def predict(self, dataset):
       '''
       does predictions based on our chosen dataset
       '''
       return self.autoencoder.predict(dataset)
    
    def AXO_predictions(self, dataset):
        '''
        determines and returns the predictions of AXO for a given dataset
        '''

        latent_axo_qk = self.model.predict(dataset, batch_size=len(dataset))
        return np.sum(latent_axo_qk**2, axis=1)


    def eval_sig(self, key):

        """
        Evaluate the signal data using the AXOL1TL architecture and calculate performance metrics.

        Parameters:
        - key (str): Identifier for the signal data.

        Returns:
        - FPR_AUC (array): False Positive Rate values for the ROC curve.
        - TPR_AUC (array): True Positive Rate values for the ROC curve.
        - AUC_AUC (float): Area Under the ROC Curve (ROC AUC).
        - precision (array): Precision values for the precision-recall curve.
        - recall (array): Recall values for the precision-recall curve.
        - PR_AUC (float): Area Under the Precision-Recall Curve (PR AUC).
        """
        signal_data = load_axo_dataset(dataset='BSM_preprocessed.h5', key=key)
        sig_pred = self.AXO_predictions(signal_data)
        labels = np.concatenate((np.ones(np.shape(sig_pred)), np.zeros(np.shape(self.bkg_score))))
        total_loss = np.concatenate((sig_pred, self.bkg_score))

        FPR_AUC, TPR_AUC, _ = roc_curve(labels, total_loss)
        AUC_AUC = auc(FPR_AUC, TPR_AUC)
        precision, recall, _ = precision_recall_curve(labels, total_loss)
        PR_AUC = auc(recall, precision)
        return FPR_AUC, TPR_AUC, AUC_AUC, precision, recall, PR_AUC

    def get_and_save_eval(self, sig):
        """
        Perform evaluation on signal data and save results to a file.

        Parameters:
        - sig (str): Identifier for the signal data.
        """
        directory_path = "results/AXO/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        FPR_AUC, TPR_AUC, AUC_AUC, precision, recall, PR_AUC = self.eval_sig(sig)

        results_dict = {
            "ROCAUC": {
                "FPR_AUC": FPR_AUC,
                "TPR_AUC": TPR_AUC,
                "AUC_AUC": AUC_AUC,
            },
            "PRAUC": {
                "precision": precision,
                "recall": recall,
                "PR_AUC": PR_AUC,
            }
        }
        output_file = "results/AXO/AXO_" + f"{sig}"
        with open(output_file, 'wb') as f:
            pickle.dump(results_dict, f)



axo = AXOL1TL()

sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]
for sig in sigkeys:
    axo.get_and_save_eval(sig=sig)

results_table = []


for sigkey in sigkeys:
    results = f"results/AXO/AXO_{sigkey}"
    with open(results, 'rb') as f:
        results_dict = pickle.load(f)
    ll = [sigkey, results_dict["ROCAUC"]["AUC_AUC"], results_dict["PRAUC"]["PR_AUC"]]
    results_table.append(ll)

pd.DataFrame(results_table, columns =['signal', 'rocauc', 'prauc']).to_csv(path_or_buf="results/AXO/results_AXO.csv")   


    


