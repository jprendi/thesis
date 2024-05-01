"""
this was heavily taken from this github notebook:

https://github.com/mpp-hep/ADC2021-examplecode/blob/main/Convolutional_AE.ipynb

but changed up a little for my own use case :)

loading this script will 
        1) train the CAE based on bkg
        2) get predictions on different BSM signatures
        3) save the performance assessement in terms of ROC/PR in the end as csv file for easy overview! :)
                (and other data like fpr, tpr, precision, recall and the AUC respectively of course)

                
tf.__version__ must give a version around 2  !

"""

import numpy as np
import math
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,ZeroPadding2D, BatchNormalization, Activation, Layer, ReLU, LeakyReLU,Conv2D,AveragePooling2D,UpSampling2D,Reshape,Flatten
from tensorflow.keras import backend as K
from scripts import dataset
from scripts.func import load_model, save_model, mse_loss
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pickle
import pandas as pd

class CAE:
    '''
    this class defines the CAE and helps the prediction process
    '''
    def __init__(self):
        """
        ☆★☆ init of the class!
        
        what happens here is that we load our train, test and validation dataset.
        we also define important model parameter, the model itself, and also train it directly!
        it also directly determines the loss the bkg has with said model.
        """
        self.X_train, self.X_test, self.X_val =  dataset.create_CAE_datset()
        self.image_shape = (33,3,1)
        self.latent_dimension = 8
        self.num_nodes=[16,8]
        self.autoencoder = self.CAE_model()
        self.autoencoder.compile(optimizer = keras.optimizers.Adam(), loss='mse')
        self.train_CAE()
        self.bkg_loss = self.CAE_loss(self.X_test)

    def CAE_model(self):
        """
        this function defines the Convolutional Autoencoder (CAE).
        it consists of an encoder and decoder part. the architecture was stolen from: 
        https://github.com/mpp-hep/ADC2021-examplecode/blob/main/Convolutional_AE.ipynb

        while i had to update the ZeroPadding2D layer s.t. it would match the image_shape i have w the current dataset!
           
        Returns: 
        - tensorflow.keras.Model : our defined CAE

        """
        #encoder
        input_encoder = Input(shape=(self.image_shape))
        x = Conv2D(10, kernel_size=(3, 3),
         use_bias=False, data_format='channels_last', padding='same')(input_encoder)
        x = AveragePooling2D(pool_size = (2, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(self.latent_dimension)(x)
        enc = Activation('relu')(x)
        encoder = Model(inputs=input_encoder, outputs=enc)

        #decoder
        x = Dense(270)(enc)
        x = Activation('relu')(x)
        x = Reshape((9,3,10))(x)
        x = UpSampling2D((2, 1))(x)
        x = ZeroPadding2D(((7, 8),(0,0)))(x) ## changed this   # ZeroPadding2D(((1, 0),(0,0)))(x) 
        x = Conv2D(1, kernel_size=(3,3), use_bias=False, data_format='channels_last', padding='same')(x)
        x = BatchNormalization()(x)
        dec = Activation('relu')(x)

        return Model(inputs=input_encoder, outputs=dec)


    def train_CAE(self):
        '''
        trains the CAE
        '''
        self.autoencoder.fit(self.X_train, self.X_train, epochs = 10, batch_size = 1024, validation_data=(self.X_val, self.X_val))
        save_model('CNNS/CNN_AE', self.autoencoder)


    def predict(self, dataset):
       '''
       does predictions based on our chosen dataset
       '''
       return self.autoencoder.predict(dataset)
    
    def CAE_loss(self, dataset):
        '''
        determines the loss for the CAE while also needing the predictions
        it returns the respective loss of a dataset
        '''
        predicttion = self.predict(dataset)
        loss = mse_loss(dataset.reshape((dataset.shape[0],dataset.shape[1]*dataset.shape[2])),\
                           (predicttion.reshape((predicttion.shape[0],predicttion.shape[1]*predicttion.shape[2]))).astype(np.float32)).numpy()
        return loss


    def eval_sig(self, key):

        """
        Evaluate the signal data using the Convolutional Autoencoder (CAE) and calculate performance metrics.

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
        signal_data = dataset.load_sig_CAE(key)
        sig_loss = self.CAE_loss(signal_data)
        labels = np.concatenate((np.ones(np.shape(sig_loss)), np.zeros(np.shape(self.bkg_loss))))
        total_loss = np.concatenate((sig_loss, self.bkg_loss))

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
        directory_path = "results/CAE/"
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
        output_file = "results/CAE/CAE" + f"{sig}"
        with open(output_file, 'wb') as f:
            pickle.dump(results_dict, f)



cae = CAE()

sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]
for sig in sigkeys:
    cae.get_and_save_eval(sig=sig)

results_table = []


for sigkey in sigkeys:
    results = f"results/CAE/CAE{sigkey}"
    with open(results, 'rb') as f:
        results_dict = pickle.load(f)
    ll = [sigkey, results_dict["ROCAUC"]["AUC_AUC"], results_dict["PRAUC"]["PR_AUC"]]
    results_table.append(ll)

pd.DataFrame(results_table, columns =['signal', 'rocauc', 'prauc']).to_csv(path_or_buf="results/CAE/results_CAE.csv")   


    




# bin_size=100

# plt.figure(figsize=(10,8))
# plt.hist(total_loss, bins=bin_size, density = True, histtype='step', fill=False, linewidth=1.5)
# plt.yscale('log')
# plt.xlabel("Autoencoder Loss")
# plt.ylabel("Probability (a.u.)")
# plt.title('MSE loss')
# plt.legend(loc='best')
# plt.show()
