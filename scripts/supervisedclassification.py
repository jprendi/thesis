"""
this assumes you already loaded the do_crossvalidation.py script!

"""
import pandas as pd

from sklearn.preprocessing import StandardScaler
import numpy as np
import dataset
import tensorflow as tf
from callbacks import all_callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve


## most of this code is borrowed from the hls4ml tutorial :)

class SupervisedClassifier():

    def __init__(self, signal, random_seed=1, train=True):
        self.sig_key = signal
        self.X_train_val, self.X_test, self.y_train_val, self.y_test  = dataset.supervised_xtrain_xtest(sig_key=signal)
        self.seed = random_seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.model = 0

        if train:
            self.architecture()
            self.train_model()
        else:
            self.model = load_model(f'trained_models/basic_classifier/model_{self.sig_key}/KERAS_check_best_model.h5')


    def architecture(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(99,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
        model.add(Activation(activation='relu', name='relu1'))
        model.add(Dense(32, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
        model.add(Activation(activation='relu', name='relu2'))
        model.add(Dense(32, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
        model.add(Activation(activation='relu', name='relu3'))
        model.add(Dense(2, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
        model.add(Activation(activation='softmax', name='softmax'))
        self.model = model

    
    def train_model(self):
        adam = Adam(lr=0.0001)
        self.model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
        callbacks = all_callbacks(
            stop_patience=1000,
            lr_factor=0.5,
            lr_patience=10,
            lr_epsilon=0.000001,
            lr_cooldown=2,
            lr_minimum=0.0000001,
            outputDir=f'trained_models/basic_classifier/model_{self.sig_key}',
        )
        self.model.fit(
            self.X_train_val,
            self.y_train_val,
            batch_size=1024,
            epochs=30,
            validation_split=0.25,
            shuffle=True,
            callbacks=callbacks.callbacks,
        )

    def predict_and_save(self):
        results_dict = {}
        y_keras = self.model.predict(self.X_test)
        
        FPR, TPR, _ = roc_curve(y_score=np.max(y_keras, axis=1), y_true=self.y_test[:,0])
        AUC = auc(FPR, TPR)
        precision, recall, _ = precision_recall_curve(probas_pred=np.max(y_keras, axis=1), y_true=self.y_test[:,0])
        PR_AUC = auc(recall, precision)

        results_dict["FPR"] = FPR
        results_dict["TPR"] = TPR
        results_dict["AUC"] = AUC
        results_dict["precision"] = precision
        results_dict["recall"] = recall
        results_dict["PR_AUC"] = PR_AUC


        with open(f'results/supervised_classifier_performance/supervised_{self.sig_key}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
        print("Data saved successfully.")


sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]

for sig in sigkeys:
    print(f"{sig}")
    SC = SupervisedClassifier(signal=sig, train=False)
    SC.predict_and_save()
    print(len(SC.X_test), len(SC.y_test))

df = pd.read_csv("results/isotree/kfold_models/results_kfold.csv").drop('Unnamed: 0', axis=1) 

for sig_key in sigkeys:
    results = f'results/supervised_classifier_performance/supervised_{sig_key}.pkl'
    with open(results, 'rb') as f:
        results_dict = pickle.load(f)
    ll = [sig_key, "supervised_NN", results_dict["AUC"], 0, results_dict["PR_AUC"], 0]
    df.loc[len(df)] = ll

df.sort_values(by=['signal', 'scoring metric'], ascending=[True, True]).to_csv("results/results_iso_supervised.csv")

print("done!")