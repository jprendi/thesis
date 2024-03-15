# this class is done to perform the whole cross validation task! :)
from scripts import dataset#, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import numpy as np
# import matplotlib.pyplot as plt
import os, glob, pickle, math
from isotree import IsolationForest


class Cross_Validation:

    def __init__(self, kfold, model_parameters, sigkey, train_models = False): #works!
        self.k = kfold
        self.kwargs = model_parameters
        self.sig = dataset.load_dataset('BSM_preprocessed.h5', sigkey)
        self.bkg = dataset.load_dataset('NuGun_preprocessed.h5','full_data_cyl')
        self.train_indices = 0
        self.test_indices = 0 
        self.fold = 0
        self.cross_val_splits()
        if train_models == True:
            self.train_k_models()

    def cross_val_splits(self):  #works!
        kf = KFold(n_splits=self.k)
        for i, (train, test) in enumerate(kf.split(self.bkg)):
            if i == self.fold:
                self.train_indices = train
                self.test_indices = test

    def get_trainset(self):   #works!
        train_set = self.bkg[self.train_indices]
        return train_set
    
    def get_testset_labels(self):   #works!
        train_set = np.concatenate((self.bkg[self.train_indices], self.sig))
        labels = np.concatenate((np.zeros(len(self.train_indices)), np.ones(len(self.sig))))
        return train_set, labels 
    
    def train_and_save_model(self):  #works!
    # kwargs argument is a dictionary
        model_string = "trained_models/model__" + "__".join([f"{key}_{value}" for key, value in self.kwargs.items()])+ f"__fold_{self.fold+1}_of_{self.k}" # this line is thanks to chatgpt
        x_train = self.get_trainset()
        if os.path.isfile(model_string)==True:
            print(f"This model already exists: {model_string}.")
        else:
            model = IsolationForest(**self.kwargs).fit(x_train)
            pickle.dump(model, open(model_string, 'wb'))
            print(f"Model trained: {model_string}.")

    def train_k_models(self): #works !
        for i in range(self.k):
            self.fold = i
            self.cross_val_splits()
            self.train_and_save_model()
   
    def predict_value(self, model, output1="score"): #works!  # maybe need to indicate which fold we're at ????
        loaded_model = pickle.load(open(model, 'rb'))
        x_test, labels = self.get_testset_labels()
        score_x_test = loaded_model.predict(x_test, output=output1)
        FPR, TPR, _ = roc_curve(labels, score_x_test)
        AUC = auc(FPR, TPR)
        return FPR, TPR, AUC

    def get_kfold_uncertainty(self):#(kwargs, x_test, signal):
        # here we assume that we have already trained the models :-)
        model_string = "trained_models/model__" + "__".join([f"{key}_{value}" for key, value in self.kwargs.items()])+"*"
        model_names = glob.glob(model_string)
        fprs = {}
        tprs = {}
        aucs = {}

        for i in range(self.k):
            self.fold = i
            self.cross_val_splits()
            fprs[i], tprs[i], aucs[i] = self.predict_value(model=model_names[i])

        # interpolate

        interp_tpr={}
        base = np.exp(np.linspace(math.log(0.00000005), 0., 1000000)) # bc we are interested in the loglog plot later! :)
        # folds = [0,1,2,3,4]
        for i in range(self.k):
            interp_tpr[i] = np.interp(base, fprs[i], tprs[i])

        mean_curve = np.mean(list(interp_tpr.values()), axis=0)
        error_curve = np.std(list(interp_tpr.values()), axis=0)
        auc_mean = np.mean(list(aucs.values()))
        auc_unc = np.std(list(aucs.values()))

        return mean_curve, error_curve, base, auc_mean, auc_unc

    def get_model_name(self):#, kfold=False, k="all"):
        model_names = "trained_models/model__" + "__".join([f"{key}_{value}" for key, value in self.kwargs.items()])+"__fold_"+"*"
        return glob.glob(model_names)
        # if kfold==True and k="all":
        #     return "trained_models/model__" + "__".join([f"{key}_{value}" for key, value in self.kwargs.items()])+"__fold_+"*""     #  f"__fold_{self.fold+1}_of_{self.k}"
        # elif kfold ==True:
        #     return "trained_models/model__" + "__".join([f"{key}_{value}" for key, value in self.kwargs.items()])+ f"__fold_{k}_of_{self.k}"
        # else:
        #     return "trained_models/model__" + "__".join([f"{key}_{value}" for key, value in self.kwargs.items()])+"*"

        

