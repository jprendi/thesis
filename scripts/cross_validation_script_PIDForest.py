"""

this class is to help perform the whole cross validation task BUT specifically for PIDForest !!! 
it trains your k models and then evaluates the uncertainty of the model with the means of ROC-AUC and PR-AUC!
it is the same as cross_validation_script.py but adapted to PID needs :) 

"""

from scripts import dataset
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
import numpy as np
import os, glob, pickle, math
from scripts.forest import Forest

class Cross_Validation:
    """
    Cross validation class!
    """
    def __init__(self, model_parameters, kfold=5, train_models = False): 
        """
        ☆★☆ init of the class!

        input parameteters:
        - kfold (int): how many folds for cross-validation
        - model_parameters (dict): the parameters to train/call isotree on.
            e.g. model_parameters = {"ntrees":100, "scoring_metric": "density"}
        - sigkey (str): key to indicate which signal from 'BSM_preprocessed.h5' to load
        - train_models (bool): just to know wether to train models or just to do the k-fold evaluation!
        """
        self.k = kfold
        self.kwargs = model_parameters
        self.bkg = dataset.load_dataset('NuGun_preprocessed.h5','full_data_cyl', pid=True)
        self.sigkey = []
        self.sig = []
        self.train_indices = 0
        self.test_indices = 0 
        self.fold = 0
        self.x_test_predict = {}
        self.cross_val_splits()
        if train_models == True:
            self.train_k_models()

        self.models = self.get_model_name()


    def cross_val_splits(self): 
        """
        creates the splits for our training/test set
        """
        kf = KFold(n_splits=self.k)
        for i, (train, test) in enumerate(kf.split(self.bkg)):
            if i == self.fold:
                self.train_indices = train
                self.test_indices = test

    def get_trainset(self):
        """
        defines our training data set based on a given splits defined via self.cross_val_splits()
        """
        train_set = self.bkg[self.train_indices]
        return train_set
    
    def get_teststet_bkgonly(self):
        """
        defines the test data set based part of the bkg on a given splits defined via self.cross_val_splits()
        """
        test_set = self.bkg[self.train_indices]
        return test_set   

    def train_and_save_model(self):  
        """
        trains and saves the model! :)
        """
        model_string = "trained_models/isotree/other_models/model__PIDForest" + f"__fold_{self.fold+1}_of_{self.k}" 
        x_train = self.get_trainset()
        if os.path.isfile(model_string)==True:
            print(f"This model already exists: {model_string}.")
        else:
            forest = Forest(**self.kwargs)
            forest.fit(np.transpose(x_train))
            pickle.dump(forest.tree, open(model_string, 'wb'))
            print(f"Model trained: {model_string}.")
        return model_string
    

    def train_k_models(self): #works !
        """
        trains k isolation forest models w isotree!
        """
        directory_path = "trained_models/isotree/other_models/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        for i in range(self.k):
            self.fold = i
            self.cross_val_splits() 
            model_string = self.train_and_save_model()
            model = pickle.load(open(model_string, 'rb'))
            forest = Forest(**self.kwargs)
            forest.tree = model
            x_test = self.get_teststet_bkgonly()
            _,_,_,_, self.x_test_predict[i] = forest.predict(np.transpose(x_test))
            print(f"trained model nr {i}")

    

    def get_predictions(self, model, i):
        """
        prepares predictions for performance assessment

        output:
        - score_test (ndarray): anomaly scores received from prediction
        - labels (ndarray): labels for our scores
            bkg data point gets labeled 0, sig data point gets labeled 1
        """
        bkg_pred = self.x_test_predict[i]
        loaded_model = pickle.load(open(model, 'rb'))
        forest = Forest(**self.kwargs)
        forest.tree = loaded_model
        _, _, _, _, sig_pred = forest.predict(np.transpose(self.sig), err=0.1)
        filtered_score_sig = sig_pred[sig_pred < -400]
        score_test = np.concatenate((-bkg_pred, -filtered_score_sig))
        labels = np.concatenate((np.zeros(len(bkg_pred)), np.ones(len(filtered_score_sig))))
        return score_test, labels


    def predict_value(self, score, labels): 
        """
        ☆★☆ model performance assessment (receiver operating characteristic curve and precision recall curve)
            predicts values from the trained model and returns ROC/AUC and PR/AUC

        input parameter:
        - model (str): path of trained model (to indicate which model to evaluate)

        output:
        - FPR_AUC (ndarray): false positive rate for ROC curve
        - TPR_AUC (ndarray): true positive rate for ROC curve
        - AUC_AUC (float): area under ROC curve
        - precision (ndarray): positive predictive value (number of true positives / number of positive calls)
        - recall (ndarray): true positive rate
        """

        FPR_AUC, TPR_AUC, _ = roc_curve(labels, score)
        AUC_AUC = auc(FPR_AUC, TPR_AUC)
        precision, recall, _ = precision_recall_curve(labels, score)
        PR_AUC = auc(recall, precision)
   
        return FPR_AUC, TPR_AUC, AUC_AUC, precision, recall, PR_AUC


    def get_kfold_uncertainty(self):
        """
        ☆★☆ gives us uncertainty on ROC for k-fold cross validation
            - computes ROC of each fold 
            - interpolates the mean of the k folds with an uncertainty

        output:
        - mean_curve (ndarray): mean ROC curve of all k folds
        - error_curve (ndarray): uncertainty of ROC curve over all k folds
        - base (ndarray): base values for interpolation
        - auc_mean (float): mean of AUC of ROC curve for all k folds
        - auc_unc (float): uncertainty of AUC of ROC over all k folds
        - mean_precision (ndarray): mean precision curve of all k folds
        - error_precision (ndarray): uncertainty of PR curve over all k folds
        - pr_auc_mean (float): mean of AUC of PR curve for all k folds
        - pr_auc_unc (float): uncertainty of AUC of PR curve for all k folds
        """
        model_string = "trained_models/isotree/other_models/model__PIDForest" + "*"
        model_names = glob.glob(model_string)
        fprs = {}
        tprs = {}
        aucs = {}
        precisions = {}
        recalls = {}
        pr_aucs = {}

        for i in range(self.k):
            self.fold = i
            self.cross_val_splits()
            score, labels = self.get_predictions(model=model_names[i], i=i)
            fprs[i], tprs[i], aucs[i], precisions[i], recalls[i], pr_aucs[i] = self.predict_value(score= score, labels=labels)

        interp_tpr={}
        interp_tpr_pr = {}
        base = np.exp(np.linspace(math.log(0.00000005), 0., 1000000)) # bc we are interested in the loglog plot later! :)
        base_pr = np.linspace(0, 1, 1000000)

        for i in range(self.k):
            interp_tpr[i] = np.interp(base, fprs[i], tprs[i])
            interp_tpr_pr[i] = np.interp(base_pr, recalls[i][::-1], precisions[i][::-1])


        mean_curve = np.mean(list(interp_tpr.values()), axis=0)
        error_curve = np.std(list(interp_tpr.values()), axis=0)
        auc_mean = np.mean(list(aucs.values()))
        auc_unc = np.std(list(aucs.values()))

        mean_precision = np.mean(list(interp_tpr_pr.values()), axis=0)
        error_precision = np.std(list(interp_tpr_pr.values()), axis=0)
        pr_auc_mean = np.mean(list(pr_aucs.values()))
        pr_auc_unc = np.std(list(pr_aucs.values()))


        return mean_curve, error_curve, base, auc_mean, auc_unc, mean_precision, error_precision, base_pr, pr_auc_mean, pr_auc_unc


    def getnsave_kfold_auc_prauc(self, sigkey):
        """
        - computes performance of kfold cross validation and uncertainties for both ROC and PR with the uncertainties
        - saves it
        """
        self.sigkey = sigkey
        self.sig = dataset.load_dataset('BSM_preprocessed.h5', sigkey)

        directory_path = "results/isotree/other_models/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        #self.get_predictions()

        mean_curve, error_curve, base, auc_mean, auc_unc, mean_precision, error_precision, base_recall, pr_auc_mean, pr_auc_unc = self.get_kfold_uncertainty()
        # mean_precision, error_precision, base_recall, pr_auc_mean, pr_auc_unc = self.get_kfold_uncertainty_PR_AUC()

        results_dict = {
            "ROCAUC": {
                "mean_curve": mean_curve,
                "error_curve": error_curve,
                "base": base,
                "auc_mean": auc_mean,
                "auc_unc": auc_unc
            },
            "PRAUC": {
                "mean_precision": mean_precision,
                "error_precision": error_precision,
                "interp_recall": base_recall,
                "pr_auc_mean": pr_auc_mean,
                "pr_auc_unc": pr_auc_unc
            }
        }
        output_file = "results/isotree/other_models/" + f"{self.sigkey}_" + "__PIDForest"+ "__" + f"{self.k}"
        with open(output_file, 'wb') as f:
            pickle.dump(results_dict, f)



    def get_model_name(self):
        """
        returns a list of all the model names to loop ver
        """
        model_names = "trained_models/isotree/other_models/model__PIDForest" + "_*"+"fold"+"*"
        return list(glob.glob(model_names))