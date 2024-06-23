
from scripts import cv_reduced
import pickle
import pandas as pd

kwargs = {"ntrees":100, "scoring_metric": "depth"}
sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]
scoring_metrics = ["depth"]

for metric in scoring_metrics:
    kwargs["scoring_metric"] = metric
    print(metric)
    cv = cv_reduced.Cross_Validation(kfold=5, model_parameters=kwargs, train_models = True)
    for sig in sigkeys:
        print(sig)
        cv.getnsave_kfold_auc_prauc(sigkey=sig)

results_table = []

for sigkey in sigkeys:
    for scoring_metric in scoring_metrics:
        results = f"results/isotree/reduced/{sigkey}_ntrees_100__scoring_metric_{scoring_metric}__5"
        with open(results, 'rb') as f:
            results_dict = pickle.load(f)
        # print(results)
        ll = [sigkey, scoring_metric, results_dict["ROCAUC"]["auc_mean"], results_dict["ROCAUC"]["auc_unc"], results_dict["PRAUC"]["pr_auc_mean"], results_dict["PRAUC"]["pr_auc_unc"]]
        results_table.append(ll)

pd.DataFrame(results_table, columns =['signal', 'scoring metric', 'rocauc', 'rocauc unc', 'prauc', 'prauc unc']).to_csv(path_or_buf="results/isotree/reduced/results_kfold.csv")