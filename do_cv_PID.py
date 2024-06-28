'''
runs cross validation with the PIDForest model
'''

from scripts.cross_validation_script_PIDForest import Cross_Validation
import pickle
import pandas as pd


kwargs = {'max_depth': 10, 'n_trees': 10,  'max_samples': 100, 'max_buckets': 3, 'epsilon': 0.1, 'sample_axis': 1, 
  'threshold': 0}

sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]


cv = Cross_Validation(model_parameters=kwargs, train_models = True)

for sig in sigkeys:
    print(sig)
    cv.getnsave_kfold_auc_prauc(sigkey=sig)

results_table = []

for sigkey in sigkeys:
    results = "results/isotree/other_models/" + f"{sigkey}_" + "__PIDForest"+ "__" + f"5"
    with open(results, 'rb') as f:
        results_dict = pickle.load(f)
    ll = [sigkey, "PID", results_dict["ROCAUC"]["auc_mean"], results_dict["ROCAUC"]["auc_unc"], results_dict["PRAUC"]["pr_auc_mean"], results_dict["PRAUC"]["pr_auc_unc"]]
    results_table.append(ll)

pd.DataFrame(results_table, columns =['signal', 'PID', 'rocauc', 'rocauc unc', 'prauc', 'prauc unc']).to_csv(path_or_buf="results/isotree/other_models/results_PID_10.csv")