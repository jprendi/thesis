from scripts.cross_validation_script_othermodels import Cross_Validation
import pickle
import pandas as pd


SCiForest = {'ndim':2, 'sample_size':256, 'max_depth':8, 'ntrees':100, 'missing_action':"fail", 'coefs':"normal", 'ntry':10, 'prob_pick_avg_gain':1, 'penalize_range':True}
FCF = {'ndim':2, 'sample_size':256, 'max_depth': None, 'ntrees':200, 'missing_action':"fail", 'coefs':"normal", 'ntry':1, 'prob_pick_pooled_gain':1}
RRCF = {'ndim':1, 'sample_size':256, 'max_depth':None, 'ntrees':100, 'missing_action':"fail", 'prob_pick_col_by_range':1}
algo_dict = {'SciForest':SCiForest, 'FCF':FCF, 'RRCF':RRCF}
algos = ['SciForest', 'FCF', 'RRCF']

sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]


for algo in algos:
    kwargs = algo_dict[algo]
    print(kwargs, algo)
    cv = Cross_Validation(kfold=5, model_parameters=kwargs, train_models = True)
    for sig in sigkeys:
        print(sig)
        cv.getnsave_kfold_auc_prauc(sigkey=sig)

results_table = []

for sigkey in sigkeys:
    for algo in algos:
        kwargs = algo_dict[algo]
        results = "results/isotree/other_models/" + f"{sigkey}_" + "__".join([f"{key}_{value}" for key, value in kwargs.items()]) + "__" + "5"
        
        with open(results, 'rb') as f:
            results_dict = pickle.load(f)
        # print(results)
        ll = [sigkey, algo, results_dict["ROCAUC"]["auc_mean"], results_dict["ROCAUC"]["auc_unc"], results_dict["PRAUC"]["pr_auc_mean"], results_dict["PRAUC"]["pr_auc_unc"]]
        results_table.append(ll)

pd.DataFrame(results_table, columns =['signal', 'algo', 'rocauc', 'rocauc unc', 'prauc', 'prauc unc']).to_csv(path_or_buf="results/isotree/other_models/results_others.csv")