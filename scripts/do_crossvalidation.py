"""
This script here performs the cross validation over the several signals and scoring_metrics. The script:
                    - trains & saves all the models
                    - performs cross validation with the means of ROC, PR and respective AUC and gives also uncertainty
                    - saves all the data from the cross validation needed for a plot
                    - saves it all in a .csv file to be viewed :)

It is important to keep the sig in sigkeys loop in the innerst loop as this ensures the code to run "fast" 
(although it is not fast and the whole script will take approx 160 mins to run because I couldn't enable multithreading). 

For multithreading, make sure to follow the guide given on: 
https://github.com/david-cortes/isotree/blob/master/README.md 

Unfortunately macOS+python+multithreading doesn't seem to be best friends. I then also followed this guide: 
https://github.com/david-cortes/installing-optimized-libraries. I then also contacted the author and got as additional advice:
    - Uninstall this library.
    - Install the library again, without using any cached wheel it might have left before (e.g. 'pip install --no-cache-dir -U isotree')
This unfortunately couldn't help my problem either. 

If it works out, one could theoretically put e.g. "nthreads": 10 within the kwargs dictionary and then theoretically it should work. :)
"""

import cross_validation_script

kwargs = {"ntrees":100, "scoring_metric": "depth"}
sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm", "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]
scoring_metrics = ["depth", "density", "adj_depth", "adj_density", "boxed_ratio", "boxed_density2", "boxed_density"]

for metric in scoring_metrics:
    kwargs["scoring_metric"] = metric
    print(metric)
    cv = cross_validation_script.Cross_Validation(kfold=5, model_parameters=kwargs, train_models = True)
    for sig in sigkeys:
        print(sig)
        cv.getnsave_kfold_auc_prauc(sigkey=sig)

results_table = []

for sigkey in sigkeys:
    for scoring_metric in scoring_metrics:
        results = f"results/isotree/kfold_models/{sigkey}_ntrees_100__scoring_metric_{scoring_metric}__5"
        with open(results, 'rb') as f:
            results_dict = pickle.load(f)
        # print(results)
        ll = [sigkey, scoring_metric, results_dict["ROCAUC"]["auc_mean"], results_dict["ROCAUC"]["auc_unc"], results_dict["PRAUC"]["pr_auc_mean"], results_dict["PRAUC"]["pr_auc_unc"]]
        results_table.append(ll)