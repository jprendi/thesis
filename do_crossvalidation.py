from scripts import cross_validation_script
# takes approx 160 mins
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