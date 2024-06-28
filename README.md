# Welcome to the repo of my master thesis!✨

*Foraging for new physics: machine learning-based, model-agnostic and real-time new physics searches at the 40 MHz Scouting system of CMS.*

What does this mean?

I am developing a stream line where we used **isolation forest** based anomaly detection algorithms to find new physics at the 40 MHz **Scouting system** of CMS (which is where the name Foraging for New Physics comes from). This entails several stages.

👉🏻  1) **Finding the "right" model.** For this, I use the david-cortes/isotree library. I benchmark different isolation forest models for outlier/anomaly (== new physics) detection on a Monte Carlo dataset, that aims at emulating the Level-1 Trigger conditions present at CMS. We use eight different types of beyond the standard model signals. The performance of the models is assessed with the means of ROC, AUROC, PRC and PRAUC. The uncertainty of the outgoing isotree models are determined by k-fold cross validation. The results are compared to a basic classifier and an convolutional autoencoder (which is similar to the current anomaly detection architecture (AXOL1TL, which is actually a variational autoencoder..))

The relevant scripts (to be run in this order) are:

    - do_crossvalidation.py
    
    - supervisedclassification.py
    
    - do_cv_other.py
    
    - do_cv_PID.py
    
    - cv_reduced_other.py
    
    - AXO_scan.py

👉🏻  2) **Evaluate them with NPLM.** NPLM (new physics learning machine, found at GaiaGrosso/NPLM_package) is used to evaluate these models further. Would we actually detect the anomalies "experimentally"? For that, I inject signal into the NuGun data sample and see how the anomaly score distributions but also the sculpting within a given dimension of interest changes!

Relevant script of it: - anomaly_detection_and_NPLM.py

👉🏻  3) **Think of how to implement at 40 MHz Scouting System.** If stage 2) is successful in its study, the next step would be to think about how one would actually implement this within a 40 MHz Scouting system... :)
