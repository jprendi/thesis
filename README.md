# Welcome to the repo of my master thesis!✨

*Foraging for new physics: machine learning-based, model-agnostic and real-time new physics searches at the 40 MHz Scouting system of CMS.*

What does this mean?

I am developing a stream line where we used **isolation forest** based anomaly detection algorithms to find new physics at the 40 MHz **Scouting system** of CMS (which is where the name Foraging for New Physics comes from). This entails several stages.

👉🏻  1) **Finding the "right" model.** For this, I use the david-cortes/isotree library. I benchmark different isolation forest models for outlier/anomaly (== new physics) detection on a Monte Carlo dataset, that aims at emulating the Level-1 Trigger conditions present at CMS. We use eight different types of beyond the standard model signals. The performance of the models is assessed with the means of ROC, AUROC, PRC and PRAUC. The uncertainty of the outgoing isotree models are determined by k-fold cross validation. The results are compared to a basic classifier and to the current anomaly detection architecture (AXOL1TL).

Here it is important to first run do_crossvalidation.py and then supervisedclassification.py . 

