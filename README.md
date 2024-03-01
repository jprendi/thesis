# Welcome to the repo of my master thesis!✨

*Foraging for new physics: machine learning-based, model-agnostic new physics searches at the 40 MHz Scouting system of CMS.*

What does this mean?

I am developing a stream line where we used **isolation forest** based anomaly detection algorithms to find new physics at the 40 MHz **Scouting system** of CMS (which is where the name Foraging for New Physics comes from). This entails several stages.

👉🏻  1) **Finding the "right" model.** For this, I use the david-cortes/isotree library. I benchmark different isolation forest models for outlier/anomaly (== new physics) detection on a Monte Carlo dataset, that aims at emulating the Level-1 Trigger conditions present at CMS. We use three different types of beyond the standard model signals. A good performance metric has to be determined.

