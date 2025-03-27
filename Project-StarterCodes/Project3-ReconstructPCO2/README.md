# **Project 3: Machine Learning Reconstruction of Surface Ocean pCO₂**

This project reproduces and extends portions of the analysis presented by Gloege et al. (2020) and Heimdal et al. (2024), using machine learning to reconstruct surface ocean partial pressure of CO₂ (pCO₂) and evaluate reconstruction performance under sparse observational coverage.

The notebook implements a **pCO₂-Residual** approach with an **XGBoost** model to improve upon standard pCO₂ reconstructions by isolating and removing the temperature-driven signal prior to machine learning regression. It also evaluates performance using data from the **Large Ensemble Testbed (LET)**.


## **Folder Structure**

To reduce complexity in the main notebook, several helper functions and figures are modularized into the `lib/` directory. You may modify these as needed for your project.

```bash
Project3/
├── lib/                       # Helper scripts
│   ├── __init__.py
│   ├── bias_figure2.py        # Code for bias calculation and visualization
│   ├── corr_figure3.py        # Code for correlation calculation and visualization
│   ├── residual_utils.py      # Prepares data for ML, tools for dataset splitting, model evaluation, and saving files.
│   └── visualization.py       # Custom plotting class SpatialMap2 for creating high-quality spatial visualizations with colorbars and map features using Cartopy and Matplotlib.
├── notebooks/
│   └── Project3_Starter.ipynb # Main notebook containing full analysis & data story
```

 ## **Objective**

This project aims to:
1. Implement an **XGBoost-based pCO₂-Residual reconstruction**.
2. Evaluate reconstruction accuracy using **bias and correlation metrics**.
3. Compare reconstructions against gridded output from the **Large Ensemble Testbed (LET)**.



## **References**

- **Gloege et al. (2020)**  
  *Quantifying Errors in Observationally Based Estimates of Ocean Carbon Sink Variability.*  
  [DOI: 10.1029/2020GB006788](https://doi.org/10.1029/2020GB006788)

- **Heimdal et al. (2024)**  
  *Assessing improvements in global ocean pCO₂ machine learning reconstructions with Southern Ocean autonomous sampling.*  
  [DOI: 10.5194/bg-21-2159-2024](https://doi.org/10.5194/bg-21-2159-2024)


## **Contributions and Collaboration**

We encourage collaborative version control using GitHub. If working in a team, please follow contribution guidelines (e.g., commit messages, branches).

You may find [this GitHub tutorial](https://github.com/leap-stc/LEAPCourse-Climate-Pred-Challenges/blob/main/Tutorials/Github-Tutorial.md) helpful for getting started.


