# Tropical Cyclone Track Clustering Analysis
### Spring 2025 - Climate Prediction Challenges Project 1

## Team Members
+ Ahinoam Toubia
+ Dhruv Chakraborty
+ Martin Celedon
+ Xingyao Li

## Project Summary
Using historical storm data from the North Atlantic and West Pacific, our research applied different clustering methods to classify tropical cyclone tracks based on their movement patterns. By representing each storm’s trajectory using mass moments capturing its center of motion and how much it deviates, we tested three clustering methods: K-Means, Gaussian Mixture Models (GMM), and Spectral Clustering.

Our findings revealed key differences between Atlantic hurricanes and Pacific typhoons. In the Atlantic, K-Means worked well, grouping storms into clear pathways—those heading into the Gulf of Mexico, those traveling up the U.S. East Coast, and those veering toward Europe. However, in the West Pacific, typhoon paths were far more varied and unpredictable, often moving toward Asia or looping in erratic patterns. As a result, Gaussian Mixture Models (GMM) performed better. Spectral Clustering proved useful in both basins, particularly for identifying storms with nonlinear paths.

Despite these differences, Atlantic hurricanes and Pacific typhoons showed distinct groupings based on their trajectories, confirming that clustering techniques can help classify storm movements. More importantly, these insights suggest that storm forecasting in the West Pacific may require more advanced probabilistic models, while simpler clustering methods can still be effective for Atlantic hurricanes. As climate change continues to alter storm behavior, improving these predictive tools to understand which cyclones are more likey to be devastating will be essential for issuing earlier warnings and protecting communities.

## Key Features
- Implementation of multiple clustering algorithms
- Comprehensive evaluation metrics
- Comparison between North Atlantic and West Pacific basins
- Analysis of cluster characteristics and trends

## Required Libraries
- xarray
- numpy
- matplotlib
- cartopy
- scikit-learn
- seaborn
- pandas

## File Structure
project/
├── data/
│   └── README.md
├── doc/
│   └── project1_desc.md
├── figs/
│   └── README.md
├── lib/
│   └── README.md
├── output/
│   └── README.md
└── README.md

## How to Run
1. Install required packages
2. Run the Jupyter notebook
3. Results will be saved in the output directory

**Contribution statement**: All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz).