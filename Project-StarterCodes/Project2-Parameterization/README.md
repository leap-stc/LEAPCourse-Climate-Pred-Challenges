# Project 2 – Condtional Modeling Approach for Parametrizing Ocean Mixing

This project extends the work of Sane et al. (2023), "Parameterizing Vertical Mixing Coefficients in the Ocean Surface Boundary Layer Using Neural Networks", by exploring whether specialized models based on distinct ocean regimes can outperform a universal neural network in predicting vertical mixing shape functions ( g(sigma) ).

---

## Motivation

Vertical mixing in the ocean surface boundary layer is driven by complex, small-scale processes. Traditional climate models use fixed parameterizations which cannot fully capture variability across different regions and seasons.

We investigate three complementary approaches:
- **Clustering based on physically interpretable features**
- **Clustering by input features** (e.g., Coriolis, buoyancy flux, wind stress) to group physical regimes
- **Clustering by shape function structure** (i.e., clustering on the outputs) to learn distinct vertical mixing behaviors

---


## Team & Contributions

**Group 5**  

- **Sarah Pariser**  
  Explored physical drivers (latitude, heat flux, wind stress) and their effect on shape structure; applied subsetting and filtering strategies; contributed to early data preprocessing.

- **Dhruv Chakraborty**  
  Led the design of input- and shape-based clustering approaches using KMeans and PCA; trained neural networks for each cluster; built visualization functions.

- **Kihyun Jye**  
  Trained the baseline model and implemented a node-wise modeling strategy with early stopping; helped structure the data story to match rubric.

- **Sarika de Bruyn** Applied hierarchical clustering and data resampling techniques to balance training; experimented with Gaussian-shaped distributions and Mixture of Experts frameworks; helped structure final notebook, including adding relevant details for figures etc.

---

## References

Sane et al. (2023), *Parameterizing Vertical Mixing Coefficients in the Ocean Surface Boundary Layer Using Neural Networks*, Journal of Advances in Modeling Earth Systems, 15(10), e2023MS003890.  
DOI: https://doi./10.1029/2023MS003890