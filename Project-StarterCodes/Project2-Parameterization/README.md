# Project 2: Parameterizing Vertical Mixing Coefficients in the Ocean Surface Boundary Layer Using Neural Networks

In this project, you will extend existing work on parameterization of upper ocean mixing using a neural network (Sane et al. 2023), and write a "data story" that can be shared with a scientific audience. You will need to learn basic concepts of upper ocean mixing, understand why these critical small-scale processes must be represented in ocean and climate models, and how machine learning can be used to bridge existing gaps in this space. 

The starter notebook reproduces and extends a portion of the analysis from Sane et al. (2023), "Parameterizing vertical mixing coefficients in the ocean surface boundary layer using neural networks." The study explores the use of machine learning techniques to improve the parameterization of vertical mixing coefficients in the ocean surface boundary layer.

## Folder Structure

To reduce the complexity of the main notebook, we have developed three additional scripts in the lib/ directory. You can also use these as they are in your project, modify these, or choose not to use them. 

Please make your primary notebook the full "data story" and use a comparable file structure for helper files. 

```bash
Project2-Parameterization/
├── Data/                 
│   ├── README.md         # Detailed information on data sources
├── lib/                  # Utility scripts and helper functions
│   ├── func_file.py
│   ├── visual_figure3.py
│   ├── visual_figure4.py
├── notebooks/            # Jupyter notebooks for experiments
│   ├── Project2_Starter.ipynb
├── README.md             # Project documentation
```

## **Running this Notebook**:

This notebook is designed for use on LEAP-Pangeo's GPU cluster (Pytorch option).

## **Contribution statement**: ([more information and examples](doc/a_note_on_contributions.md))  

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz).

GitHub Sharing: Recommended for collaborative coding and version control.

We have prepared a [GitHub Tutorial](https://github.com/leap-stc/LEAPCourse-Climate-Pred-Challenges/blob/main/Tutorials/Github-Tutorial.md) to assist with this process.

If you encounter any issues, feel free to reach out.

## References
Sane et al. (2023), "Parameterizing vertical mixing coefficients in the ocean surface boundary layer using neural networks." Journal of Advances in Modeling Earth Systems, 15(10), e2023MS003890.
