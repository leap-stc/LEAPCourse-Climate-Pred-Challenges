# **Project 3: notebooks**

This folder contains notebooks and resources for **Project 3**, where we reconstruct surface ocean pCO₂ from sparse observations using Earth System Model (ESM) outputs and machine learning techniques.

- **`Project3_Starter.ipynb`**  
  Main notebook for running ML training and evaluation on pre-selected ESMs and members using preprocessed inputs. Recommended for most users seeking an efficient workflow.
  
- **`Project3_Data.ipynb`**  
  A helper notebook that guides you through accessing raw ESM data from `ensemble_dir`, selecting custom members or models, and preprocessing them into ML-ready DataFrames. Use this only if you wish to go beyond the provided datasets.

## Data Overview

- The raw ESM outputs are stored in **`ensemble_dir`**, which includes many models and members.
- We have **preprocessed 4 ESMs**, each with **5 members**, and saved them under `MLinputs_path`.
- In the starter notebook, we **only use 3 ESMs × 3 members** to keep runtime and storage manageable.

## Storage Reminder

- Your workspace has a **25 GB limit**.
- Each preprocessed ML input DataFrame (covering 2004–2023) takes up about **2.35 GB**.
- You’ll also need space for model outputs and temporary files.
- Please **avoid re-downloading or duplicating files**, and delete unused data promptly.
- We recommend using the provided preprocessed data in `MLinputs_path` unless absolutely necessary.
