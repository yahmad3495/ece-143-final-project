# ECE-143 Final Project

repository for **ECE-143 Final Project**! This repository contains all the code and resources developed for our project.

## Project Structure

- **`ece143_grp12_final.py`**:  
  This is the main Python file containing all the code for the project. It is organized into two main sections:  
  1. **Dataset Cleaning**:  
     - Processes the dataset `datasets/san_diego_listings.csv`.  
     - Produces a cleaned version of the dataset named `san_diego_listing_cleaned.csv`.  
  2. **Data Analysis and Visualization**:  
     - Performs data analysis on the cleaned dataset.  
     - Generates graphs to visualize the insights.

- **`datasets/san_diego_listings.csv`**:  
  The original dataset used for analysis.  

- **`san_diego_listing_cleaned.csv`**:  
  The cleaned version of the dataset created during the data cleaning process.

## Libraries Used

The following Python libraries are used in this project:  
```python
import pandas as pd      # For data manipulation and analysis
import numpy as np       # For numerical computations
from functools import reduce  # For functional programming tools
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns    # For enhanced data visualizations
