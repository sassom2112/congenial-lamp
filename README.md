# Exploratory Data Analysis and Preprocessing of the UNSW-NB15 Network Traffic Dataset

## Project Overview

This project involves the **Exploratory Data Analysis (EDA)** and **preprocessing** of the **UNSW-NB15 dataset**, which contains network traffic data for intrusion detection. The dataset includes features such as source and destination IP addresses, port numbers, protocols, and attack categories, which are used to detect potential network intrusions. The goal of this project is to clean, transform, and visualize the dataset to prepare it for machine learning models.

## Dataset Information

The UNSW-NB15 dataset is publicly available and is widely used for network intrusion detection system (NIDS) research. The dataset contains labeled network traffic data, including both normal and attack traffic, making it useful for training and testing intrusion detection systems.

- **Dataset Source**: [UNSW-NB15 Dataset](https://www.unsw.edu.au/engineering/our-story/our-story)
- **Data Format**: CSV files containing network traffic records.

## Key Steps in the Project

1. **Data Loading**:
   - The dataset is loaded from multiple CSV files containing network traffic data.
   - Features include numerical and categorical variables such as source IP, destination IP, protocol, and attack category.

2. **Data Cleaning**:
   - Non-numeric columns like IP addresses are handled or transformed.
   - Missing values and incorrect data types are identified and addressed.

3. **Feature Engineering**:
   - Categorical features such as attack categories are cleaned and prepared for machine learning models.
   - Data is aggregated to ensure consistent categories and reduced complexity.

4. **Exploratory Data Analysis (EDA)**:
   - Distribution of numerical and categorical features is visualized using histograms, count plots, and bar charts.
   - Correlations among numerical features are visualized through a correlation heatmap.
   - Attack categories and subcategories are analyzed to understand the frequency distribution of events.

5. **Data Preprocessing**:
   - Features are scaled and transformed where necessary to ensure consistency and improve the performance of machine learning models.


## Next Steps
1. Feature Engineering:
    - Further process the data by encoding categorical variables or transforming features for machine learning models.
2. Model Training:
    - Train machine learning models like Random Forest, Logistic Regression, or XGBoost on the cleaned and preprocessed data.
3. Evaluation:
    - Evaluate the model performance using metrics like accuracy, precision, recall, and F1-score.
4. Optimization:
    - Tune model parameters and perform hyperparameter optimization to improve the results.