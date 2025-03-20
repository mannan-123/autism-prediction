# Autism Prediction Model

## Overview

This project builds a machine learning models to predict Autism Spectrum Disorder (ASD) based on various input features. It involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and hyperparameter tuning.

## Dataset

The dataset used in this project is `autism.csv`, which contains demographic and behavioral survey data.

## Features

- **Data Loading & Exploration:** Basic insights, missing values, and class distribution.
- **Feature Engineering:** Encoding categorical features and handling outliers.
- **Data Preprocessing:** SMOTE for class balancing and train-test split.
- **Model Training:** Evaluation of multiple classifiers.
- **Hyperparameter Tuning:** Optimizing the best model using `RandomizedSearchCV`.

## Libraries Used

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `sklearn` (preprocessing, model selection, metrics)
- `imblearn` (SMOTE for handling class imbalance)
- `xgboost`
- `pickle` (for model saving)

## Model Training & Evaluation

The following classifiers were tested:

- Decision Tree
- Random Forest
- XGBoost
- SVM
- KNN
- Gradient Boosting
- Naïve Bayes

The best model is selected based on cross-validation scores and fine-tuned using `RandomizedSearchCV`. The final model is saved as `best_model.pkl`.

## How to Run

1. Install dependencies:
   ```sh
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```
2. Run all cells of Jupyter file:

## Results

The model's performance is evaluated using accuracy, confusion matrix, and classification report on test data.

## Future Improvements

- Improve feature selection techniques
- Try deep learning models
- Deploy as a web service
