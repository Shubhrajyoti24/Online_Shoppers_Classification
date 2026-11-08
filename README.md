# Online Shoppers Purchasing Intention – Machine Learning Classification

**Name** : Shubhrajyoti Das 
**Bits Id** : 2026AJ05524 
**Sem** : 1st 
**ML Assignment 2** 

## 1. Project Overview

This project predicts whether an online shopping visitor is likely to make a purchase based on their browsing behaviour and session information.

The project implements and compares five machine learning classification algorithms using the UCI Online Shoppers Purchasing Intention Dataset.

An interactive web application is developed using Streamlit. Users can explore the dataset, evaluate individual models, compare model performance, and upload CSV files to generate purchase predictions.

## 2. Dataset Information

**Dataset:** Online Shoppers Purchasing Intention Dataset
**Source:** UCI Machine Learning Repository
**Dataset URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

- Original records: 12,330
- Input features: 17
- Target variable: Revenue
- Classification type: Binary classification
- Target classes:
  - False – No Purchase
  - True – Purchase

The dataset contains 10 numerical features and 7 categorical features.

### Target Variable

The target variable is `Revenue`.

- `False`: The visitor did not complete a purchase.
- `True`: The visitor completed a purchase.

## 3. Machine Learning Algorithms

The following five classification algorithms are implemented:

1. **Logistic Regression** – A linear classification algorithm used for binary classification.
2. **Decision Tree** – A tree-based algorithm that makes predictions using decision rules.
3. **K-Nearest Neighbors (KNN)** – A distance-based algorithm that classifies observations using their nearest neighbours.
4. **Gaussian Naive Bayes** – A probabilistic classification algorithm based on Bayes' theorem.
5. **Support Vector Machine (SVM)** – A classification algorithm that identifies an optimal decision boundary between classes.

All five models use the same training and testing dataset split for a fair comparison.

## 4. Data Preprocessing

The following preprocessing steps are implemented:

- Remove exact duplicate records.
- Separate input features and target variable.
- Identify numerical and categorical features.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Apply stratified sampling to maintain the target class distribution.
- Handle missing numerical values using median imputation.
- Handle missing categorical values using most-frequent imputation.
- Apply StandardScaler to numerical features.
- Apply OneHotEncoder to categorical features.

A preprocessing pipeline is integrated with each classification model.

## 5. Model Evaluation Metrics

Each classification model is evaluated using six performance metrics:

1. **Accuracy:** Measures the proportion of correctly classified observations.
2. **AUC Score:** Measures the model's ability to distinguish between the two classes.
3. **Precision:** Measures the proportion of predicted purchases that are actual purchases.
4. **Recall:** Measures the proportion of actual purchases correctly identified.
5. **F1 Score:** Represents the harmonic mean of precision and recall.
6. **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classifications while considering all four confusion matrix outcomes.

These metrics are used to compare the performance of all five classification algorithms.

## 6. Streamlit Application Features

The interactive application provides four main sections.

### A. Dataset Overview
- View the dataset dimensions.
- Explore the dataset preview.
- View numerical and categorical feature information.
- Check missing values.
- View duplicate records removed.
- Visualize the purchase distribution.

### B. Model Evaluation
- Select an individual classification model.
- View all six evaluation metrics.
- Visualize its confusion matrix.
- View its ROC curve.

### C. Model Comparison
- Compare all five models using a performance metrics table.
- Select a metric for graphical comparison.
- View the combined confusion matrix visualization.
- Download model comparison results as a CSV file.

### D. Purchase Prediction
- Upload a CSV file containing the 17 input features.
- Select a trained classification model.
- Generate purchase predictions.
- View predicted purchase probabilities.
- Download prediction results as a CSV file.

## 7. Project Structure

```text
Online_Shoppers_Classification/
│
├── app.py
│
├── online_shoppers_intention.csv
├── test_data.csv
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
└── results/
    ├── model_comparison.csv
    └── confusion_matrices.png
```

All Python implementation and Streamlit application code are contained in a single file: `app.py`.
The trained models are saved in the `model/` directory using Joblib.
The `results/` directory contains model comparison results and confusion matrix visualizations generated by the application.

## 8. Technologies and Libraries Used

- Python 3.15.2
- Streamlit 1.67.0
- NumPy 2.5.4
- Pandas 3.1.1
- Scikit-learn 1.9.1
- Matplotlib 3.11.2
- Joblib 1.6.0

## 9. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/Shubhrajyoti24/Online_Shoppers_Classification
```

### Step 2: Navigate to the Project Folder

```bash
cd Online_Shoppers_Classification
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

On its first run, the application trains and saves the five classification models if all five saved model files are not already available.

## 10. Deployment

The application is deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload the complete project to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Connect the GitHub repository.
4. Select the `main` branch.
5. Set the main file path to `app.py`.
6. Deploy the application.

Streamlit Community Cloud installs the required libraries using `requirements.txt` and hosts the application.

**Live Application:** https://online-shoppers-classification-2026aj05524-assignment.streamlit.app/

**GitHub Repository:** https://github.com/Shubhrajyoti24/Online_Shoppers_Classification

---