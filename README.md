# Online Shoppers Purchasing Intention -- Classification

## 1. Problem Statement

The objective of this project is to develop and evaluate five machine-learning classification models to predict whether an online shopping session will result in a purchase (`Revenue = True`) or not (`Revenue = False`). The models are trained and evaluated on the same dataset using a common training/testing split. An interactive Streamlit application is provided to explore the dataset, compare model performance, and make predictions using uploaded test data.

## 2. Dataset Description

  ---------------------------------------------------------------------------------------------------------------------------------------
  Item                                Description
  ----------------------------------- ---------------------------------------------------------------------------------------------------
  Dataset name                        Online Shoppers Purchasing Intention Dataset

  Source                              [UCI Machine Learning
                                      Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

  Original number of instances        12,330

  Number of input features            17

  Target variable                     `Revenue`

  Classification type                 Binary classification

  Target classes                      `False` -- no purchase; `True` -- purchase

  Numerical features                  10

  Categorical features                7

  Data preparation                    125 exact duplicate rows removed before splitting

  Records used after duplicate        12,205                             
  removal

  Train/test split                    80% training and 20% testing, stratified
  
  Random state                        42
  ---------------------------------------------------------------------------------------------------------------------------------------

**Input features**

-   **Numerical:** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`,
    `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`.
-   **Categorical:** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

**Preprocessing**

-   Numerical features: median imputation followed by standardization using `StandardScaler`.
-   Categorical features: most-frequent imputation followed by one-hot encoding using `OneHotEncoder(handle_unknown="ignore")`.
-   The preprocessing steps are included in each model pipeline so the same transformations are applied during training and prediction.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/Shubhrajyoti24/Online_Shoppers_Classification

The repository contains the application source code (`app.py`), dependency list (`requirements.txt`), this README, the original dataset,
test data, five saved model pipelines, and generated evaluation outputs.

## 4. Models Used

All five classifiers were trained using the same dataset and the same stratified 80:20 train/test split.

1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbors (KNN) Classifier
4.  Gaussian Naive Bayes Classifier
5.  Support Vector Machine (SVM) with an RBF kernel

### 4.1 Model Comparison -- Evaluation Metrics

The following table reports the six required evaluation metrics for each
classifier. **Replace the blank cells with the actual values generated
by your application in `results/model_comparison.csv`. Do not enter
estimated or example values.**

  ML Model Name              Accuracy   AUC    Precision   Recall    F1      MCC
  ------------------------ ---------- -------  ---------- -------- -------  -------
  Logistic Regression        0.8869    0.8996    0.7524    0.4136   0.5338   0.5032
  
  Decision Tree                   ---   ---         ---      ---   ---   ---
  
  KNN                             ---   ---         ---      ---   ---   ---
  
  Naive Bayes (Gaussian)          ---   ---         ---      ---   ---   ---
  
  Support Vector Machine          ---   ---         ---      ---   ---   ---
  

### 4.2 Observations on Model Performance

  -----------------------------------------------------------------------
  ML Model Name                       Observation about model performance
  ----------------------------------- -----------------------------------
  Logistic Regression                 Add an observation based on the
                                      model's actual Accuracy, AUC,
                                      Precision, Recall, F1 and MCC.

  Decision Tree                       Add an observation based on the
                                      model's actual metrics and
                                      confusion matrix.

  KNN                                 Add an observation based on the
                                      model's actual metrics and
                                      confusion matrix.

  Naive Bayes                         Add an observation based on the
                                      model's actual metrics and
                                      confusion matrix.

  Support Vector Machine              Add an observation based on the
                                      model's actual metrics and
                                      confusion matrix.

  **Overall winner for this dataset** Identify the model that performs
                                      best overall and justify the choice
                                      using the evaluation metrics.
  -----------------------------------------------------------------------

**Note:** The target classes are imbalanced. Therefore, model
performance should not be judged using accuracy alone. Consider AUC,
precision, recall, F1 and MCC together when identifying the strongest
model.

## 5. Streamlit Application

**Live Streamlit App:** https://online-shoppers-classification-2026aj05524-assignment.streamlit.app/

The application is deployed using Streamlit Community Cloud and provides the following functionality:

-   Dataset overview and exploration.
-   Evaluation of the five trained classification models.
-   Comparison of model performance using the required metrics.
-   Confusion-matrix visualization.
-   Interactive prediction using uploaded CSV data.

## 6. Project Structure

``` text
Online_Shoppers_Classification/
│
├── app.py
├── online_shoppers_intention.csv
├── test_data.csv
├── requirements.txt
├── README.md
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

## 7. How to Run the Application Locally

1.  Clone the GitHub repository.
2.  Install the dependencies listed in `requirements.txt`.
3.  Run the following command from the project directory:

``` bash
streamlit run app.py
```

## 8. Technologies Used

-   Python
-   Streamlit
-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib
-   Joblib
