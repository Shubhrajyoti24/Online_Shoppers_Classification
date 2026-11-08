# Online Shoppers Purchasing Intention -- Classification

## 1. Problem Statement

The objective of this project is to develop and evaluate five machine-learning classification models to predict whether an online shopping session will result in a purchase (`Revenue = True`) or not (`Revenue = False`). The models are trained and evaluated on the same dataset using a common training/testing split. An interactive Streamlit application is provided to explore the dataset, compare model performance, and make predictions using uploaded test data.

## 2. Dataset Description

| Item | Description |
|---|---|
| Dataset name | Online Shoppers Purchasing Intention Dataset |
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| Original number of instances | 12,330 |
| Number of input features | 17 |
| Target variable | `Revenue` |
| Classification type | Binary classification |
| Target classes | `False` -- no purchase; `True` -- purchase |
| Numerical features | 10 |
| Categorical features | 7 |
| Data preparation | 125 exact duplicate rows removed before splitting |
| Records used after duplicate removal | 12,205 |
| Train/test split | 80% training and 20% testing, stratified |
| Random state | 42 |

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
classifier.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8869 | 0.8996 | 0.7524 | 0.4136 | 0.5338 | 0.5032 |
| Decision Tree | 0.8558 | 0.7376 | 0.5373 | 0.5654 | 0.5510 | 0.4654 |
| KNN | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Naive Bayes (Gaussian) | 0.2773 | 0.7502 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8955 | 0.8733 | 0.7361 | 0.5183 | 0.6083 | 0.5614 |

### 4.2 Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved good accuracy (88.69%) and the highest AUC score (0.8996), indicating strong classification capability. However, its recall (41.36%) is relatively low, meaning it misses some actual purchase sessions. |
| Decision Tree | Achieved moderate accuracy (85.58%) with a recall of 56.54%, identifying more actual purchase sessions than Logistic Regression and KNN. Its AUC (0.7376) is the lowest among the five models, indicating comparatively weaker overall discrimination. |
| KNN | Achieved 87.34% accuracy with an AUC of 0.7958, showing reasonable classification performance. However, its recall (39.01%) and F1 score (0.4909) are relatively low, indicating difficulty identifying actual purchase sessions. |
| Naive Bayes | Achieved the highest recall (97.91%), meaning it identifies almost all actual purchase sessions. However, its accuracy (27.73%) and precision (17.56%) are very low, indicating a large number of false-positive predictions. |
| Support Vector Machine | Achieved the highest accuracy (89.55%), F1 score (0.6083), and MCC (0.5614), demonstrating the strongest balanced performance overall. Its AUC (0.8733) is also high, although slightly lower than Logistic Regression. |
| **Overall winner for this dataset** | Support Vector Machine (SVM) is the overall winner because it achieves the highest accuracy, F1 score, and MCC, offering the best balance among the evaluated models. |


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
