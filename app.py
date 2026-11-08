
# ================================================================
# ONLINE SHOPPERS PURCHASING INTENTION
# COMPLETE MACHINE LEARNING CLASSIFICATION PROJECT
#
# All implementation and Streamlit application in ONE Python file
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ================================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)


# ================================================================
# 2. PROJECT PATHS AND CONFIGURATION
# ================================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "online_shoppers_intention.csv"

MODEL_DIR = PROJECT_DIR / "model"

RESULTS_DIR = PROJECT_DIR / "results"

TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"

TARGET_COLUMN = "Revenue"

RANDOM_STATE = 42

TEST_SIZE = 0.20

MODEL_DIR.mkdir(exist_ok=True)

RESULTS_DIR.mkdir(exist_ok=True)


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}


# ================================================================
# 3. LOAD AND PREPARE DATASET
# ================================================================

def load_and_prepare_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "online_shoppers_intention.csv not found "
            "in the project folder."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            "Revenue target column is missing."
        )

    original_records = len(df)

    # Remove exact duplicate observations
    df = df.drop_duplicates().reset_index(drop=True)

    duplicates_removed = original_records - len(df)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN].astype(int)

    # Identify categorical features
    categorical_features = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend"
    ]

    numerical_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    return (
        df,
        X,
        y,
        numerical_features,
        categorical_features,
        duplicates_removed
    )


# ================================================================
# 4. CREATE PREPROCESSING PIPELINE
# ================================================================

def create_preprocessor(
    numerical_features,
    categorical_features
):

    numerical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "scaler",
                StandardScaler()
            )

        ]

    )

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )

        ]

    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )

        ]

    )

    return preprocessor


# ================================================================
# 5. DEFINE FIVE CLASSIFICATION MODELS
# ================================================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Gaussian Naive Bayes": GaussianNB(),

        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        )

    }

    return models


# ================================================================
# 6. CALCULATE EVALUATION METRICS
# ================================================================

def calculate_metrics(y_test, y_pred, y_prob):

    metrics = {

        "Accuracy": accuracy_score(
            y_test, y_pred
        ),

        "AUC": roc_auc_score(
            y_test, y_prob
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test, y_pred
        )

    }

    return metrics


# ================================================================
# 7. TRAIN OR LOAD ALL FIVE MODELS
# ================================================================

@st.cache_resource(show_spinner=False)
def train_or_load_models():

    (
        df,
        X,
        y,
        numerical_features,
        categorical_features,
        duplicates_removed
    ) = load_and_prepare_data()

    # Identical stratified split for every model
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    models = get_models()

    all_models_exist = all(

        (
            MODEL_DIR / filename
        ).exists()

        for filename in MODEL_FILES.values()

    )

    predictions = {}

    probabilities = {}

    results = []

    if all_models_exist:

        # Load previously saved trained pipelines
        for model_name, filename in MODEL_FILES.items():

            pipeline = joblib.load(
                MODEL_DIR / filename
            )

            models[model_name] = pipeline

    else:

        # Train and save all five pipelines
        for model_name, classifier in models.items():

            preprocessor = create_preprocessor(
                numerical_features,
                categorical_features
            )

            pipeline = Pipeline(

                steps=[

                    ("preprocessor", preprocessor),

                    ("classifier", classifier)

                ]

            )

            pipeline.fit(X_train, y_train)

            model_path = (
                MODEL_DIR / MODEL_FILES[model_name]
            )

            joblib.dump(
                pipeline,
                model_path
            )

            models[model_name] = pipeline

    # Evaluate all models using the same test set
    for model_name, pipeline in models.items():

        y_pred = pipeline.predict(X_test)

        y_prob = pipeline.predict_proba(X_test)[:, 1]

        predictions[model_name] = y_pred

        probabilities[model_name] = y_prob

        metrics = calculate_metrics(
            y_test,
            y_pred,
            y_prob
        )

        metrics["Model"] = model_name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]
    ]

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)

    # Export evaluation metrics
    results_df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    # Export held-out test observations
    test_data = X_test.copy()

    test_data[TARGET_COLUMN] = y_test.to_numpy()

    test_data.to_csv(
        TEST_DATA_PATH,
        index=False
    )

    # Save confusion matrix image
    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 10)
    )

    axes = axes.flatten()

    for index, model_name in enumerate(models):

        ConfusionMatrixDisplay.from_predictions(

            y_test,

            predictions[model_name],

            display_labels=[
                "No Purchase",
                "Purchase"
            ],

            cmap="Blues",

            ax=axes[index],

            colorbar=False

        )

        axes[index].set_title(model_name)

    axes[5].axis("off")

    fig.suptitle(
        "Confusion Matrices - Five Classification Models",
        fontsize=16
    )

    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {

        "df": df,

        "X_test": X_test,

        "y_test": y_test,

        "models": models,

        "predictions": predictions,

        "probabilities": probabilities,

        "results": results_df,

        "numerical_features": numerical_features,

        "categorical_features": categorical_features,

        "duplicates_removed": duplicates_removed

    }


# ================================================================
# 8. APPLICATION TITLE
# ================================================================

st.title("🛒 Online Shoppers Purchasing Intention")

st.markdown(
    """
    ### Machine Learning Classification Dashboard

    This application predicts whether an online shopping
    visitor is likely to make a purchase.

    Five classification algorithms are implemented and
    evaluated using six performance metrics.
    """
)

st.divider()


# ================================================================
# 9. LOAD MODELS AND RESULTS
# ================================================================

try:

    with st.spinner(
        "Loading dataset and preparing classification models..."
    ):

        data = train_or_load_models()

except Exception as error:

    st.error(f"Application error: {error}")

    st.info(
        "Check your dataset, installed libraries and model files."
    )

    st.stop()


df = data["df"]

models = data["models"]

results_df = data["results"]

X_test = data["X_test"]

y_test = data["y_test"]

predictions = data["predictions"]

probabilities = data["probabilities"]


st.success(
    "All five classification models are ready!"
)


# ================================================================
# 10. SIDEBAR
# ================================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select a section",

    [

        "Dataset Overview",

        "Model Evaluation",

        "Model Comparison",

        "Purchase Prediction"

    ]

)


# ================================================================
# 11. DATASET OVERVIEW
# ================================================================

if page == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Input Features",
        len(df.columns) - 1
    )

    col3.metric(
        "Numerical Features",
        len(data["numerical_features"])
    )

    col4.metric(
        "Categorical Features",
        len(data["categorical_features"])
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({

        "Column": df.columns,

        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],

        "Missing Values": [
            df[col].isnull().sum()
            for col in df.columns
        ]

    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.metric(
        "Duplicate Records Removed",
        data["duplicates_removed"]
    )

    st.subheader("Purchase Distribution")

    target_counts = df[TARGET_COLUMN].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["No Purchase", "Purchase"],
        [
            target_counts.get(False, 0),
            target_counts.get(True, 0)
        ]
    )

    ax.set_title("Revenue Class Distribution")

    ax.set_xlabel("Purchase Status")

    ax.set_ylabel("Number of Sessions")

    st.pyplot(fig)

    plt.close(fig)


# ================================================================
# 12. MODEL EVALUATION
# ================================================================

elif page == "Model Evaluation":

    st.header("Individual Model Evaluation")

    selected_model = st.selectbox(
        "Select Classification Model",
        list(MODEL_FILES.keys())
    )

    selected_result = results_df[
        results_df["Model"] == selected_model
    ].iloc[0]

    st.subheader("Evaluation Metrics")

    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]

    columns = st.columns(3)

    for index, metric in enumerate(metric_names):

        columns[index % 3].metric(
            metric,
            f"{selected_result[metric]:.4f}"
        )

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(

        y_test,

        predictions[selected_model],

        display_labels=[
            "No Purchase",
            "Purchase"
        ],

        cmap="Blues",

        ax=ax

    )

    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("ROC Curve")

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities[selected_model]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {selected_result['AUC']:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel("False Positive Rate")

    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        f"ROC Curve - {selected_model}"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    st.pyplot(fig)

    plt.close(fig)


# ================================================================
# 13. MODEL COMPARISON
# ================================================================

elif page == "Model Comparison":

    st.header("Comparison of Five Classification Models")

    st.subheader("Performance Metrics Table")

    st.dataframe(

        results_df.style.format({

            "Accuracy": "{:.4f}",

            "AUC": "{:.4f}",

            "Precision": "{:.4f}",

            "Recall": "{:.4f}",

            "F1 Score": "{:.4f}",

            "MCC": "{:.4f}"

        }),

        use_container_width=True,

        hide_index=True

    )

    st.subheader("Model Performance Comparison")

    selected_metric = st.selectbox(

        "Select Evaluation Metric",

        [

            "Accuracy",

            "AUC",

            "Precision",

            "Recall",

            "F1 Score",

            "MCC"

        ]

    )

    chart_df = results_df.sort_values(
        selected_metric,
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        chart_df["Model"],
        chart_df[selected_metric]
    )

    ax.set_title(
        f"Model Comparison - {selected_metric}"
    )

    ax.set_xlabel("Classification Models")

    ax.set_ylabel(selected_metric)

    ax.tick_params(
        axis="x",
        rotation=25
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("All Confusion Matrices")

    confusion_image = (
        RESULTS_DIR / "confusion_matrices.png"
    )

    if confusion_image.exists():

        st.image(
            str(confusion_image),
            use_container_width=True
        )

    # Download evaluation results
    csv_results = results_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="Download Model Comparison CSV",

        data=csv_results,

        file_name="model_comparison.csv",

        mime="text/csv"

    )


# ================================================================
# 14. PURCHASE PREDICTION
# ================================================================

elif page == "Purchase Prediction":

    st.header("Predict Online Shopping Purchase Intention")

    st.write(
        """
        Upload a CSV file containing the 17 input features.
        The application will predict the purchase intention
        using your selected trained classification model.
        """
    )

    selected_model = st.selectbox(

        "Select Prediction Model",

        list(MODEL_FILES.keys()),

        key="prediction_model"

    )

    uploaded_file = st.file_uploader(

        "Upload Test CSV File",

        type=["csv"]

    )

    if uploaded_file is not None:

        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")

        st.dataframe(
            input_df.head(),
            use_container_width=True
        )

        # Remove target column if present
        prediction_df = input_df.drop(
            columns=[TARGET_COLUMN],
            errors="ignore"
        )

        expected_features = list(
            data["X_test"].columns
        )

        missing_features = [
            col for col in expected_features
            if col not in prediction_df.columns
        ]

        extra_features = [
            col for col in prediction_df.columns
            if col not in expected_features
        ]

        if missing_features:

            st.error(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        elif extra_features:

            st.warning(
                "Extra columns will be ignored: "
                + ", ".join(extra_features)
            )

            prediction_df = prediction_df[
                expected_features
            ]

        else:

            prediction_df = prediction_df[
                expected_features
            ]

        if not missing_features:

            if st.button(
                "Predict Purchase Intention",
                type="primary"
            ):

                try:

                    model = models[selected_model]

                    predicted_classes = model.predict(
                        prediction_df
                    )

                    predicted_probabilities = (
                        model.predict_proba(
                            prediction_df
                        )[:, 1]
                    )

                    output_df = input_df.copy()

                    output_df["Predicted_Revenue"] = [
                        "Purchase" if value == 1
                        else "No Purchase"
                        for value in predicted_classes
                    ]

                    output_df["Purchase_Probability"] = (
                        predicted_probabilities
                    )

                    st.subheader("Prediction Results")

                    st.dataframe(
                        output_df,
                        use_container_width=True
                    )

                    purchase_count = int(
                        np.sum(predicted_classes == 1)
                    )

                    no_purchase_count = int(
                        np.sum(predicted_classes == 0)
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Predicted Purchases",
                        purchase_count
                    )

                    col2.metric(
                        "Predicted No Purchases",
                        no_purchase_count
                    )

                    st.download_button(

                        "Download Prediction Results",

                        data=output_df.to_csv(
                            index=False
                        ).encode("utf-8"),

                        file_name="purchase_predictions.csv",

                        mime="text/csv"

                    )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )


# ================================================================
# 15. FOOTER
# ================================================================

st.divider()

st.caption(
    "Online Shoppers Purchasing Intention | "
    "Machine Learning Classification Project | "
    "Five Classification Models"
)
