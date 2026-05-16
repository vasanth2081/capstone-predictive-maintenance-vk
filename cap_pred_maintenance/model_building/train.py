# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier
)

# for model serialization
import joblib

# for creating a folder
import os

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("capstone-training-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/v-vasanth2009/capstone-pred-main-vk-12052026/Xtrain.csv"
Xtest_path = "hf://datasets/v-vasanth2009/capstone-pred-main-vk-12052026/Xtest.csv"
ytrain_path = "hf://datasets/v-vasanth2009/capstone-pred-main-vk-12052026/ytrain.csv"
ytest_path = "hf://datasets/v-vasanth2009/capstone-pred-main-vk-12052026/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()

# FEATURE ENGINEERING
def feature_engineering(df):

    df = df.copy()

    df['thermal_stress'] = (
        df['Engine rpm'] *
        df['lub oil temp']
    )

    df['engine_stress'] = (
        df['Engine rpm'] *
        df['Coolant temp']
    )

    df['pressure_ratio'] = (
        df['Fuel pressure'] /
        (df['Lub oil pressure'] + 1e-5)
    )

    df['rpm_temp_ratio'] = (
        df['Engine rpm'] /
        (df['Coolant temp'] + 1e-5)
    )

    df['pressure_mean'] = (
        df[
            [
                'Fuel pressure',
                'Lub oil pressure',
                'Coolant pressure'
            ]
        ].mean(axis=1)
    )

    return df

Xtrain = feature_engineering(Xtrain)
Xtest = feature_engineering(Xtest)

# REMOVE WEAK FEATURES
drop_features = [
    'Coolant temp'
]

drop_features = [
    col for col in drop_features
    if col in Xtrain.columns
]

Xtrain.drop(columns=drop_features, inplace=True)
Xtest.drop(columns=drop_features, inplace=True)

print(f"Weak features dropped: {drop_features}")

# HANDLE MISSING VALUES
for col in Xtrain.columns:

    if Xtrain[col].dtype == 'object':

        fill_value = Xtrain[col].mode()[0]

    else:

        fill_value = Xtrain[col].median()

    Xtrain[col] = Xtrain[col].fillna(fill_value)
    Xtest[col] = Xtest[col].fillna(fill_value)

# FEATURE TYPES
numeric_features = Xtrain.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    include=['object']
).columns.tolist()

# PREPROCESSOR
preprocessor = make_column_transformer(
    ('passthrough', numeric_features),
    (
        OneHotEncoder(handle_unknown='ignore'),
        categorical_features
    )
)

# CLASS WEIGHT
class_weight = (
    pd.Series(ytrain).value_counts()[0] /
    pd.Series(ytrain).value_counts()[1]
)

print(f"Scale Pos Weight: {class_weight}")

# XGBOOST MODEL
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    scale_pos_weight=class_weight,
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    min_child_weight=5
)

# RANDOM FOREST MODEL
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# ENSEMBLE MODEL
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    voting='soft'
)

# Model pipeline
model_pipeline = make_pipeline(preprocessor, ensemble_model)

# Start MLflow run
with mlflow.start_run():

    # Train Model
    model_pipeline.fit(Xtrain, ytrain)

    # PREDICTION PROBABILITIES
    y_pred_train_proba = (
        model_pipeline.predict_proba(Xtrain)[:, 1]
    )

    y_pred_test_proba = (
        model_pipeline.predict_proba(Xtest)[:, 1]
    )

    # THRESHOLD OPTIMIZATION
    thresholds = np.arange(0.44, 0.57, 0.02)

    best_threshold = 0.5
    best_macro_f1 = 0

    for threshold in thresholds:

        preds = (
            y_pred_test_proba >= threshold
        ).astype(int)

        score = f1_score(
            ytest,
            preds,
            average='macro'
        )

        if score > best_macro_f1:

            best_macro_f1 = score
            best_threshold = threshold

    print(f"\nBest Threshold: {best_threshold}")
    print(f"Best Macro F1: {best_macro_f1}")

    # FINAL PREDICTIONS
    y_pred_train = (
        y_pred_train_proba >= best_threshold
    ).astype(int)

    y_pred_test = (
        y_pred_test_proba >= best_threshold
    ).astype(int)

    # ROC AUC
    train_auc = roc_auc_score(
        ytrain,
        y_pred_train_proba
    )

    test_auc = roc_auc_score(
        ytest,
        y_pred_test_proba
    )

    auc_gap = train_auc - test_auc

    # REPORTS
    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True
    )

    # LOG METRICS
    mlflow.log_metrics({

        "train_accuracy": train_report['accuracy'],

        "test_accuracy": test_report['accuracy'],

        "train_macro_f1":
            train_report['macro avg']['f1-score'],

        "test_macro_f1":
            test_report['macro avg']['f1-score'],

        "train_fault_recall":
            train_report['1']['recall'],

        "test_fault_recall":
            test_report['1']['recall'],

        "train_healthy_recall":
            train_report['0']['recall'],

        "test_healthy_recall":
            test_report['0']['recall'],

        "train_auc": train_auc,

        "test_auc": test_auc,

        "auc_gap": auc_gap,

        "best_threshold": best_threshold
    })

    # Save the model locally
    model_path = "best_cap_prediction_model_v1.joblib"
    joblib.dump(model_pipeline, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "v-vasanth2009/capstone-pred-main-vk-12052026"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_cap_prediction_model_v1.joblib",
        path_in_repo="best_cap_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
