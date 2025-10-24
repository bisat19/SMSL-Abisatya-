import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sys
import warnings
import joblib
import os
import dagshub

dagshub.init(repo_owner='bisat19',
             repo_name='Membangun_model',
             mlflow=True)

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Memuat data dari path CSV."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File data tidak ditemukan di {file_path}", file=sys.stderr)
        print("Pastikan file 'PCOS_preprocessing.csv' ada di dalam folder 'namadataset_preprocessing/'", file=sys.stderr)
        sys.exit(1)

def eval_metrics(y_test, y_pred, y_pred_proba):
    """Menghitung metrik (termasuk advance) dan mengembalikannya sebagai dictionary."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    ll = log_loss(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "log_loss": ll, "roc_auc": auc}

def main():
    DATA_PATH = "../Membangun_model/PCOS_preprocessing.csv"
    
    print("Memuat data...")
    data = load_data(DATA_PATH)
    
    try:
        X = data.drop('PCOS (Y/N)', axis=1)
        y = data['PCOS (Y/N)']
    except KeyError as e:
        print(f"Error: Kolom target 'PCOS (Y/N)' tidak ditemukan. {e}", file=sys.stderr)
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Definisikan Parameter untuk Tuning ---
    # Parameter untuk Random Forest
    rf_params_list = [
        {"n_estimators": 100, "max_depth": None, "random_state": 42},
        {"n_estimators": 150, "max_depth": None, "random_state": 42},
        {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        {"n_estimators": 150, "max_depth": 10, "random_state": 42},
        {"n_estimators": 200, "max_depth": 10, "random_state": 42},
        {"n_estimators": 100, "max_depth": 20, "random_state": 42},
        {"n_estimators": 150, "max_depth": 20, "random_state": 42}
    ]



    # Parameter untuk Logistic Regression
    lr_params_list = [
        {"C": 0.1, "max_iter": 1000, "random_state": 42},
        {"C": 1.0, "max_iter": 1000, "random_state": 42},
        {"C": 10, "max_iter": 1000, "random_state": 42}
    ]

    # Parameter untuk SVM
    svm_params_list = [
        {"C": 0.1, "kernel": "linear", "random_state": 42, "probability": True},
        {"C": 1.0, "kernel": "rbf", "random_state": 42, "probability": True},
        {"C": 10, "kernel": "rbf", "random_state": 42, "probability": True}

    ]

    with mlflow.start_run(run_name="All Model Hyperparameter Tuning") as parent_run:
        print(f"Memulai Parent Run: {parent_run.info.run_id}")

        # --- 1. Tuning Random Forest ---
        print("\n--- Memulai Tuning Random Forest ---")
        for i, params in enumerate(rf_params_list):
            run_name = f"RF_Run_{i+1}"
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                print(f"Memulai {run_name} dengan params: {params}")
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                metrics = eval_metrics(y_test, y_pred, y_pred_proba)
                
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                model_filename = f"{run_name}.pkl"
                joblib.dump(model, model_filename)
                mlflow.log_artifact(model_filename, artifact_path="model_files")
                os.remove(model_filename)
                
                print(f"Metrik untuk {run_name}: {metrics}")

        # --- 2. Tuning Logistic Regression ---
        print("\n--- Memulai Tuning Logistic Regression ---")
        for i, params in enumerate(lr_params_list):
            run_name = f"LR_Run_{i+1}"
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                print(f"Memulai {run_name} dengan params: {params}")
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                metrics = eval_metrics(y_test, y_pred, y_pred_proba)
                
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                model_filename = f"{run_name}.pkl"
                joblib.dump(model, model_filename)
                mlflow.log_artifact(model_filename, artifact_path="model_files")
                os.remove(model_filename)
                
                print(f"Metrik untuk {run_name}: {metrics}")

        # --- 3. Tuning SVM ---
        print("\n--- Memulai Tuning SVM ---")
        for i, params in enumerate(svm_params_list):
            run_name = f"SVM_Run_{i+1}"
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                print(f"Memulai {run_name} dengan params: {params}")
                model = SVC(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                metrics = eval_metrics(y_test, y_pred, y_pred_proba)
                
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                model_filename = f"{run_name}.pkl"
                joblib.dump(model, model_filename)
                mlflow.log_artifact(model_filename, artifact_path="model_files")
                os.remove(model_filename)
                
                print(f"Metrik untuk {run_name}: {metrics}")

    print(f"\nSemua eksperimen tuning selesai. Parent Run ID: {parent_run.info.run_id}")

if __name__ == "__main__":
    main()