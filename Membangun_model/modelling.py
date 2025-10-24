import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Memuat data dari path CSV."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File data tidak ditemukan di {file_path}", file=sys.stderr)
        sys.exit(1)

def print_metrics(model_name, y_test, y_pred):
    """Mencetak metrik evaluasi ke konsol."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nMetrik untuk {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

def main():
    DATA_PATH = "../Membangun_model/PCOS_preprocessing.csv"
    
    # 1. Muat Data
    print("Memuat data...")
    data = load_data(DATA_PATH)
    
    # 2. Pisahkan Fitur (X) dan Target (y)
    try:
        X = data.drop('PCOS (Y/N)', axis=1)
        y = data['PCOS (Y/N)']
    except KeyError as e:
        print(f"Error: Kolom target 'PCOS (Y/N)' tidak ditemukan. {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Aktifkan MLflow Autologging 
    print("Mengaktifkan MLflow autolog...")
    mlflow.sklearn.autolog()

    # --- Eksperimen 1: Logistic Regression ---
    print("\nMemulai Run: Logistic Regression...")
    with mlflow.start_run(run_name="Logistic Regression") as run_lr:
        model_lr = LogisticRegression(random_state=42, max_iter=1000)
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        
        print_metrics("Logistic Regression", y_test, y_pred_lr)
        print(f"Run ID: {run_lr.info.run_id}")

    # --- Eksperimen 2: Random Forest ---
    print("\nMemulai Run: Random Forest...")
    with mlflow.start_run(run_name="Random Forest") as run_rf:
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        
        print_metrics("Random Forest", y_test, y_pred_rf)
        print(f"Run ID: {run_rf.info.run_id}")

    # --- Eksperimen 3: SVM ---
    print("\nMemulai Run: SVM (Support Vector Machine)...")
    with mlflow.start_run(run_name="SVM") as run_svm:
        model_svm = SVC(random_state=42)
        model_svm.fit(X_train, y_train)
        y_pred_svm = model_svm.predict(X_test)
        
        print_metrics("SVM", y_test, y_pred_svm)
        print(f"Run ID: {run_svm.info.run_id}")

    print("\nSemua eksperimen selesai.")

if __name__ == "__main__":
    main()