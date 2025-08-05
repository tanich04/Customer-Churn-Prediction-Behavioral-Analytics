import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
from joblib import dump
import sys

# Add the src directory to the system path to import your functions
sys.path.append('../src')
from ingest_data import get_clean_data
from feature_engineering import create_engagement_features, one_hot_encode_categorical

def train_and_evaluate_models():
    """
    Trains and evaluates multiple models, handling class imbalance.
    """
    # Load and prepare data
    file_path = '../data/CreditCardCustomers.csv'
    df_clean = get_clean_data(file_path)
    df_engineered = create_engagement_features(df_clean)
    
    # Define the categorical columns to one-hot encode
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df_final = one_hot_encode_categorical(df_engineered, columns=categorical_cols)

    # Separate features (X) and target (y)
    X = df_final.drop(columns=['ChurnStatus', 'CustomerID'])
    y = df_final['ChurnStatus']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Handle Class Imbalance with SMOTE ---
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Resampled data shape:", X_train_resampled.shape)

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    best_model = None
    best_f1_score = 0.0

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        
        # --- Evaluate with Business-Relevant Metrics ---
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Model: {name}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = model

    # Save the best model
    model_filename = f"../models/{best_model.__class__.__name__}.joblib"
    dump(best_model, model_filename)
    print(f"\n✅ Best model ({best_model.__class__.__name__}) saved to {model_filename}")

if __name__ == "__main__":
    train_and_evaluate_models()