import pandas as pd
import sys
from joblib import load

# Add src directory to path
sys.path.append('../src')
from ingest_data import get_clean_data
from feature_engineering import create_engagement_features, one_hot_encode_categorical

def prepare_data_for_bi_dashboard():
    """
    Loads the trained model, makes predictions, and saves the final,
    engineered DataFrame with churn predictions to a new CSV file.
    """
    print("Loading and preparing data for BI dashboard...")
    
    # Load the trained model
    try:
        model = load('../models/XGBClassifier.joblib')
    except FileNotFoundError:
        print("Error: XGBClassifier.joblib not found. Please run src/train_model.py first.")
        return

    # Prepare the data in the exact same way as in the training script
    data_path = '../data/CreditCardCustomers.csv'
    df_clean = get_clean_data(data_path)
    df_engineered = create_engagement_features(df_clean)
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df_final = one_hot_encode_categorical(df_engineered, columns=categorical_cols)

    # Get features for prediction
    # Drop the target variable and customer ID before making predictions
    X = df_final.drop(columns=['ChurnStatus', 'CustomerID'])
    
    # Predict churn probability for each customer
    print("Generating churn predictions...")
    df_final['Churn_Prediction_Probability'] = model.predict_proba(X)[:, 1]
    
    # Save the final DataFrame to a new CSV file for BI tools
    output_path = '../data/final_churn_data.csv'
    df_final.to_csv(output_path, index=False)
    
    print(f"✅ Final dataset for BI tools saved to '{output_path}'")

if __name__ == "__main__":
    prepare_data_for_bi_dashboard()