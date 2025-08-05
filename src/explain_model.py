import pandas as pd
import sys
import shap
import matplotlib.pyplot as plt
from joblib import load
from sklearn.model_selection import train_test_split

# Add src directory to path to import your functions
sys.path.append('../src')
from ingest_data import get_clean_data
from feature_engineering import create_engagement_features, one_hot_encode_categorical

def explain_model_predictions():
    """
    Loads the trained model, prepares data, and generates SHAP plots.
    The plots are saved as image files instead of being displayed interactively.
    """
    print("Loading and preparing data...")
    # Load the final model
    model = load('../models/XGBClassifier.joblib')

    # Prepare the data in the exact same way as in the training script
    data_path = '../data/CreditCardCustomers.csv'
    df_clean = get_clean_data(data_path)
    df_engineered = create_engagement_features(df_clean)
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df_final = one_hot_encode_categorical(df_engineered, columns=categorical_cols)

    # Separate features and target
    X = df_final.drop(columns=['ChurnStatus', 'CustomerID'])
    y = df_final['ChurnStatus']

    # Split data to get the test set for explaining predictions
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Generate SHAP plots ---
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    print("Generating and saving SHAP summary plots...")
    # Global feature importance summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('../images/shap_feature_importance.png')
    plt.clf()  # Clear the current figure

    # Detailed summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('../images/shap_summary_plot.png')
    plt.clf() # Clear the current figure

    # Local interpretability for a single customer (e.g., the first test customer)
    print("Saving SHAP waterfall plot for a single prediction...")
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig('../images/shap_waterfall_plot.png')
    plt.clf()

    # Local interpretability for another specific customer
    print("Saving SHAP force plot for a single prediction...")
    shap.plots.force(shap_values[100], show=False)
    plt.tight_layout()
    plt.savefig('../images/shap_force_plot.png')
    plt.clf()
    
    print("✅ All SHAP plots saved to the 'images' folder.")

if __name__ == "__main__":
    # Create an images directory to save the plots
    import os
    if not os.path.exists('../images'):
        os.makedirs('../images')
        
    explain_model_predictions()