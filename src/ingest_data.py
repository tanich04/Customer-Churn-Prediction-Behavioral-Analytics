import pandas as pd
import numpy as np

def get_raw_data(file_path):
    """
    Loads the raw credit card customer data from a CSV file.
    In a production environment, this would be the result of a complex SQL query.
    """
    df = pd.read_csv(file_path)
    return df

def prepare_data(df):
    """
    Performs initial data preparation and feature engineering.
    This simulates the 'Transform' step in an ETL pipeline.
    """
    # Create a new feature: Average Transaction Amount per Transaction Count
    # This also handles cases where Total_Trans_Ct is zero.
    df['Avg_Trans_Amt_per_Ct'] = np.where(df['Total_Trans_Ct'] > 0, df['Total_Trans_Amt'] / df['Total_Trans_Ct'], 0)
    
    # Rename columns to be more business-friendly and aligned with financial terms
    column_mapping = {
        'CLIENTNUM': 'CustomerID',
        'Attrition_Flag': 'ChurnStatus',
        'Customer_Age': 'Age',
        'Months_on_book': 'Tenure',
        'Credit_Limit': 'CreditLimit',
        'Total_Trans_Ct': 'TransactionCount',
        'Total_Trans_Amt': 'TransactionAmount',
        'Avg_Open_To_Buy': 'AvgOpenToBuy',
        'Total_Revolving_Bal': 'TotalRevolvingBalance',
        'Total_Amt_Chng_Q4_Q1': 'TransactionAmountChange_Q4_Q1',
        'Total_Ct_Chng_Q4_Q1': 'TransactionCountChange_Q4_Q1',
        'Avg_Utilization_Ratio': 'CreditUtilizationRatio'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # --- CRITICAL STEP: REMOVE DATA LEAKAGE FEATURES ---
    # The dataset documentation explicitly states to ignore these columns.
    # They are the source of data leakage and will lead to an unrealistically perfect model.
    leakage_columns = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    df.drop(columns=leakage_columns, inplace=True, errors='ignore')
            
    # Convert `ChurnStatus` to a binary numeric format (0 for 'Existing Customer', 1 for 'Attrited Customer')
    df['ChurnStatus'] = df['ChurnStatus'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
    
    return df

def get_clean_data(file_path):
    """
    A single function to run the entire data ingestion and preparation process.
    """
    raw_df = get_raw_data(file_path)
    clean_df = prepare_data(raw_df)
    return clean_df