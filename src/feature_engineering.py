import pandas as pd
import numpy as np

def create_engagement_features(df):
    """
    Creates new features related to customer engagement and activity.
    """
    # Feature 1: Ratio of inactive months to total tenure
    # Handling potential division by zero
    df['Inactive_to_Tenure_Ratio'] = np.where(df['Tenure'] > 0, df['Months_Inactive_12_mon'] / df['Tenure'], 0)
    
    # Feature 2: Ratio of total change in transaction amount to total transaction amount
    df['Trans_Amt_Change_Ratio'] = np.where(df['TransactionAmount'] > 0, df['TransactionAmountChange_Q4_Q1'] / df['TransactionAmount'], 0)
    
    # Feature 3: Ratio of total change in transaction count to total transaction count
    df['Trans_Ct_Change_Ratio'] = np.where(df['TransactionCount'] > 0, df['TransactionCountChange_Q4_Q1'] / df['TransactionCount'], 0)

    return df

def one_hot_encode_categorical(df, columns):
    """
    One-hot encodes specified categorical columns.
    """
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df