import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_credit_data(filepath="creditcard.csv"):
    print("Loading and scaling dataset...")
    df = pd.read_csv(filepath)
    
    # Scale 'Amount'
    scaler = StandardScaler()
    df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Drop raw unscaled columns
    df = df.drop(['Time', 'Amount'], axis=1)
    
    # Separate Features and Target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Train-test split completed successfully!")
    return X_train, X_test, y_train, y_test