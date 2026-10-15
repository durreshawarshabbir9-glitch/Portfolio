from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train):
    print("Applying SMOTE balancing to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("SMOTE balancing completed successfully!")
    return X_train_resampled, y_train_resampled