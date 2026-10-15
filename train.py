from spiltter import split_credit_data  
from balancer import apply_smote

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Split the data
X_train, X_test, y_train, y_test = split_credit_data()

# 2. Balance ONLY the training data using SMOTE
X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

# 3. Define models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
}

# 4. Train and evaluate each model in a loop
for name, model in models.items():
    print(f"\n" + "="*40)
    print(f"Training {name}...")
    print("="*40)
    
    # Train the model
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Print results
    print(f"\n--- Confusion Matrix ({name}) ---")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\n--- Classification Report ({name}) ---")
    print(classification_report(y_test, y_pred))