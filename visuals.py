import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from spiltter import split_credit_data
from balancer import apply_smote
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def generate_visualizations():
    print("Generating project visualizations...")
    
    # 1. Create 'output' directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load and split data
    X_train, X_test, y_train, y_test = split_credit_data()
    
    # 3. Plot Class Imbalance (Before vs After SMOTE)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train, palette="Blues")
    plt.title("Original Class Distribution (Train Set)")
    plt.xlabel("Class (0: Legitimate, 1: Fraud)")
    plt.ylabel("Count")
    
    # Apply SMOTE to show balanced counts
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_train_bal, palette="Purples")
    plt.title("Balanced Class Distribution (After SMOTE)")
    plt.xlabel("Class (0: Legitimate, 1: Fraud)")
    plt.ylabel("Count")
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "class_distribution_comparison.png")
    plt.savefig(dist_path)
    print(f"Saved: {dist_path}")
    plt.close()
    
    # 4. Train Random Forest & Plot Confusion Matrix Heatmap
    print("\nTraining model for confusion matrix evaluation...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Legit", "Predicted Fraud"],
                yticklabels=["Actual Legit", "Actual Fraud"])
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_rf.png")
    plt.savefig(cm_path)
    print(f"Saved: {cm_path}")
    plt.close()
    
    print("\nAll visualizations generated and saved to the 'output' folder successfully!")

if __name__ == "__main__":
    generate_visualizations()