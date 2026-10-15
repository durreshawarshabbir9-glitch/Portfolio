#Loading Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
 #load dataset
df=pd.read_csv("creditcard.csv")
print ("Dataset Loaded Successfully")
print("___________________________________________")
df.shape
print(df.shape)
print("___________________________________________")
df.info()
print(df.info())
print("_____________________________________________")
df.head(10)
print("Intial Rows foR Data Inspection")
print(df.head(10))
df.describe
print("Statistical Summary of Dataset")
print(df.describe())
print("_______________________________________________")
df.isnull().sum()
print("----Checking for Null Values In Dataset----")
print(df.isnull().sum())
# Check class distribution
fraud_counts = df['Class'].value_counts()
fraud_percentage = df['Class'].value_counts(normalize=True) * 100

print("\n--- Fraud Class Distribution ---")
print(fraud_counts)
print("\n--- Percentage Split ---")
print(fraud_percentage)