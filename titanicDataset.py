import pandas as pd

df=pd.read_csv(r'C:\LearnPy\practice\train.csv')

print(df.head())
print(df.info())
print(df.describe())

#check missing values
print(df.isnull().sum())

#basic statistics
print(df["Age"].mean())
print(df["Fare"].max())

#check unique values
print(df["Sex"].unique())
print(df["Embarked"].unique())