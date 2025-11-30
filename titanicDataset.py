import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#visualization
#survival count
# sns.countplot(data=df, x= 'Survived')
# plt.show()

#survival by gender
# sns.countplot(data=df, x='Survived', hue ='Sex')
# plt.show()

#age Distribution
sns.histplot(df['Age'].dropna(), bins=30)
plt.show()

#SurvivalByClass
sns.countplot(data = df, x='Pclass', hue = 'Survived')
plt.show()