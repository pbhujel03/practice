import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv(r'C:\LearnPy\mlAlgo\Salary_Data.csv')
# print(df.head())
# print(df.info())

x=df[['YearsExperience']]
y=df['Salary']


model = LinearRegression()
model.fit(x,y)

print("slope(w)", model.coef_)
print("Intercept(b):", model.intercept_)

y_pred = model.predict(x)

#Best fit line
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Best Fit Line")
plt.show()

MSE = mean_squared_error(y,y_pred)
r2 = r2_score(y,y_pred)

print("MSE:", MSE)
print("R2 Score:", r2)

while True:
    try:
        experience = float(input("Enter your experience:"))
        break
    except ValueError:
        print("Please enter a valid number!")

predicted_salary = model.predict([[experience]])
print(f"Predicted salary for {experience} years of experience:{predicted_salary[0]:.2f}")




