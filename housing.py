import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\LearnPy\mlAlgo\Housing.csv")
print(df.head())
print(df.info())

x=df[["area"]]
y=df["price"]

x_test, x_train, y_test, y_train = train_test_split(x,y,textsize=0.3, random_state=42)

model= LinearRegression()
model.fit(x,y)

print("Slope:",model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(x_test)

#best Fit Line

plt.scatter(x,y)
plt.plot(x,y_pred, color="red", label = "Predicted Line")
plt.xlabel("Area of the House")
plt.ylabel("Price of the House")
plt.title("Best Fit Line")
plt.show()

MSE = mean_squared_error(y,y_pred)
RMSE = np.sqrt(MSE)
r2 = r2_score(y,y_pred)

print("MSE:",MSE)
print("RMSE:",RMSE)
print("R2 Score:", r2)

while True:
    try:
        area = float(input("Enter your Area:"))
        break
    except ValueError:
        print("Enter valid number.")

predicted_price = model.predict([[area]])
print("Predicted Price:", predicted_price)




