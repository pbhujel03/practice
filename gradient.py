import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([500,750,1000,1200,1500,1800,2000,2500,3000], dtype=float)
y = np.array([25,35,50,58,72,85,95,118,140], dtype=float)


x_scaled = (x- x.mean())/x.std()

#initialize
m=0
b=0
learning_rate = 0.01
epochs = 1000
n = len(x)

#training loop
for i in range(epochs):
    y_pred = m*x_scaled+b

    gradient_m = (-2/n)* np.sum(x_scaled*(y-y_pred))
    gradient_b = (-2/n)* np.sum(y-y_pred)

    #with learning rate 0.0001
    m = m - learning_rate * (gradient_m)
    b = b - learning_rate * (gradient_b)

print ("Final Slope m:",m)
print("Final intercept b:", b)

#sklearn
x_2d = x_scaled.reshape(-1,1)
sk_model = LinearRegression()
sk_model.fit(x_2d, y)

print("sklearn slope M:", sk_model.coef_[0])
print("SkLearn intercept b:", sk_model.intercept_)\

#plot
plt.scatter(x, y, label = "Actual")
plt.plot(x, m*x_scaled+b, color="red", label = "Gradient Descent Line")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Gradient Descent")
plt.legend()
plt.tight_layout()
plt.axis('auto')
plt.show()

