import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv(r"C:\LearnPy\mlAlgo\Housing.csv")

binary_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes':1, 'no':0})

df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)

# print(df.columns.tolist())
# print(df.head())

# Features and Target
x = df.drop(columns = ['price'])
y = df['price']

#split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)

#model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 score:", r2)
print("RMSE:", rmse)

#coefficients
# coef_df = pd.DataFrame({
#     'Feature': x.columns, 
#     'Coefficient': model.coef_
#     })
# print("\nCoefficients:")
# print(coef_df)


def get_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def get_binary(prompt):
    while True:
        val = input(prompt).strip().lower()
        if val in ['yes','no']:
            return 1 if val == 'yes' else 0
        print("Please enter yes or no.")

def get_furnishing():
    while True:
        val = input("Furnishing Status (furnished/semi-furnished/unfurnished):").strip().lower()
        if val in ['furnished','semi-furnished','unfurnished']:
            semi_furnished = 1 if val == 'semi-furnished' else 0
            unfurnished = 1 if val == 'unfurnished' else 0
            return semi_furnished, unfurnished
        print("Please enter furnished, semi- furnished, or unfurnished")


area = get_number("Area:")  
bedrooms = get_number("Bedrooms:")
bathrooms = get_number("Bathrooms:")
stories = get_number("Stories:")
mainroad = get_binary("Main road? (yes/no):")
aircon = get_binary("Air Conditioning? (Yes/No):")
parking = get_number("Parking spots:")
prefarea = get_binary("Preferred area? (yes/No):")
semi_furnished, unfurnished = get_furnishing()

input_data = pd.DataFrame([[
    area, 
    bedrooms, 
    bathrooms,
    stories,
    mainroad, 
    0,0,0,
    aircon, 
    parking, 
    prefarea, 
    semi_furnished, 
    unfurnished
]], 
columns = x.columns)

predicted_price = model.predict(input_data)
print(f"Predicted House Price : Rs. {predicted_price[0]:,.0f}")



