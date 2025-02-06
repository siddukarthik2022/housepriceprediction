import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pickle

#Step1 : Data Collection

data = pd.DataFrame({
    "square_feet": [2000, 3500, 4000, 5000, 6000],
    "rooms": [2, 3, 4, 5, 6],
    "price": [300000, 400000, 500000, 600000, 700000]
})

df = pd.DataFrame(data)
print(df.head())

# Step 2 : Data PreProcessing
# convert the data into input(x) and output(y)

X = data[["square_feet", "rooms"]]
y = data["price"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)  # ✅ This should work

# Step 4: Make a prediction

y_pred= model.predict(X_test)

print("predicted prices",y_pred)
print("actual prices",y_test.values)

# step 5: Evaluate the model

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
smse=np.sqrt(mse)

print(f"Mean absolute Error:{mae}")
print(f"Root mean squared Error: {smse}")

# save the model

with open("house_price_model.pkl","wb") as f:
    pickle.dump(model,f)

# create Flask api

from flask import Flask, request, jsonify

app = Flask(__name__)  # ✅ This must be defined

@app.route("/")
def home():
    return "Flask is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    square_feet = data.get("square_feet", 0)
    bedrooms = data.get("bedrooms", 0)

    # Dummy prediction logic (Replace with actual ML model)
    predicted_price = (square_feet * 200) + (bedrooms * 10000)

    return jsonify({"predicted_price": predicted_price})

if __name__ == "__main__":
    app.run(debug=True)
