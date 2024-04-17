import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler



# Step 1 data preparation
data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

X = data.iloc[:-1]
Y = data[:]['549295'].iloc[1:]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train = X[:-36]
X_test = X[-36:]
Y_train = Y[:-36]
Y_test = Y[-36:]

# Step 2 define model
poly_features = PolynomialFeatures(degree=2)  # Example with a polynomial of degree 2
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
# Step 3 Fit the model to our training data
model.fit(X_train_poly, Y_train)

# Step 4 Predict using the test set
y_pred = model.predict(X_test_poly)

# Step 5 Evaluate the model
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

for item in y_pred:
    print(item)
print("------------")
for i, item in enumerate(Y_test):
    print(100 * (1 - abs(item - y_pred[i]) / item))
accuracy = 100 * (1 - abs((Y_test - y_pred) / Y_test))
print(accuracy.mean())
