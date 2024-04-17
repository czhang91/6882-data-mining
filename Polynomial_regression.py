import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

X = data.iloc[:-1]
Y = data[:]['549295'].iloc[1:]


X_train = X.iloc[:-36]  # 36 is the size of test sample#
X_test = X.iloc[-36:]
Y_train = Y.iloc[:-36]
Y_test = Y.iloc[-36:]

poly_features = PolynomialFeatures(degree=2)  # Example with a polynomial of degree 2
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, Y_train)

y_pred = model.predict(X_test_poly)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
for item in y_pred:
    print(item)
print("------------")
for i, item in enumerate(Y_test):
    print(100 * (1 - abs(item - y_pred[i]) / item))
accuracy = 100 * (1 - abs((Y_test - y_pred) / Y_test))
print(accuracy.mean())
