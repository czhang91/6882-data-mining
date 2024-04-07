import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

X = data.drop('67321', axis=1).iloc[:-1]
Y = data[:]['67321'].iloc[1:]
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
# for item in y_pred:
#     print(item)
# print("------------")
# for i, item in enumerate(Y_test):
#     print(100 * (1 - abs(item - y_pred[i]) / item))
# accuracy = 100 * (1 - abs((Y_test - y_pred) / Y_test))
# print(accuracy.mean())

l = plt.scatter(Y_test, y_pred, color='red', marker='o')
plt.title('Prediction results of Polynomial Regression')
plt.legend(handles=[l], labels=["Predict values"])
line_start = min(Y_test.min(), y_pred.min())
line_end = max(Y_test.max(), y_pred.max())

plt.plot([line_start, line_end], [line_start, line_end])
plt.show()
