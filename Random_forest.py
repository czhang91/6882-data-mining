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
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, Y_train)
Y_pred = linear_regression_model.predict(X_test)
accuracy = 100 * (1 - abs((Y_test - Y_pred) / Y_test))
print(accuracy.mean())

polynomial_features= PolynomialFeatures(degree=4)
X_poly = polynomial_features.fit_transform(X_train)
X_pr_train, X_pr_test, Y_pr_train, Y_pr_test = train_test_split(X_poly, Y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_pr_train, Y_pr_train)
Y_pred_pr = model.predict(X_test)
print(Y_pred_pr)
for item in Y_pred_pr:
    print(item)
accuracy = 100 * (1 - abs((Y_test - Y_pred_pr) / Y_test))
print(accuracy.mean())


random_forest_model = RandomForestRegressor(random_state=1)
random_forest_model.fit(X_train, Y_train)
Y_pred_random_forest = random_forest_model.predict(X_test)

accuracy = 100 * (1 - abs((Y_test - Y_pred_random_forest) / Y_test))
print(accuracy.mean())

# mse = mean_squared_error(Y_test, Y_pred)
# print("Mean Squared Error (MSE) for linear regression is:", mse)

# mse = mean_squared_error(Y_test, Y_pred_poly)
# print("Mean Squared Error (MSE) for polynomial regression is:", mse)

mse = mean_squared_error(Y_test, Y_pred_random_forest)

print("Mean Squared Error (MSE) for random forests is:", mse)
plt.figure(figsize=(10, 6))
# l = plt.scatter(Y_test, Y_pred, color='red', marker='o')
# plt.title('Prediction results of Linear Regression')
# plt.legend(handles=[l], labels=["Predict values"])
# line_start = min(Y_test.min(), Y_pred.min())
# line_end = max(Y_test.max(), Y_pred.max())
#
# plt.plot([line_start, line_end], [line_start, line_end])
# l = plt.scatter(Y_test, Y_pred_poly, color='red', marker='o')
# plt.title('Prediction results of Random Forest')
# plt.legend(handles=[l], labels=["Predict values"])
# line_start = min(Y_test.min(), Y_pred_poly.min())
# line_end = max(Y_test.max(), Y_pred_poly.max())
#
# plt.plot([line_start, line_end], [line_start, line_end])
# plt.show()
