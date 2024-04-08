import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge

import matplotlib
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

X = data.drop('67321', axis=1).iloc[:-1]
Y = data[:]['67321'].iloc[1:]
X_train = X.iloc[:-36]  # 36 is the size of test sample#
X_test = X.iloc[-36:]
Y_train = Y.iloc[:-36]
Y_test = Y.iloc[-36:]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
ridge = Ridge(alpha=1.0)

ridge.fit(X_train, Y_train)
Y_pred = ridge.predict(X_test)
# lasso = Lasso(alpha=1.0)
#
# # Fit the model to our training data
# lasso.fit(X_train, Y_train)
#
# # Display the coefficients
# print("Coefficients:", lasso.coef_)
#
# # Predict using the test set
# Y_pred = lasso.predict(X_test)

for item in Y_pred:
    print(item)
print("________")
for i, item in enumerate(Y_test):
    print(100 * (1 - abs(item - Y_pred[i]) / item))

accuracy = 100 * (1 - abs((Y_test - Y_pred) / Y_test))
print(accuracy.mean())

