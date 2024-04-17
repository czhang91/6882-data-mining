import pandas as pd
from sklearn.linear_model import Lasso

# Step 1 data preparation
data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

X = data.iloc[:-1]
Y = data[:]['549295'].iloc[1:]

X_train = X.iloc[:-36]  # 36 is the size of test sample#
X_test = X.iloc[-36:]
Y_train = Y.iloc[:-36]
Y_test = Y.iloc[-36:]

# Step 2 definition of model
lasso = Lasso(alpha=1.0)

# Step 3 Fit the model to our training data
lasso.fit(X_train, Y_train)

print("Coefficients:", lasso.coef_)

# Step 4 Predict using the test set
Y_pred = lasso.predict(X_test)

# Step 5 Evaluate the model
for item in Y_pred:
    print(item)
print("________")
for i, item in enumerate(Y_test):
    print(100 * (1 - abs(item - Y_pred[i]) / item))

accuracy = 100 * (1 - abs((Y_test - Y_pred) / Y_test))
print(accuracy.mean())
