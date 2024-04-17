import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

from sklearn.preprocessing import MinMaxScaler

# Step 1 data preparation
data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)
# Set lag=1
lag = 2
X = data
Y = data[:]['549295']

lagged_X = X.shift(lag-1)
X = X.shift(lag)
Y = Y.shift(-lag)

X = pd.concat([X, lagged_X], axis=1)
X.dropna(inplace=True)
Y.dropna(inplace=True)

# Normalization
scaler = MinMaxScaler()
print(X)
X = scaler.fit_transform(X)

# PCA after normalization
pca = PCA(n_components=7)
principal_components = pca.fit_transform(X)
X = pd.DataFrame(data=principal_components)

X_train = X[:-36]
X_test = X[-36:]
Y_train = Y[:-36]
Y_test = Y[-36:]

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
