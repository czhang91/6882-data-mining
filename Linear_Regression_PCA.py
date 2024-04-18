import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1 data preparation
data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

X = data.iloc[:-1]
pca = PCA(n_components=18)
principal_components = pca.fit_transform(X)
X = pd.DataFrame(data=principal_components)
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
Y = data[:]['41108'].iloc[1:]
X_train = X.iloc[:-36]  # 36 is the size of test sample#
X_test = X.iloc[-36:]
Y_train = Y.iloc[:-36]
Y_test = Y.iloc[-36:]

# Step 2 data definition
model = LinearRegression()
model.fit(X_train, Y_train)

# Step 3 Predict using the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
for item in y_pred:
    print(item)
print("------------")
# Step 4 Evaluate the model
for i, item in enumerate(Y_test):
    print(100 * (1 - abs(item - y_pred[i]) / item))
accuracy = 100 * (1 - abs((Y_test - y_pred) / Y_test))
print(accuracy.mean())
