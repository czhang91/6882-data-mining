import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib
from sklearn.preprocessing import MinMaxScaler

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

X = data.iloc[:-1]

# Step 1 data preparation
Y = data[:]['549295'].iloc[1:]  # target ID
X_train = X.iloc[:-36]  # 36 is the size of test sample#
X_test = X.iloc[-36:]
Y_train = Y.iloc[:-36]
Y_test = Y.iloc[-36:]

# Step 2 definition of model
gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                        random_state=42)

# Step 3 Fit the model to our training data
gradient_boosting_regressor.fit(X_train, Y_train)

print("Feature Importance:", gradient_boosting_regressor.feature_importances_)

# Step 4 Predict using the test set
Y_pred = gradient_boosting_regressor.predict(X_test)

# Step 5 Evaluate the model
for item in Y_pred:
    print(item)
print("________")
for i, item in enumerate(Y_test):
    print(100 * (1 - abs(item - Y_pred[i]) / item))

accuracy = 100 * (1 - abs((Y_test - Y_pred) / Y_test))
print(accuracy.mean())
# step 6 Visualize ....
