import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Step 1 data preparation
data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

X = data.drop('541982', axis=1).iloc[:-1]
Y = data[:]['541982'].iloc[1:]
X_train = X.iloc[:-36]  # 36 is the size of test sample#
X_test = X.iloc[-36:]
Y_train = Y.iloc[:-36]
Y_test = Y.iloc[-36:]

# Step 2 data definition
random_forest_model = RandomForestRegressor(random_state=1)

# Step 4 Predict using the test set
random_forest_model.fit(X_train, Y_train)
Y_pred_random_forest = random_forest_model.predict(X_test)

# Step 5 Evaluate the model
accuracy = 100 * (1 - abs((Y_test - Y_pred_random_forest) / Y_test))
print(accuracy.mean())
