import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# Step 1 data preparation
data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

X = data.iloc[:-1]
Y = data[:]['541982'].iloc[1:]
X_train = X.iloc[:-36]  # 36 is the size of test sample#
X_test = X.iloc[-36:]
Y_train = Y.iloc[:-36]
Y_test = Y.iloc[-36:]

# Step 2 data definition
random_forest_model = RandomForestRegressor(random_state=1)

# Step 4 Predict using the test set
random_forest_model.fit(X_train, Y_train)
# Feature selection using feature importance from the random forest
selector = SelectFromModel(random_forest_model, threshold="median", prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

rf_selected = RandomForestRegressor(random_state=1)

# Fit the new model to the selected features from the training data
rf_selected.fit(X_train_selected, Y_train)

# Predict on the test data using the model with selected features
y_pred_selected = rf_selected.predict(X_test_selected)


# Step 5 Evaluate the model
accuracy = 100 * (1 - abs((Y_test - y_pred_selected) / Y_test))
print(accuracy.mean())
