import pandas as pd


from sklearn.tree import DecisionTreeRegressor


# step 1 data import
data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

# step 2 data preparation
X = data.iloc[:-1]
Y = data[:]['549295'].iloc[1:]
X_train = X.iloc[:-36]
X_test = X.iloc[-36:]
Y_train = Y.iloc[:-36]
Y_test = Y.iloc[-36:]

# step 3 define model and parameter
decision_tree_regressor = DecisionTreeRegressor(max_depth=5)

#step 4 Fit the model to the training data
decision_tree_regressor.fit(X_train, Y_train)
importances = decision_tree_regressor.feature_importances_

#step 5 Predict data
Y_pred = decision_tree_regressor.predict(X_test)

#step 6 Select features that have an importance greater than a threshold (e.g., median)
important_indices = [index for index, importance in enumerate(importances) if importance > 0]
selected_features = X_train.columns[important_indices]
X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]
#step 7 Re-train the model with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

tree_model_selected = DecisionTreeRegressor(random_state=42)
tree_model_selected.fit(X_train_selected, Y_train)

# step 8: Re-evaluate the model
Y_pred = tree_model_selected.predict(X_test_selected)

#step 8 performance evaluation
for item in Y_pred:
    print(item)
print("________")
for i, item in enumerate(Y_test):
    print(100 * (1 - abs(item - Y_pred[i]) / item))

accuracy = 100 * (1 - abs((Y_test - Y_pred) / Y_test))
print(accuracy.mean())
