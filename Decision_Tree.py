import pandas as pd


from sklearn.tree import DecisionTreeRegressor


# step 1 data import
data = pd.read_csv("Sub_Oil_VLCC_Monthly.csv")
data = data.drop(data.columns[0], axis=1)

# step 2 data preparation
X = data.iloc[:-1]
Y = data[:]['67321'].iloc[1:]
X_train = X.iloc[:-36]  # 36 is the size of test sample#
X_test = X.iloc[-36:]
Y_train = Y.iloc[:-36]
Y_test = Y.iloc[-36:]

# step 3 define model and parameter
decision_tree_regressor = DecisionTreeRegressor(max_depth=5)

#step 4 Fit the model to the training data
decision_tree_regressor.fit(X_train, Y_train)


#step 5 predict data
Y_pred = decision_tree_regressor.predict(X_test)

#step 6 performance evaluation
for item in Y_pred:
    print(item)
print("________")
for i, item in enumerate(Y_test):
    print(100 * (1 - abs(item - Y_pred[i]) / item))

accuracy = 100 * (1 - abs((Y_test - Y_pred) / Y_test))
print(accuracy.mean())
