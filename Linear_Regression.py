import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data_path = 'Sub_Oil_VLCC_Monthly.csv'
df = pd.read_csv(data_path)

# Display the first few rows of the dataset to understand its structure
# print(df.head())


# Identify the target column index
target_column_id = "67321"


# Prepare X by excluding the first (date) and target columns
X = df.drop(columns = [df.columns[0], target_column_id])

# Prepare Y by selecting the target column
Y = df[target_column_id]

# Adjusting X and Y to ensure proper alignment
X = X.iloc[:-1] # Drop the last row from X
Y = Y.iloc[1:] # Drop the first row from Y

# Verify alignment by showing the last row of X and the first row of Y
X_tail = X.tail(1)
Y_head = Y.head(1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

# Convert X and Y to numpy arrays for easier manipulation
X_np = X.to_numpy()
Y_np = Y.to_numpy()

# Splitting the data into training and testing sets based on the last 36 months
X_train, X_test = X_np[:-36], X_np[-36:]
Y_train, Y_test = Y_np[:-36], Y_np[-36:]

# Initialize and train the Linear Regression model
md = LinearRegression()
md.fit(X_train, Y_train)

# Predict on the testing set
Y_pred = md.predict(X_test)

# Calculate accuracy for each prediction
accuracies = 100 *(1 - np.abs((Y_test - Y_pred) / Y_test))

# Calculate the average accuracy over the test set
average_accuracy = np.mean(accuracies)

print("average_accuracy: ", average_accuracy)


# output the accuracy
from datetime import datetime

# Prepare the data for the Excel file
dates_test = df.iloc[-36:, 0].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%Y-%m'))  # Format dates
true_values = Y_test  # True target values
predicted_values = Y_pred  # Predicted values

# Create a DataFrame for the Excel file
results_df = pd.DataFrame({
    'Date': dates_test,
    'True Target Values': true_values,
    'Predicted Values': predicted_values,
    'Accuracy': accuracies
})

# File path to save the Excel file
results_file_path = 'maritime_shipping_forecast_results.xlsx'

# Save the DataFrame to an Excel file
results_df.to_excel(results_file_path, index=False)

results_file_path
