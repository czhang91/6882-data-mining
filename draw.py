import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = './Result_detail_Decision_Tree_Regression_feature selection.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime for better plotting
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')

# Set up the figure for subplots with 4 plots
fig, axs = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

# List of column indices for target and predict values
value_pairs = [
    ('Target Value(542236)', 'Predict Value(542236)'),
    ('Target Value(67321)', 'Predict Value(67321)'),
    ('Target Value(549295)', 'Predict Value(549295)'),
    ('Target Value(41108)', 'Predict Value(41108)'),
    ('Target Value(541982)', 'Predict Value(541982)')
]

# Titles for each subplot
titles = [
    'Target 542236',
    'Target 67321',
    'Target 549295',
    'Target 41108',
    'Target 541982'
]

# Plot each pair of actual and predicted values
for i, (target_col, predict_col) in enumerate(value_pairs):
    axs[i].plot(data['Date'], data[target_col], label='Actual Values', color='blue', marker='o')
    axs[i].plot(data['Date'], data[predict_col], label='Predicted Values', color='red', marker='o')
    axs[i].set_title(f'Actual vs. Predicted Values for {titles[i]}')  # Set title for each subplot
    axs[i].legend()  # Add legend to each subplot
    axs[i].grid(True)  # Add grid for better readability

# Set common labels
plt.xlabel('Time')  # Label for the x-axis
fig.text(0.04, 0.5, 'Values', va='center', rotation='vertical')  # Vertical label for the y-axis

# Rotate date labels for better visibility
plt.xticks(rotation=45)

# Adjust layout to prevent overlap and ensure everything fits well
plt.tight_layout()

# Save the figure to a file
plt.savefig('output_figure.png')

# Optionally display the plot if running in an interactive environment (like Jupyter)
plt.show()
