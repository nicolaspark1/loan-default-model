import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "task_3_4.csv"  
data = pd.read_csv(file_path)

# Ensure columns exist and clean the data
if 'fico_score' not in data.columns or 'default' not in data.columns:
    raise KeyError("The dataset must contain 'fico_score' and 'default' columns.")

# Create bins for FICO scores
bins = list(range(300, 851, 50))  # Binning scores from 300 to 850 with a width of 50
data['FICO_Bucket'] = pd.cut(data['fico_score'], bins=bins)

# Calculate default rates by bucket
default_rates = data.groupby('FICO_Bucket', observed=False)['default'].mean()

# Ensure default rates are valid
if default_rates.empty:
    raise ValueError("Default rates calculation failed. Check your data or bucket definitions.")

# Plot the default rates
plt.figure(figsize=(10, 6))
default_rates.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Default Rates by FICO Score Bucket")
plt.xlabel("FICO Score Range")
plt.ylabel("Default Rate")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot and display it
plt.savefig("fico_default_rates_plot.png")
print("Plot saved as fico_default_rates_plot.png")
plt.show()
