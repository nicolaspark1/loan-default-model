import pandas as pd
import matplotlib.pyplot as plt
from fico_bucket_optimizer import optimize_buckets

# Load and clean the dataset
file_path = "task_3_4.csv"  
data = pd.read_csv(file_path)

if 'fico_score' not in data.columns or 'default' not in data.columns:
    raise KeyError("The dataset must contain 'fico_score' and 'default' columns.")

# Filter valid FICO scores
data = data[(data['fico_score'] >= 300) & (data['fico_score'] <= 850)]
data = data.sort_values(by='fico_score')

# Step 1: Plot default rates for initial bins
bins = list(range(300, 851, 50))  # Define bins
data['FICO_Bucket'] = pd.cut(data['fico_score'], bins=bins)

default_rates = data.groupby('FICO_Bucket', observed=False)['default'].mean()

if default_rates.empty:
    raise ValueError("Default rates calculation failed. Check your data or bucket definitions.")

plt.figure(figsize=(10, 6))
default_rates.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Default Rates by FICO Score Bucket")
plt.xlabel("FICO Score Range")
plt.ylabel("Default Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fico_default_rates_plot.png")
print("Plot saved as fico_default_rates_plot.png")
plt.show()

# Step 2: Optimize buckets
optimal_buckets = optimize_buckets(data, num_buckets=5)

print("\nOptimal Buckets:")
for b in optimal_buckets:
    print(f"FICO Range: {b[0]} - {b[1]}")
