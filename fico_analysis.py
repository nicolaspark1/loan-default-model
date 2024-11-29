import pandas as pd
import matplotlib.pyplot as plt
from fico_bucket_optimizer import optimize_buckets
import json  # Import to save rating map dynamically

# Load the dataset
file_path = "task_3_4.csv"
data = pd.read_csv(file_path)

# Create a sample of the data for testing
sample_data = data.head(2000)
sample_data.to_csv("sample_task_3_4.csv", index=False)  # Save it as a new file for reuse

# Use the sample data instead of the full dataset for testing
data = sample_data  # Switch to using the sample data

# Ensure FICO scores are in the valid range
data = data[(data['fico_score'] >= 300) & (data['fico_score'] <= 850)]
data = data.sort_values(by='fico_score')

# Optimize buckets dynamically
num_buckets = 5  # You can adjust this dynamically as needed
optimal_buckets = optimize_buckets(data, num_buckets=num_buckets)

# Get bucket boundaries dynamically
bucket_boundaries = [b[0] for b in optimal_buckets] + [850]  # Append the upper limit of FICO scores
data['Bucket'] = pd.cut(data['fico_score'], bins=bucket_boundaries)

# Calculate default rates for each bucket
default_rates = data.groupby('Bucket', observed=False)['default'].mean()

# Sort buckets by default rates and assign ratings dynamically
default_rates = default_rates.sort_values(ascending=True)  # Lower PD gets lower ratings
ratings = {str(bucket): i+1 for i, bucket in enumerate(default_rates.index)}

# Convert bucket boundaries and ratings to Python-native types
rating_map = {
    "boundaries": [int(b) for b in bucket_boundaries],  # Convert to Python int
    "ratings": {str(k): int(v) for k, v in ratings.items()},  # Convert keys to str and values to int
}

# Save bucket boundaries and ratings dynamically to a JSON file
with open("rating_map.json", "w") as f:
    json.dump(rating_map, f)

print("Dynamic Rating Map saved to rating_map.json")

# Plot the default rates by bucket
plt.figure(figsize=(10, 6))
default_rates.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Default Rates by FICO Score Bucket")
plt.xlabel("FICO Bucket")
plt.ylabel("Default Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dynamic_fico_default_rates_plot.png")
plt.show()
