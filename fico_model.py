import json
import pandas as pd

# Load the rating map from JSON
with open("rating_map.json", "r") as f:
    rating_map = json.load(f)

boundaries = rating_map["boundaries"]
ratings = rating_map["ratings"]

# Define a dynamic mapping function
def fico_to_rating(fico_score, boundaries, ratings):
    for i in range(len(boundaries) - 1):
        if boundaries[i] < fico_score <= boundaries[i + 1]:
            bucket = f"({boundaries[i]}, {boundaries[i + 1]}]"
            return ratings.get(bucket, "Unknown")
    return "Invalid FICO Score"

# Load sample data for testing
sample_data_path = "sample_task_3_4.csv"  # Use the sample data file
sample_data = pd.read_csv(sample_data_path)

# Test dynamic mapping function on sample data
print("\nTesting on Sample Data:")
for _, row in sample_data.iterrows():
    score = row["fico_score"]
    rating = fico_to_rating(score, boundaries, ratings)
    print(f"FICO Score: {score}, Rating: {rating}")
