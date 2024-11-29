import pandas as pd
import numpy as np

# Define a cache for log-likelihood calculations
log_likelihood_cache = {}

# Optimized log-likelihood calculation with caching
def compute_log_likelihood(bucket):
    bucket_key = (bucket.index[0], bucket.index[-1])  # Unique key for this range
    if bucket_key in log_likelihood_cache:
        return log_likelihood_cache[bucket_key]
    
    n = len(bucket)  # Total records in the bucket
    k = bucket['default'].sum()  # Number of defaults in the bucket
    if n == 0 or k == 0 or k == n:
        return 0  # Avoid log(0) by ignoring empty or pure buckets
    p = k / n
    log_likelihood = k * np.log(p) + (n - k) * np.log(1 - p)
    log_likelihood_cache[bucket_key] = log_likelihood  # Cache the result
    return log_likelihood

# Optimized dynamic programming for bucket optimization
def optimize_buckets(data, num_buckets):
    scores = data['fico_score'].values
    n = len(scores)

    # DP table to store the best log-likelihood for each range and bucket count
    dp = np.zeros((num_buckets + 1, n + 1))
    splits = np.zeros((num_buckets + 1, n + 1), dtype=int)

    # Base case: One bucket for the entire range
    for i in range(1, n + 1):
        dp[1][i] = compute_log_likelihood(data.iloc[:i])

    # Fill DP table
    # Inside the loop for bucket optimization
    for b in range(2, num_buckets + 1):
        print(f"Processing bucket {b}/{num_buckets}...")
        for i in range(b, n + 1):
            print(f" - Evaluating range ending at {i}/{n}")

            best_split = -1
            best_value = -float('inf')

            for j in range(b - 1, i):  # Potential split point
                # Use caching for compute_log_likelihood
                current_value = dp[b - 1][j] + compute_log_likelihood(data.iloc[j:i])
                if current_value > best_value:
                    best_value = current_value
                    best_split = j

            dp[b][i] = best_value
            splits[b][i] = best_split

    # Backtrack to find the bucket boundaries
    boundaries = []
    endpoint = n
    for b in range(num_buckets, 0, -1):
        start = splits[b][endpoint]
        boundaries.append((scores[start], scores[endpoint - 1]))
        endpoint = start

    return boundaries[::-1]  # Return boundaries in ascending order
