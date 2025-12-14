import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random
import csv
from numpy import sqrt, exp

''' Functions to implement '''

# TODO: Implement this function!
def read_dataset(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# TODO: Implement this function!
def get_histogram(dataset, state='TX', year='2020'):
    year_int = int(year)
    # Filter by state and year
    df_filtered = dataset[(dataset['state'] == state) & (dataset['date'].dt.year == year_int)]
    # Group by month and sum the 'positive' column
    monthly_counts = df_filtered.groupby(df_filtered['date'].dt.month)['positive'].sum().to_dict()
    # Create the histogram list, ensuring all 12 months are present (0 if missing)
    histogram = [monthly_counts.get(m, 0) for m in range(1, 13)]
    return histogram

# TODO: Implement this function!
def get_dp_histogram(dataset, state, year, epsilon, N):
    true_hist = get_histogram(dataset, state, year)
    # Sensitivity is N
    # Scale for Laplace noise is Sensitivity / epsilon
    scale = N / epsilon
    dp_hist = []
    for count in true_hist:
        noise = np.random.laplace(loc=0.0, scale=scale)
        dp_hist.append(count + noise)
    return dp_hist

# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    n_bins = len(actual_hist)
    total_error = sum(abs(actual_hist[i] - noisy_hist[i]) for i in range(n_bins))
    return total_error / n_bins

# TODO: Implement this function!
def epsilon_experiment(dataset, state, year, eps_values, N):
    results = []
    true_hist = get_histogram(dataset, state, year)

    for eps in eps_values:
        errors = []
        for _ in range(10):
            dp_hist = get_dp_histogram(dataset, state, year, eps, N)
            err = calculate_average_error(true_hist, dp_hist)
            errors.append(err)
        results.append(statistics.mean(errors))

    return results

# TODO: Implement this function!
def N_experiment(dataset, state, year, epsilon, N_values):
    results = []
    true_hist = get_histogram(dataset, state, year)

    for N in N_values:
        errors = []
        for _ in range(10):
            dp_hist = get_dp_histogram(dataset, state, year, epsilon, N)
            err = calculate_average_error(true_hist, dp_hist)
            errors.append(err)
        results.append(statistics.mean(errors))

    return results

# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #

# TODO: Implement this function!
def max_deaths_exponential(dataset, state, year, epsilon):
    year_int = int(year)
    df_filtered = dataset[(dataset['state'] == state) & (dataset['date'].dt.year == year_int)]
    # Get death counts for each month
    monthly_deaths = df_filtered.groupby(df_filtered['date'].dt.month)['death'].sum().to_dict()
    # Candidates are months 1 to 12
    candidates = list(range(1, 13))
    # Utility function u(d, r) is the death count for the month
    scores = [monthly_deaths.get(m, 0) for m in candidates]
    # Sensitivity: Adding/removing one person changes the death count of a month by at most 1.
    sensitivity = 1.0

    # Calculate probabilities: exp(epsilon * score / (2 * sensitivity))
    max_score = max(scores)
    # Numerical stability shift
    scaled_scores = [(epsilon * (s - max_score)) / (2 * sensitivity) for s in scores]
    probabilities = [exp(s) for s in scaled_scores]

    # Normalize probabilities
    total_prob = sum(probabilities)
    normalized_probs = [p / total_prob for p in probabilities]

    # Select a month based on the probabilities
    selected_month = np.random.choice(candidates, p=normalized_probs)

    return selected_month

# TODO: Implement this function!
def exponential_experiment(dataset, state, year, epsilon_list):
    year_int = int(year)
    df_filtered = dataset[(dataset['state'] == state) & (dataset['date'].dt.year == year_int)]

    # Determine the true max month(s)
    monthly_deaths = df_filtered.groupby(df_filtered['date'].dt.month)['death'].sum().to_dict()
    true_counts = [monthly_deaths.get(m, 0) for m in range(1, 13)]
    max_deaths = max(true_counts)

    # There might be multiple months with the same max death count
    true_max_months = [m for m in range(1, 13) if monthly_deaths.get(m, 0) == max_deaths]

    accuracies = []
    num_trials = 10000

    for eps in epsilon_list:
        correct_count = 0
        for _ in range(num_trials):
            result = max_deaths_exponential(dataset, state, year, eps)
            if result in true_max_months:
                correct_count += 1

        accuracies.append(correct_count / num_trials)

    return accuracies

# FUNCTIONS TO IMPLEMENT END #

def main():
    filename = "covid19-states-history.csv"
    dataset = read_dataset(filename)

    state = "TX"
    year = "2020"



    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg = epsilon_experiment(dataset, state, year, eps_values, 2)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])


    print("**** N EXPERIMENT RESULTS ****")
    N_values = [1, 2, 4, 8]
    error_avg = N_experiment(dataset, state, year, 0.5, N_values)
    for i in range(len(N_values)):
        print("N = ", N_values[i], " error = ", error_avg[i])

    state = "WY"
    year = "2020"

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
    exponential_experiment_result = exponential_experiment(dataset, state, year, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])



if __name__ == "__main__":
    main()