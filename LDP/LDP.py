import math, random
import matplotlib.pyplot as plt

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """


def read_dataset(filename):
    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


def avg_error(true_hist, est_hist):
    """
    Calculates the Average Error between true and estimated histograms.
    AvgErr = Sum(|True - Est|) / |Domain|
    """
    if len(true_hist) != len(est_hist):
        raise ValueError("Histograms must be same length")

    total_diff = sum(abs(t - e) for t, e in zip(true_hist, est_hist))
    return total_diff / len(true_hist)


def get_true_histogram(dataset):
    """
    Computes the true frequency of each category in the dataset.
    Returns a list of counts for categories 1 to 17.
    """
    counts = {cat: 0 for cat in DOMAIN}
    for val in dataset:
        if val in counts:
            counts[val] += 1
    # Convert to list ordered by domain (1..17)
    return [counts[cat] for cat in DOMAIN]

# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    d = len(DOMAIN)
    exp_eps = math.exp(epsilon)

    # Probability p of reporting the true value
    p = exp_eps / (exp_eps + d - 1)

    # Probability q of reporting any other specific value
    # q = 1 / (exp_eps + d - 1)

    if random.random() < p:
        return val
    else:
        other_values = [x for x in DOMAIN if x != val]
        return random.choice(other_values)


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    n = len(perturbed_values)
    d = len(DOMAIN)
    exp_eps = math.exp(epsilon)

    p = exp_eps / (exp_eps + d - 1)
    q = 1.0 / (exp_eps + d - 1)

    # Count frequencies of perturbed values
    observed_counts = {cat: 0 for cat in DOMAIN}
    for val in perturbed_values:
        if val in observed_counts:
            observed_counts[val] += 1

    estimated_hist = []
    for cat in DOMAIN:
        n_y = observed_counts[cat]
        # Unbiased estimator formula: (n_y - n*q) / (p - q)
        est_count = (n_y - n * q) / (p - q)
        estimated_hist.append(est_count)

    return estimated_hist


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    perturbed_data = [perturb_grr(val, epsilon) for val in dataset]

    estimated_counts = estimate_grr(perturbed_data, epsilon)
    true_counts = get_true_histogram(dataset)
    return avg_error(true_counts, estimated_counts)


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    # Initialize all zeros
    bit_vector = [0] * len(DOMAIN)

    if val in DOMAIN:
        bit_vector[val - 1] = 1
    return bit_vector


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    p = math.exp(epsilon / 2) / (math.exp(epsilon / 2) + 1)

    perturbed_vector = []
    for bit in encoded_val:

        if random.random() < p:
            perturbed_vector.append(bit)
        else:
            perturbed_vector.append(1 - bit)

    return perturbed_vector


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    n = len(perturbed_values)
    d = len(DOMAIN)

    p = math.exp(epsilon / 2) / (math.exp(epsilon / 2) + 1)
    q = 1 - p

    observed_sums = [0] * d
    for vector in perturbed_values:
        for i in range(d):
            observed_sums[i] += vector[i]

    estimated_hist = []
    for i in range(d):
        t_i = observed_sums[i]
        est_count = (t_i - n * q) / (p - q)
        estimated_hist.append(est_count)

    return estimated_hist


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    perturbed_data = []
    for val in dataset:
        encoded = encode_rappor(val)
        perturbed = perturb_rappor(encoded, epsilon)
        perturbed_data.append(perturbed)

    estimated_counts = estimate_rappor(perturbed_data, epsilon)

    true_counts = get_true_histogram(dataset)
    return avg_error(true_counts, estimated_counts)


# OUE

# TODO: Implement this function!
def encode_oue(val):
    return encode_rappor(val)


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    p = 0.5
    q = 1.0 / (math.exp(epsilon) + 1)

    perturbed_vector = []
    for bit in encoded_val:
        if bit == 1:
            # Output 1 with prob p
            if random.random() < p:
                perturbed_vector.append(1)
            else:
                perturbed_vector.append(0)
        else:  # bit == 0
            # Output 1 with prob q
            if random.random() < q:
                perturbed_vector.append(1)
            else:
                perturbed_vector.append(0)

    return perturbed_vector


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    n = len(perturbed_values)
    d = len(DOMAIN)

    p = 0.5
    q = 1.0 / (math.exp(epsilon) + 1)

    # Sum up the columns
    observed_sums = [0] * d
    for vector in perturbed_values:
        for i in range(d):
            observed_sums[i] += vector[i]

    estimated_hist = []
    for i in range(d):
        t_i = observed_sums[i]
        # Estimator: (T_i - n*q) / (p - q)
        est_count = (t_i - n * q) / (p - q)
        estimated_hist.append(est_count)

    return estimated_hist


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    perturbed_data = []
    for val in dataset:
        encoded = encode_oue(val)
        perturbed = perturb_oue(encoded, epsilon)
        perturbed_data.append(perturbed)

    # 2. Estimate
    estimated_counts = estimate_oue(perturbed_data, epsilon)

    # 3. Calculate Error
    true_counts = get_true_histogram(dataset)
    return avg_error(true_counts, estimated_counts)


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")

    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))



if __name__ == "__main__":
    main()

