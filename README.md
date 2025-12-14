# Advanced Differential Privacy Algorithms

This repository contains the Python implementation of core Differential Privacy (DP) mechanisms, focusing on both Global and Local DP protocols.

## Project Goal

To implement and empirically evaluate the privacy-utility trade-off across several DP algorithms, demonstrating a deep understanding of data security and privacy-preserving techniques.

## I. Global Differential Privacy (Global DP)

These mechanisms are applied to aggregate results, protecting the output from being used to infer details about individuals.

* **Laplace Mechanism:** Implemented for secure **range queries** (histogram estimation) on a COVID-19 dataset.
* **Exponential Mechanism:** Implemented for securely selecting the **maximum value** (the month with the highest deaths).

## II. Local Differential Privacy (LDP)

These mechanisms add noise directly to individual data points before they are collected, ensuring privacy from the start.

* **Generalized Random Response (GRR):** Implemented for accurate **frequency estimation**.
* **Optimized Unary Encoding (OUE):** Implemented for improved LDP **frequency estimation**.



## Skills Demonstrated

* Python, NumPy, Pandas
* Algorithmic implementation from scratch.
* Mastery of $\epsilon$-Differential Privacy and sensitivity.
* Empirical analysis of error (utility) vs. privacy ($\epsilon$).
