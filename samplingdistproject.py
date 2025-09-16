import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------------------------------------
# 1-3. Create example data and focus on tempo
# ---------------------------------------
np.random.seed(42)  # For reproducibility
# Let's simulate 500 songs with tempos roughly between 100-180 bpm
song_tempos = np.random.normal(loc=130, scale=15, size=500)
spotify_data = pd.DataFrame({'tempo': song_tempos})

# Preview the data
print(spotify_data.head())

# ---------------------------------------
# 5. Population Distribution
# ---------------------------------------
plt.hist(song_tempos, bins=20, edgecolor='black')
plt.title("Population Distribution of Song Tempos")
plt.xlabel("Tempo (bpm)")
plt.ylabel("Frequency")
plt.show()

# ---------------------------------------
# 6. Sampling Distribution of the Mean
# ---------------------------------------
sample_size = 30
num_samples = 1000
sample_means = [np.mean(np.random.choice(song_tempos, sample_size)) for _ in range(num_samples)]

plt.hist(sample_means, bins=20, edgecolor='black')
plt.title("Sampling Distribution of the Mean")
plt.axvline(np.mean(sample_means), color='r', linestyle='dashed', linewidth=2, label='Mean of Sample Means')
plt.axvline(np.mean(song_tempos), color='g', linestyle='solid', linewidth=2, label='Population Mean')
plt.legend()
plt.show()

# Mean of sample means vs population mean
print(f"Mean of Sample Means: {np.mean(sample_means):.2f}")
print(f"Population Mean: {np.mean(song_tempos):.2f}")

# ---------------------------------------
# 8. Sampling Distribution of the Minimum
# ---------------------------------------
sample_mins = [np.min(np.random.choice(song_tempos, sample_size)) for _ in range(num_samples)]

plt.hist(sample_mins, bins=20, edgecolor='black')
plt.title("Sampling Distribution of the Minimum")
plt.axvline(np.mean(sample_mins), color='r', linestyle='dashed', linewidth=2, label='Mean of Sample Minimums')
plt.axvline(np.min(song_tempos), color='g', linestyle='solid', linewidth=2, label='Population Minimum')
plt.legend()
plt.show()

# ---------------------------------------
# 10. Sampling Distribution of the Variance
# ---------------------------------------
# Use ddof=1 to get unbiased sample variance
sample_vars = [np.var(np.random.choice(song_tempos, sample_size), ddof=1) for _ in range(num_samples)]

plt.hist(sample_vars, bins=20, edgecolor='black')
plt.title("Sampling Distribution of the Variance")
plt.axvline(np.mean(sample_vars), color='r', linestyle='dashed', linewidth=2, label='Mean of Sample Variances')
plt.axvline(np.var(song_tempos, ddof=1), color='g', linestyle='solid', linewidth=2, label='Population Variance')
plt.legend()
plt.show()

# ---------------------------------------
# 13. Population mean and std
# ---------------------------------------
population_mean = np.mean(song_tempos)
population_std = np.std(song_tempos, ddof=1)
print(f"Population Mean: {population_mean:.2f}")
print(f"Population Std Dev: {population_std:.2f}")

# ---------------------------------------
# 14. Standard Error for sample mean (n=30)
# ---------------------------------------
standard_error = population_std / np.sqrt(sample_size)
print(f"Standard Error: {standard_error:.2f}")

# ---------------------------------------
# 15. Probability sample mean < 140
# ---------------------------------------
prob_less_140 = norm.cdf(140, loc=population_mean, scale=standard_error)
print(f"Probability sample mean < 140 bpm: {prob_less_140:.4f}")

# ---------------------------------------
# 16. Probability sample mean > 150
# ---------------------------------------
prob_greater_150 = 1 - norm.cdf(150, loc=population_mean, scale=standard_error)
print(f"Probability sample mean > 150 bpm: {prob_greater_150:.4f}")
