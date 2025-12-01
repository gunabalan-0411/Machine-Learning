### Linear Regression

### Before scaling:
We need to check if the distribution is skewed to right or left, if not scaling will fit into the scale but training will get affect
Transformation methods:
  - to handle heavy right tail we can replace the values with square-root
  - for really long tails we can replace with the *logarithm value* (np.log).

#### Feature scaling:
Standardization:

  - $z = \frac{x - \mu}{\sigma}$.
      - 𝑥 = original value
      - 𝜇 = mean of the original data
      - 𝜎 = standard deviation of the original data
  - First it subtracts the mean values (so that there is zero mean) and divide by standard deviation (so that standardized values has standard deviation of 1)
  - Standardization is much less affected by outliers.
  - 
