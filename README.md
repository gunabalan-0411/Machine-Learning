## Notes:



## Linear Regression

### Feature Transformation and Scaling:
**If data is skewed → apply a transformer first, THEN apply a scaler.**

Transformer: change the shape of the distribution
  - Ex: log transform, square root, box-cox, RBF, Percentile, quantile transformer
Scaling: Scaling NEVER changes the shape of the distribution. It only changes the size.
  - Ex: StandardScaler, MinMaxScaler, RobustScaler
    
A. **Data is right-skewed (long tail)**
  -   Log transform (best for multiplicative data)
  -   Square root (mild skew)
  Then apply StandardScaler or MinMaxScaler.
B. Data has outliers
  - RobustScaler (because it uses median + IQR → ignores outliers)
  - Transform (log) → scale (RobustScaler)
C. Data has multiple peaks (multimodal)
  - Bucketizing (binning)
  - One-hot encoding
  - RBF kernel (for similarity)
D. Data has no skew but needs normalization
  - StandardScaler, MinMaxScaler

  **After transforming, scaling makes more sense.**

#### StandardScaler
  - $z = \frac{x - \mu}{\sigma}$.
      - 𝑥 = original value
      - 𝜇 = mean of the original data
      - 𝜎 = standard deviation of the original data
  - First it subtracts the mean values (so that there is zero mean) and divide by standard deviation (so that standardized values has standard deviation of 1)
  - Standardization is much less affected by outliers.

#### Inverse Transform
  - Easy way to automatically re-calculate the original value after scaling, using .inverse_transform()
  - Much more easy way to do the above is using TransformedTargetRegressor(regression_model, transformer = anytransformer)

#### Custom Transformers
    
