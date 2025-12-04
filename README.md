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
  In Two ways
  1. By using sklearn.preprocessing FunctionTransformer(), we can pass a custom function and thereby we can also combine two features while computing.
  2. A transformer class or function is
    - takes X (input data)
    - learns something from it (in fit()) --> mean_, scale_
    - transforms the data (in transform()) --> Uses what was learned in fit() to change the data.: subtract mean, divide by std
    - fit_transform(X) is optional, usually comes with if we inherit TransformerMixin
  3. Why inherit from BaseEstimator and TransformerMixin
       - get_params(), set_params() these are useful in gridsearchcv
       - TransformerMixin gives the optional fit_transform(X)
     
  ```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
  def __init__(self, with_mean = True):
    self.with_mean = with_mean
  def fit(self, X, y=None):
    # data validation
    X = check_array(X)
    # This is what we learn in fit function
    self.mean_ = X.mean(axis = 0)
    self.scale_ = X.std(axis = 0)
    self.n_features_in_ = X.shape[1]
    return self # should always return self to avoid base functionality interruption
  def transform(self, X):
    check_is_fitted(self)
    X = check_array(X)
    
    assert self.n_features_in_ == X.shape[1]
    if self.with_mean:
      X = X - self.mean_
    return X / self.scale_

std_scaler = StandardScalerClone()
std_scaler.fit(housing[['median_income']])
std_scaler.transform(housing[['median_income']])

# One more example is in fit we can use Kmeans to find the cluster center and use rbf in transform to find the similarity of each value in features with cluster center.
cluster center: [5, 10, 30], X = 12, it will calculate the similarity and brings nonlinearity for model 
  ```
  
