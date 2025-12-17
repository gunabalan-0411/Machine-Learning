## Notes:

- In Sklearn Machine learning fit always means that the model calculate or learn some stats from the data first
- Predict means based on the information got from fit function, it will calculate or predict.


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
  - RBF(Radial Basis function) kernel (for similarity)
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
### Pipeline
  - Pipeline() helps to sequentially orchestrate the data transformation and prediction
  - pipeline.fit, predict, transform is based last estimator and everything else before that will fit_transform.
  - by using ColumnTransformer, to perform numerical and text transformation all in one
  - And use make_column_selector to automatically select columns based on data types.
  ``` python
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import StandardScaler
  from sklearn.pipeline import make_pipeline
  from sklearn.compose import make_column_selector, make_column_transformer
  
  num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
                 "total_bedrooms", "population", "households", "median_income"]
  cat_attribs = ["ocean_proximity"]
  
  num_pipeline = Pipeline([
      ("impute", SimpleImputer(strategy="median")),
      ("standardize", StandardScaler()),
  ])
  
  cat_pipeline = make_pipeline(
      SimpleImputer(strategy="most_frequent"),
      OneHotEncoder(handle_unknown="ignore")
  )
  
  # preprocessing = ColumnTransformer([
  #     ('num', num_pipeline, num_attribs),
  #     ('cat', cat_pipeline, cat_attribs)
  # ])
  
  # Automatically select the numerical and category columns
  
  preprocessing = make_column_transformer(
      (num_pipeline, make_column_selector(dtype_include = np.number)),
      (cat_pipeline, make_column_selector(dtype_include = object))
  )
  housing_preprocessed = preprocessing.fit_transform(housing)
  ```
### Cross val score and evaluation
using cross_val_score to check how generalize the model is

  - using sklearn.metrics , .model_selection to find root_mean_square_error, cross_val_score to find the model with minimum error

### Finetuning model
using gridsearchcv to perform hyperparameter tuning, includig cross val score
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

full_pipeline = Pipeline(
    [
        ('preprocessing', preprocessing),
        ('random_forest', RandomForestRegressor(random_state=42))
    ]
)

param_grid = [
    {
        'preprocessing__geo__n_clusters': [5,8,10],
        'random_forest__max_features': [4,6,8]
    },
    {
        'preprocessing__geo__n_clusters': [10, 15],
        'random_forest__max_features': [6, 8, 10]
    }
]

grid = GridSearchCV(full_pipeline, param_grid, cv=3, scoring = 'neg_root_mean_squared_error')

grid.git(housing, housing_labels)
grid._best_params
```
#### RandomizedSearchCV
RandomizedSearchCV is often preferable, especially when the hyperparameter search space is large. This class can be used in much the same way as the GridSearchCV class, but instead of trying out all possible combinations it evaluates a fixed number of combinations, selecting a random value for each hyperparameter at every iteration.

```python
final_model = rand_search.best_estimator_
final_model['random_forest'].feature_importance_ # print the feature importance score, so that we can remove features with low importance.
```
#### Confidence Interval
Point estimate tells you “how good” your model is.
Confidence interval tells you “how sure” you are.
If your confidence interval is wide, the improvement is meaningless.
bootstrap: Randomly resample from set of sample again and again, and see how RMSE changes.
```python
from scipy.stats import bootstrap
def rmse(square_errors):
  return np.sqrt(np.mean(square_errors))

square_errors = (y_pred - y_test) ** 2

ci = bootstrap([square_errors], rmse, confidence_level = 0.95, random_state = 42)
print(ci.confidence_interval)

#Old model: 41,500
#New model CI: [41,200 , 41,300] ->Clear improvement → safe to deploy
#New model CI: [39,521 , 43,702] ->Too much uncertainty → risky to deploy
```
#### Deployment of Models
using joblib to save model in pkl format.

## K Means
k-means is a stochastic algorithm, meaning that it relies on randomness to locate the clusters, so if you want reproducible results, you must set the random_state parameter

