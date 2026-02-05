## Notes:

- In Sklearn Machine learning fit always means that the model calculate or learn some stats from the data first
- Predict means based on the information got from fit function, it will calculate or predict.

## K Nearest Neighbors
* hyper-parameter:
  * weights: uniform, distance // weight for ranking (uniform make each neighbor equal, decision based on count alone) but distance is closest more weight.
  * n_neighbors: range(1 to ...) // no of neighbors to consider
* new sample x will be compare against all sample and choose the k nearest neighbors and choose the one that wins by weights.


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

* If x is left-skewed:
* Make it right-skew: x_reflect = max(x) + 1 - x
* Then apply log/sqrt on x_reflect
📌 Typical examples:
  * Exam marks where most score high
  * “Battery health %” where most are near 100%
  * “Customer satisfaction rating” mostly 4–5

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
Important variables from grid_search
* grid_search.best_params_ 
* grid_search.best_score_
* grid_search.best_estimator_

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


## Classification
### Binary Classification
### Stochastic Gradient Descent (SGD) Classifier (Linear Model)
* Production grade model for very large dataset and used for online training
* Initialize weights randomly.
* Pick one training sample (or a small batch).
* Make a prediction using current weights.
* Compute the loss (error) for that sample.
* Update weights in the opposite direction of the gradient:

### Evaluation:
* $\text{Precision} = \frac{TP}{TP + FP}$
* $\text{Recall} = \frac{TP}{TP + FN}$
* When it claims an image represents a 5, it is correct only 83.7% of the time. Moreover, it only detects 65.1% of the 5s
* F1 score is the harmonic mean of precision and recall
* $\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
* In text classification, based on precision recall we determine the decision_function threshold value by plotting the precision_recall_curve along with threshold.
* Suppose based on the projet, we determine 90% precision then choose the thresold which will provide that precision in precision_recall graph.
The Precision/Recall Trade-Off
<img width="879" height="394" alt="image" src="https://github.com/user-attachments/assets/bc8d095d-9637-4db6-8e4a-39b3fca5ef20" />

ROC (Receiver Operating Characteristic) Curve
* Many classifiers don’t just output 0 or 1 directly, Instead, they output a probability score: Ex: if score ≥ 0.5 → predict Positive
* Classification performance changes depending on threshold.
* ROC curve shows how performance changes across all thresholds.
* the ROC curve plots the true positive rate (another name for recall) against the false positive rate (FPR)
* $\text{FPR} = \frac{FP}{FP + TN}$
* The FPR (also called the fall-out) is the ratio of negative instances that are incorrectly classified as positive.
* X-axis --> FPR, Y-axis --> TPR is plotted against all 0 to 1.0 threshold
* Random classifier If model is guessing: the ROC curve becomes a diagonal line: TPR = FPR
Area Under the Curve (AUC)
* It is the area under ROC curve (0 to 1).
* AUC = 1.0 → perfect classifierc.
* So AUC answers: “How good is this model at separating positives vs negatives using scores?”

ROC vs Precision-Recall (very important)

* When dataset is highly imbalanced, ROC can still look very good even if model is bad.
* That’s why:
* ROC is good generally
* PR curve is better when positives are rare
Example:
* Fraud detection (1% fraud)
* You should prefer Precision-Recall curve.

Error rate calculation (- 1 means, it brings the how much percentage up/ shit by reducing 1 to ratio)
* error_rate = (1- new_accuracy /1 - old_accuracy) - 1

### Multiclass Classification

* Support Vector Machine Classifiers scale poorly with the size of the training set
* When Multi-class data passed on to Scikit learn, it automatically train One vs One class (for 10 classes it will 45 model kind of). and the decision score is calculated for all target classes.
* There is dedicated class in sklearn to perform it as well. `from sklearn.multiclass import OneVsRestClassifier`
* Using confusion matrix to plot and check with class is lacking to be predicted and based on that we can add features
* Ex: writing an algorithm to count the number of closed loops (e.g., 8 has two, 6 has one, 5 has none).
* we can plot images as confusion matrix (ex: actual 3 but predicted 5 and get that feature to plot that in plt)

### Multi-Label Classification
* for the model that do not support multi-label classifier by default(such as svc), we can create one model for one label or we can use ClassifierChain from sklearn.multioutput

### Multi-output Classification
* the classifier’s output is multilabel (one label per pixel) and each label can have multiple values (pixel intensity ranges from 0 to 255).
* Ex: we can add noise to digit dataset and using a classifier to predict clean image, here multilabel is (0-9) and each pixel values can have multiple values range from 0-255

## Data Augmentation
* from scipy.ndimage import shift, shift(img, (dy, dx)) will reshape the image.

## Model Training

