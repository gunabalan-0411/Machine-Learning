# Machine Learning Study Notes — Organized, Corrected, and Expanded

> Goal: Clean structure + correct technical issues + add missing topics.

---

# 0) Big Picture: AI → ML → DL

## AI (Artificial Intelligence)

* **Goal:** build systems that perform tasks that require human-like intelligence.
* Includes reasoning, planning, perception, language, decision-making.

## ML (Machine Learning)

* **Subset of AI**.
* Learns patterns from data to make predictions/decisions.
* Example: classify spam, predict churn.

## DL (Deep Learning)

* **Subset of ML**.
* Uses neural networks with many layers.
* Learns representations automatically (feature learning).

✅ Correction: DL is not literally “think like human”; it is function approximation with layered representation learning.

---

# 1) Core Learning Types

## 1.1 Supervised Learning

* Data: `(X, y)` (features + labels)
* Goal: learn mapping `f(X) → y`

Tasks:

* Classification (spam / not spam)
* Regression (price prediction)

---

## 1.2 Unsupervised Learning

* Data: `X` only (no labels)

Tasks:

* Clustering (group similar customers)
* Dimensionality reduction (PCA)

---

## 1.3 Semi-supervised Learning

* Small amount of labeled data + lots of unlabeled data.

Use case:

* labeling is expensive (medical images).

---

## 1.4 Self-supervised Learning (Correction)

Your note:

> randomly masked images as input and full original image as label

✅ Correct idea, but general definition:

* Model creates its own labels from input.
* It learns representations without human labels.

Examples:

* Masked token prediction (BERT)
* Masked image modeling (MAE)
* Contrastive learning (SimCLR)

---

## 1.5 Reinforcement Learning (Missing topic ✅)

* Agent interacts with environment.
* Learns a policy to maximize reward.

Example:

* game playing
* robotics
* RLHF for LLM alignment

---

# 2) Training Styles

## Offline vs Online learning (Correction)

### Offline / Batch Learning

* Train once on a fixed dataset.
* Deploy model.
* Retrain periodically.

### Online / Incremental Learning

* Model updates continuously as new data arrives.
* Used in streaming/real-time systems.

⚠️ Gradient descent can be used in both.
Online learning often uses mini-batch / SGD on streams.

---

# 3) Key Concepts

## 3.1 Underfitting vs Overfitting

### Underfitting

* Model too simple.
* High bias.
* Poor performance on train + test.

### Overfitting

* Model too complex.
* High variance.
* Great performance on train, poor on test.

---

## 3.2 Bias–Variance Tradeoff

* **Bias:** error from simplifying assumptions
* **Variance:** error from sensitivity to dataset

Goal: find balance.

---

## 3.3 Regularization

Goal: reduce overfitting by penalizing complexity.

Common types:

* L1 (Lasso)
* L2 (Ridge)
* Dropout (DL)
* Early stopping

---

## 3.4 Hyperparameters vs Parameters (Correction)

### Model Parameters

* learned from data
* e.g., linear regression weights `θ`

### Hyperparameters

* chosen before/during training
* e.g., learning rate, lambda, max_depth

---

# 4) Data Splits and Validation

## 4.1 Train / Validation / Test

* Train: fit model
* Validation: tune hyperparameters
* Test: final evaluation (never touch during tuning)

---

## 4.2 Cross Validation

Purpose:

* estimate generalization
* tune hyperparameters robustly

Types (missing ✅):

* K-Fold
* Stratified K-Fold (classification imbalance)
* TimeSeriesSplit

---

# 5) Regression Models

## 5.1 Linear Regression

Equation:
`y = θ0 + θ1 x`

* `θ0` = intercept
* `θ1` = slope

Loss (MSE):
`MSE = (1/n) Σ (y - ŷ)^2`

Optimization:

* Gradient Descent (batch/mini-batch/SGD)
* Normal equation (small datasets)

✅ Correction: linear regression cost surface is convex → **no local minima**.

---

## 5.2 Regularized Regression

### Ridge Regression (L2)

Adds penalty:
`Loss = MSE + λ Σ θ^2`

* reduces large weights
* good when many correlated features

### Lasso Regression (L1) 

Adds penalty:
`Loss = MSE + λ Σ |θ|`

* can force some weights to exactly 0
* feature selection

### Elastic Net (Missing ✅)

Combination:

* L1 + L2

---

# 6) Classification Models

## 6.1 Logistic Regression

Used for binary classification.

Sigmoid:
`p = 1 / (1 + e^{-z})`
where:
`z = θ0 + θ1x1 + ... + θdx_d`

Loss:

* Log loss / binary cross entropy

Imbalance handling:

* `class_weight="balanced"`
* resampling (SMOTE / undersampling)

---

## 6.2 k-NN (Classification + Regression)

Idea:

* find k closest training points
* classification: majority vote
* regression: average

Distance:

* Euclidean
* Manhattan

Weakness:

* slow prediction for big data
* sensitive to feature scaling

---

## 6.3 Naive Bayes

Based on Bayes theorem with independence assumption.

Variants:

* MultinomialNB (text)
* GaussianNB
* BernoulliNB

Great for:

* NLP baseline

---

## 6.4 SVM

* Finds hyperplane with max margin.
* Support vectors define boundary.

Kernels:

* linear
* polynomial
* RBF

Good for:

* high-dimensional small/medium datasets

---

# 7) Tree-based Models

## 7.1 Decision Trees

Both:

* classification
* regression

Split criteria:

* Gini Impurity (CART)
* Entropy + Information Gain (ID3/C4.5)

Stopping:

* max_depth
* min_samples_split
* pure node

Weakness:

* overfits easily

---

## 7.2 Ensemble Learning

### Bagging

Train models independently in parallel.

* regression: average
* classification: majority vote

Example:

* Random Forest

### Random Forest

* bagging + feature randomness
* strong baseline for tabular

Random Forest is **less sensitive**, not “won’t affect by outlier”.

---

### Boosting

Sequential learning.
Each next model learns from previous mistakes.

Examples:

* AdaBoost
* Gradient Boosting
* XGBoost
* LightGBM (Missing ✅)
* CatBoost (Missing ✅)

✅ Correction: Boosting can overfit if:

* too many trees
* high depth
* no regularization

---

# 8) Unsupervised Learning

## 8.1 K-Means Clustering

* choose k
* assign points to nearest centroid
* update centroids

Metric:

* WCSS

Choosing k:

* elbow method

Evaluation:

* Silhouette score (your description is good)

---

## 8.2 Hierarchical Clustering

* agglomerative (bottom-up)
* divisive (top-down)

Common linkages:

* single
* complete
* average
* ward

---

## 8.3 DBSCAN

Density-based clustering.

Pros:

* arbitrary shapes
* handles noise
* no need to choose k

Cons:

* struggles with varying densities

---

## 8.4 Dimensionality Reduction

### PCA

* finds directions of maximum variance
* orthogonal components

Why useful:

* remove multicollinearity
* speed up training
* visualization

Other missing ✅:

* t-SNE (visualization)
* UMAP (visualization + clustering support)

---

# 9) Model Evaluation Metrics

## 9.1 Classification metrics

Confusion matrix:

* TP, FP, FN, TN

Metrics:

* Accuracy
* Precision = TP/(TP+FP)
* Recall = TP/(TP+FN)
* F1

Missing but important ✅:

* ROC-AUC
* PR-AUC (best for imbalance)

---

## 9.2 Regression metrics

* MAE
* MSE
* RMSE
* R²

✅ Your MAE vs RMSE note is correct.

---

# 10) Data Preprocessing (Industry topics)

## Missing values

* SimpleImputer (mean/median/mode)
* KNNImputer
* IterativeImputer

---

## Categorical Encoding

✅ Your note about OneHotEncoder is correct.

Better for production:

* OneHotEncoder(handle_unknown="ignore")

Missing ✅:

* Ordinal encoding
* Target encoding (careful leakage)

---

## Feature Scaling

Missing ✅:

* StandardScaler
* MinMaxScaler

Needed for:

* kNN, SVM, Logistic Regression

Not needed for:

* Trees, Random Forest, XGBoost

---

## Outlier handling

Missing ✅:

* IQR rule
* z-score
* winsorization

---

# 11) Neural Networks (Deep Learning Basics)

## 11.1 What is a Neural Network?

Layers:

* Input layer
* Hidden layers
* Output layer

Hidden layers learn representations:

* edges
* shapes
* patterns

✅ Correction: NN does feature learning, but not always "without guidance" — training signal guides learning.

---

## 11.2 Activation Functions

Purpose:

* add non-linearity

Common:

* Sigmoid (output probability)
* ReLU (hidden layers)
* Softmax (multi-class output)

---

## 11.3 Backpropagation

* compute loss
* propagate gradients backwards
* update weights with optimizer

---

## 11.4 Vanishing Gradient

Cause:

* very small gradients in deep nets
* common in sigmoid/tanh

Fix:

* ReLU family
* BatchNorm
* Residual networks

---

## 11.5 Dropout

Randomly drop neurons during training.

* regularization method

---

# 12) Transformers (high level)

Your note:

> vector value of a word will be updated based on other words

✅ Correct idea.

Transformer uses:

* self-attention
* positional embeddings
* feedforward networks

Used in:

* BERT (encoder)
* GPT (decoder)

---

# 13) Generative AI + LLM Concepts

## 13.1 Agentic AI

System that:

* plans
* calls tools
* takes actions
* iterates

Not just text generation.

---

## 13.2 LLM Evaluation

### Perplexity

Measures how well model predicts next token.

* low perplexity: better LM fit

⚠️ But perplexity does NOT guarantee helpfulness or truth.

### ROUGE

Used for summarization.

* overlap-based

Missing ✅:

* BLEU (translation)
* BERTScore
* Human eval
* LLM-as-a-judge

---

## 13.3 MCP (Model Context Protocol)

* standard way for tools/servers to provide context to LLM systems.

---

# 14) EDA (Exploratory Data Analysis)

✅ Your note is directionally correct:

* avoid peeking test labels

But correction:

* You can look at distributions of features in full dataset, but avoid label leakage.

EDA checklist (Missing ✅):

* target distribution
* missingness heatmap
* feature distributions
* correlation
* leakage detection

---

# 15) Missing “Must-Know” Topics (Added)

## 15.1 Feature Engineering

* interaction terms
* polynomial features
* log transforms

## 15.2 Class imbalance

* SMOTE
* focal loss (DL)

## 15.3 Learning curves

Used to diagnose bias/variance.

## 15.4 Model calibration

* predicted probabilities reflect real likelihood
* Platt scaling / isotonic

## 15.5 Pipelines (production)

* sklearn Pipeline + ColumnTransformer
* prevents leakage

---

# Final Summary (Quick)

You covered many core topics well.
Major corrections made:

* kNN is supervised, not clustering
* offline/online learning definition corrected
* Random Forest not fully immune to outliers
* DL definition made technically accurate

Added missing topics:

* reinforcement learning
* TimeSeries CV
* LightGBM/CatBoost
* ROC-AUC/PR-AUC
* scaling and feature engineering
* pipelines + calibration

---

# Logistic Regression — Complete Notes + Interview Tricky Questions

This canvas is a **complete, industry-level** guide to Logistic Regression:

* fundamentals
* equations
* training + regularization
* multiclass / multilabel
* evaluation
* assumptions
* pitfalls
* interview questions (tricky)

---

# 1) Logistic Regression: Is it regression or classification?

## Short answer

✅ **Classification model**

## Why the name contains “regression”?

Because it is a type of **generalized linear model (GLM)** and it performs **regression on the log-odds**.

It models:

* probability of class
* using regression-style linear function inside.

So:

* output is probability
* decision is classification

---

# 2) When to use Logistic Regression

## Best for

* baseline classification
* interpretable model
* tabular data
* linearly separable / near-linear decision boundary

Examples:

* churn prediction
* spam detection
* fraud baseline
* disease classification

---

# 3) Model formulation

## 3.1 Linear model score

Given input features `x ∈ R^d`:

`z = w^T x + b`

Where:

* `w` = weights
* `b` = bias/intercept

---

## 3.2 Convert score to probability (Sigmoid)

`p(y=1|x) = σ(z) = 1 / (1 + e^{-z})`

Sigmoid maps any real number to (0,1).

---

## 3.3 Decision rule

Predict class:

`ŷ = 1 if p >= threshold else 0`

Default threshold = 0.5

But in practice you tune threshold using:

* business cost
* PR curve
* recall requirement

---

# 4) What is the cost function?

Logistic regression does NOT use MSE.
It uses **Log Loss / Cross-Entropy**.

For one sample:

`L = -[ y log(p) + (1-y) log(1-p) ]`

For dataset:

`J(w,b) = (1/n) Σ L_i`

---

# 5) Why not MSE for logistic regression? (Interview favorite)

### If we use MSE with sigmoid

* objective becomes **non-convex** in parameters
* gradient descent may get stuck

### Cross entropy

✅ gives convex loss → unique global optimum
✅ penalizes confident wrong predictions heavily

---

# 6) Training (Optimization)

Logistic regression is trained by minimizing log loss using:

* Gradient Descent
* SGD / mini-batch
* Newton-Raphson
* Quasi-Newton (LBFGS) (common in sklearn)
* Coordinate descent (useful for L1)

Dataset: 1,000,000 rows

GD:
* Computes gradient using all 1,000,000 rows
* Updates once per epoch

SGD:
* Uses 1 row
* Updates 1,000,000 times per epoch

SGD
* “Take steps in the direction of the gradient.”

Adam
* “Take smarter steps by remembering past gradients and adjusting step sizes automatically.”

✅ In sklearn, solver choices:

* `lbfgs` (default often)
* `liblinear`
* `saga`
* `newton-cg`

---

# 7) Regularization (very important)

Logistic regression overfits when:

* too many features
* strong multicollinearity

So we regularize.

## 7.1 L2 regularization (Ridge)

Penalty:

* `λ ||w||^2`

Effect:
✅ shrinks weights
✅ stable under correlation

---

## 7.2 L1 regularization (Lasso)

Penalty:

* `λ ||w||_1`

Effect:
✅ drives some weights to exactly 0
✅ feature selection

---

## 7.3 Elastic Net

* mix of L1 + L2

---

## 7.4 How sklearn represents λ (Tricky!)

In sklearn LogisticRegression:

* uses `C` not λ

Relationship:

* `C = 1/λ`

So:

* smaller `C` → stronger regularization
* bigger `C` → weaker regularization

---

# 8) Multiclass Logistic Regression

## Can logistic regression do multiclass?

✅ Yes.

Two approaches:

### 8.1 One-vs-Rest (OvR)

Train k binary classifiers.

Pros:

* simple
* works with most solvers

Cons:

* probability calibration can be messy

### 8.2 Multinomial Logistic Regression (Softmax Regression)

Directly model multiple classes:

`P(y=c|x) = exp(w_c^T x) / Σ_j exp(w_j^T x)`

Loss:

* categorical cross-entropy

Pros:
✅ best probabilistic modeling
✅ more consistent than OvR

In sklearn:

* `multi_class='multinomial'` with solver lbfgs/saga/newton-cg.

---

# 9) Multi-label classification with Logistic Regression

## Can logistic regression do multilabel?

✅ Yes, but not directly like softmax.

Method:

* train **one logistic regression per label**

Equivalent to:

* Binary relevance method

Example:
Email tags:

* [spam, promotions, social]

An email can be:

* spam=1, promotions=1, social=0

So for each label:

* independent sigmoid

---

# 10) Interpretation (Why logistic regression is loved)

## 10.1 Weight meaning

`w_j` describes effect of feature `x_j` on log-odds:

`log(p/(1-p)) = w^T x + b`

So if `w_j > 0`:

* increasing feature increases probability of class 1

If `w_j < 0`:

* decreases probability

---

## 10.2 Odds ratio

`exp(w_j)` is odds multiplier for one unit increase in `x_j`.

Example:
If `w_j = 0.7` → odds ratio = exp(0.7)=2.01
Meaning:

* odds roughly double per +1 unit of that feature.

---

# 11) Assumptions of Logistic Regression

Interviewers love this.

1. **Linear decision boundary** in feature space
2. Features have linear relationship with **log-odds**, not probability
3. No perfect multicollinearity
4. Independent observations
5. Large sample size helpful

✅ It does NOT require normal distribution of features.

---

# 12) Common pitfalls

## 12.1 Feature scaling

Logistic regression is sensitive to scale.

So scale features (important):

* StandardScaler

Especially for:

* regularized LR

---

## 12.2 Imbalanced datasets

Accuracy becomes misleading.

Fix:

* `class_weight='balanced'`
* resampling
* tune decision threshold

---

## 12.3 Perfect separation

If data is perfectly separable:

* weights can blow up (→ infinity)

Fix:

* regularization

---

## 12.4 Outliers

Outliers can change boundary.

Fix:

* robust scaling
* regularization

---

# 13) Logistic Regression vs Linear Regression

| Aspect       | Linear Regression | Logistic Regression |
| ------------ | ----------------- | ------------------- |
| Output       | continuous        | probability         |
| Task         | regression        | classification      |
| Loss         | MSE               | log loss            |
| Output range | (-∞,∞)            | (0,1)               |

---

# 14) Logistic Regression vs Naive Bayes

### Logistic regression

* discriminative model
* models P(y|x)

### Naive Bayes

* generative model
* models P(x|y) and P(y)

NB works well for:

* text classification when independence approx holds

LR works well when:

* you want interpretable + strong baseline

---

# 15) Logistic Regression vs SVM

### LR

* probabilistic output
* interpretable

### SVM

* margin maximization
* very strong for high-dim
* not probabilistic (unless calibrated)

---

# 16) Metrics to evaluate Logistic Regression

For classification:

* precision, recall, f1
* ROC-AUC
* PR-AUC (for imbalance)

Probability calibration:

* Brier score
* calibration curve

---

# 17) Threshold tuning (important in interviews)

Logistic regression gives probability.

But classification depends on threshold.

Example:
Fraud:

* you need high recall
* set threshold low (0.2)

Spam:

* you need high precision
* set threshold high (0.8)

---

# 18) How to explain it in 3 lines (interview)

> Logistic regression is a linear classifier that models the log-odds of the target class as a linear function of input features. It uses the sigmoid (or softmax for multiclass) to convert scores into probabilities and is trained by minimizing cross-entropy loss with optional L1/L2 regularization.

---

# 19) Interview Tricky Questions + Answers

## Q1) Why is it called regression if it’s classification?

Because it performs regression on **log-odds**, a continuous variable.

---

## Q2) What happens if you increase regularization strength?

* weights shrink
* model becomes simpler
* bias ↑ variance ↓

In sklearn:

* C ↓ means regularization ↑

---

## Q3) Why cross entropy loss is better than MSE for LR?

* convex optimization
* better gradients
* penalizes confident wrong predictions

---

## Q4) Is logistic regression linear?

✅ Yes, the decision boundary is linear in feature space.

But with feature engineering (polynomials, interactions):

* can model nonlinear patterns

---

## Q5) How to handle multiclass?

* OvR
* Multinomial softmax regression

---

## Q6) Multiclass vs multilabel difference?

* multiclass: exactly one class
* multilabel: multiple classes can be 1

---

## Q7) What is logit?

`logit(p) = log(p/(1-p))`

---

## Q8) What is perfect separation?

When classes can be separated completely.
Weights diverge.
Regularization fixes.

---

## Q9) Explain odds ratio.

`exp(w)` indicates factor change in odds.

---

## Q10) Why scaling is required?

Because regularization penalizes weights.
If features have different scale, penalty becomes unfair.

---

## Q11) When logistic regression fails?

* non-linear boundary required
* strong interaction patterns
* high dimensional sparse text sometimes better with NB

---

## Q12) How to interpret coefficients for categorical variables?

After one-hot encoding:

* coefficient is effect relative to reference category.

---

# 20) What you should remember (cheat sheet)

* Logistic regression is a **classification model**
* Predicts probability using sigmoid
* Uses **log loss**, not MSE
* Regularization: L1/L2/ElasticNet
* sklearn uses `C = 1/λ`
* Multiclass: OvR or softmax regression
* Multilabel: one model per label

---

