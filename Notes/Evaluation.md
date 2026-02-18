# R² (R-squared) — Easy Intuition + Calculation

## What is R²?

**R² tells how well your regression model explains the target values compared to a baseline.**

✅ **Baseline** = “always predict the mean of `y`”.

So R² answers:

> **How much better is my model than simply predicting the average every time?**

---

## Easy intuition

* **R² = 1** → perfect predictions
* **R² = 0** → model is no better than predicting the mean
* **R² < 0** → model is worse than predicting the mean

---

## Formula


$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$


Where:

### 1) Total variation in data


$SS_{tot} = \sum (y_i - \bar{y})^2$

Meaning: how much the target varies around its mean.

### 2) Error left after model prediction


$SS_{res} = \sum (y_i - \hat{y}_i)^2$

Meaning: how much the model is still wrong.

---

## Calculation example (super simple)

Actual values:
[
y = [2,4,6]
]
Mean:
[
\bar{y} = 4
]

Model predictions:
[
\hat{y} = [3,5,5]
]

### Step 1: Compute (SS_{tot})

[
(2-4)^2 + (4-4)^2 + (6-4)^2 = 4 + 0 + 4 = 8
]

### Step 2: Compute (SS_{res})

[
(2-3)^2 + (4-5)^2 + (6-5)^2 = 1 + 1 + 1 = 3
]

### Step 3: Compute R²

[
R^2 = 1 - (3/8) = 0.625
]

✅ Meaning: model explains **~62.5%** of the variation in the target.

---

## Interview one-liner ✅

**R² is the proportion of variance in the target explained by a regression model compared to predicting the mean.**

### Why we use Adjusted R² (short)

* Regular **R² always increases** when you add more features—even useless ones.
* This can give a **false sense of model improvement**.
* **Adjusted R² penalizes extra predictors**, so it only increases when a new feature genuinely improves the model.
* Hence, it is better for **comparing models with different numbers of features**.

---

### Formula

$
R^2_{\text{adj}} = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}
$

where:

* (R^2) = coefficient of determination
* (n) = number of observations
* (p) = number of predictors (features)

---

### Key property

* If a new feature is useless → Adjusted R² **decreases**
* If a new feature is useful → Adjusted R² **increases**

This makes it a more reliable metric for model selection.
