# Exam Score Prediction

## Project overview

This project focuses on predicting students’ exam scores using regression models.  
The notebook covers the full workflow of a basic supervised machine learning project: data loading, preprocessing, feature encoding, train-test split, model training, and performance comparison.

The goal was to evaluate whether regularized linear models such as **Ridge**, **Lasso**, and **ElasticNet** improve prediction quality compared with standard **Linear Regression**.

---

## Dataset

The dataset was downloaded directly from Kaggle using `kagglehub`.

It contains student-related features describing:
- demographic information,
- study habits,
- sleep-related factors,
- access to resources,
- course type,
- study method,
- and exam conditions.

The target variable is:

- **`exam_score`** – the final exam score predicted by the models.

---

## Project workflow

### 1. Data loading
The dataset was downloaded from Kaggle and loaded into a pandas DataFrame.

### 2. Initial inspection
Basic inspection steps included:
- previewing the first rows of the dataset,
- checking missing values,
- reviewing data types.

### 3. Data preprocessing
Several preprocessing steps were applied before modeling:

- **Ordinal encoding** for ordered categories:
  - `facility_rating`
  - `exam_difficulty`
  - `sleep_quality`

- **Binary encoding** for:
  - `internet_access`

- **One-hot encoding** for nominal categorical features:
  - `course`
  - `study_method`
  - `gender`

- Conversion of boolean dummy columns into integer format

- Removal of the `student_id` column, as it is only an identifier and has no predictive value

### 4. Feature and target definition
The feature matrix `X` includes all predictors, while `y` contains the target variable `exam_score`.

### 5. Train-test split
The data was split into training and testing subsets using an 80/20 ratio.

### 6. Feature scaling
Feature scaling was introduced using `StandardScaler`, which is especially important for regularized regression models.

### 7. Model training
The following models were trained and evaluated:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **ElasticNet Regression**

For Ridge, Lasso, and ElasticNet, multiple values of `alpha` were tested to analyze the effect of regularization strength.

---

## Evaluation metrics

Model performance was evaluated using:

- **MAE (Mean Absolute Error)** – average absolute prediction error
- **RMSE (Root Mean Squared Error)** – penalizes larger prediction errors more strongly
- **R² (Coefficient of Determination)** – proportion of target variance explained by the model

These metrics were used to compare the baseline model with the regularized approaches.

---

## Results summary

The baseline **Linear Regression** model already achieved strong results:

- **MAE:** 7.862
- **RMSE:** 9.771
- **R²:** 0.7331

This means the model explains about **73.3% of the variance** in exam scores and makes an average prediction error of about **7.9 points**.

### Ridge Regression
Ridge performed almost identically to Linear Regression across small and moderate values of `alpha`.  
This suggests that the dataset did not require stronger L2 regularization.

### Lasso Regression
Lasso achieved the **best overall result** with:

- **alpha = 0.01**
- **MAE:** 7.862
- **RMSE:** 9.769
- **R²:** 0.7332

However, larger values of `alpha` caused performance to drop quickly, indicating that stronger L1 regularization removed too much useful information.

### ElasticNet Regression
ElasticNet performed reasonably well for very small `alpha`, but its results worsened as regularization strength increased.  
It did not outperform the simpler models in this project.

---

## Best-performing model

The best result was obtained by:

- **Lasso Regression (`alpha = 0.01`)**

Although the improvement over Linear Regression was very small, it was still the strongest result among all tested models.

---

## Final conclusions

The project shows that:

- the relationship between the features and exam score is largely **linear**,
- **Linear Regression** already provides a strong baseline,
- **Ridge** does not bring a meaningful improvement,
- **Lasso** performs best only with very weak regularization,
- stronger regularization generally reduces model quality in this dataset.

Overall, the available features provide a strong predictive signal, and simple linear approaches are sufficient to model exam score effectively.

---

## Technologies used

- Python
- pandas
- scikit-learn
- kagglehub

---

## Repository contents

- `exam_score.ipynb` – main notebook containing the full analysis, preprocessing, model training, and evaluation

---

## Possible next steps

Potential improvements for future work:
- testing tree-based models such as Random Forest or XGBoost,
- performing hyperparameter tuning with cross-validation,
- exploring feature importance in more detail,
- adding visualizations for residuals and prediction errors.
