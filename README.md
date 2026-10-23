# Diabetes Machine Learning Project Using Azure Open Datasets

## 1. Project Overview

This project demonstrates how to build two machine learning models using a public diabetes dataset stored in Azure Blob Storage.

The project uses Python, Apache Spark, Pandas, Scikit-learn, and MLflow to prepare the data, train machine learning models, and track experiments.

Two machine learning approaches are implemented:

1. **Linear Regression** – predicts a numerical target value (`Y`).
2. **Logistic Regression** – classifies records into two categories using a derived target column (`Risk`).

## 2. Technologies Used

- Python
- Apache Spark (PySpark)
- Azure Blob Storage
- Pandas
- Scikit-learn
- MLflow

## 3. Dataset Access

The dataset is stored in Azure Blob Storage.

```python
blob_account_name = "azureopendatastorage"
blob_container_name = "mlsamples"
blob_relative_path = "diabetes"
blob_sas_token = r""
```

The code defines the storage account, container, dataset path, and an empty SAS token because the example assumes anonymous access.

A WASBS path is created to access the dataset.

```python
wasbs_path = f"wasbs://%s@%s.blob.core.windows.net/%s" % (
    blob_container_name,
    blob_account_name,
    blob_relative_path
)
```

Spark is configured to access the storage location, and the Parquet dataset is loaded into a Spark DataFrame.

```python
spark.conf.set(
    "fs.azure.sas.%s.%s.blob.core.windows.net" %
    (blob_container_name, blob_account_name),
    blob_sas_token
)

df = spark.read.parquet(wasbs_path)
display(df)
```

**Purpose:** Read the dataset directly from Azure Blob Storage without manually downloading the data.

## 4. Data Preparation

The Spark DataFrame is converted into a Pandas DataFrame.

```python
df = df.toPandas()
df.head()
```

This allows the dataset to be processed using Pandas and Scikit-learn.

A copy of the dataset is created, and a new column called `Risk` is added.

```python
df_Clean = df.copy()
df_Clean["Risk"] = (df["Y"] > 211.5).astype(int)
```

The `Risk` column is created using the following rule:

- `Risk = 1` when `Y` is greater than 211.5.
- `Risk = 0` when `Y` is less than or equal to 211.5.

This converts the numerical target into a binary classification target.

The following command displays descriptive statistics for the dataset.

```python
df_Clean.describe()
```

## 5. Model 1: Linear Regression

### Objective

Predict the numerical target `Y` using the available input features.

The model uses the following ten features:

`AGE`, `SEX`, `BMI`, `BP`, `S1`, `S2`, `S3`, `S4`, `S5`, `S6`

### Train-Test Split

The input features are stored in `X`, and the target variable `Y` is stored in `y`.

```python
X = df_Clean[
    ["AGE", "SEX", "BMI", "BP", "S1",
     "S2", "S3", "S4", "S5", "S6"]
].values

y = df_Clean["Y"].values
```

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)
```

- 70% of the data is used for training.
- 30% of the data is reserved for testing.
- `random_state=0` makes the split reproducible.

### Model Training and MLflow Tracking

```python
import mlflow
from sklearn.linear_model import LinearRegression

mlflow.set_experiment("diabetes-regression")

with mlflow.start_run():
    mlflow.autolog()

    model = LinearRegression()
    model.fit(X_train, y_train)
```

The Linear Regression model learns relationships between the input features and the numerical target `Y`.

MLflow creates an experiment named `diabetes-regression` and automatically logs supported training parameters, metrics, and model information.

## 6. Model 2: Logistic Regression

### Objective

Classify records into two categories based on the derived `Risk` column.

Unlike Linear Regression, this model predicts a class rather than a continuous numerical value.

The same ten input features are used, but the target variable is changed to `Risk`.

```python
X = df_Clean[
    ["AGE", "SEX", "BMI", "BP", "S1",
     "S2", "S3", "S4", "S5", "S6"]
].values

y = df_Clean["Risk"].values
```

The data is split into training and testing sets using the same 70:30 ratio.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)
```

### Model Training and MLflow Tracking

```python
import mlflow
from sklearn.linear_model import LogisticRegression

mlflow.set_experiment("diabetes-classification")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = LogisticRegression(
        C=1/0.1,
        solver="liblinear"
    ).fit(X_train, y_train)
```

The Logistic Regression model learns to distinguish between `Risk = 0` and `Risk = 1`.

- `C=10` controls the strength of regularization; a higher value means weaker regularization.
- `solver="liblinear"` specifies the optimization algorithm used to train the model.
- `mlflow.sklearn.autolog()` enables automatic logging of supported Scikit-learn model details and metrics.

## 7. MLflow Experiments

The project creates two separate MLflow experiments.

| Experiment Name | Algorithm | Target |
|---|---|---|
| `diabetes-regression` | Linear Regression | Numerical `Y` |
| `diabetes-classification` | Logistic Regression | Binary `Risk` |

These experiments help organize and review the training runs for each model.

## 8. Key Learning Outcomes

This project demonstrates how to:

- Access a public dataset from Azure Blob Storage.
- Read Parquet data using Apache Spark.
- Convert Spark DataFrames into Pandas DataFrames.
- Create a new target column using feature engineering.
- Split data into training and testing sets.
- Train regression and classification models.
- Track machine learning experiments using MLflow.

## 9. Future Improvements

The project can be enhanced by:

- Evaluating Linear Regression using MAE, RMSE, and R².
- Evaluating Logistic Regression using accuracy, precision, recall, F1-score, and a confusion matrix.
- Comparing model performance using MLflow.
- Applying cross-validation and checking class balance.
- Testing different model parameters.

## 10. Disclaimer

This project is intended for educational purposes only. The `Risk` column is an experimental label derived from the numerical target `Y`. It is not a clinically validated measure of diabetes risk and should not be used for medical diagnosis.

## 11. References

- Azure Open Datasets: https://azure.microsoft.com/en-us/products/open-datasets/
- Scikit-learn: https://scikit-learn.org/
- MLflow: https://mlflow.org/
- Apache Spark: https://spark.apache.org/docs/latest/
