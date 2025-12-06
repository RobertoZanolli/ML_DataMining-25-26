> 
> **Subject**: ML slides pack - clustering with and without kmeans by Claudio Sartori
>
> **Course:** Artificial Intelligence - LM
> 
>**Department**: DISI (Department of Computer Science and Engineering) - University of Bologna, Italy
>
> **Author**: Roberto Zanolli
> 
> **TIPS**
> - w.r.t means "with respect to"
> 


```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
```

## Dealing with missing values in scikit-learn

Scikit-learn provides various methods to handle missing values:
- Removing rows or columns with missing values
- Simple imputation with mean, median, or mode
- Iterative imputation
- Using indicators for missing values

### Removing rows or columns with missing values

The simplest way to handle missing data is to remove rows or columns containing NaN values.

Use `pandas.dropna()` for DataFrame preprocessing:

```python
data = pd.DataFrame({
    'A': [1, 2, None],
    'B': [4, None, 6]
})

# Drop columns with missing values
cleaned_columns = data.dropna(axis=1)

# Drop rows with missing values (default)
cleaned_data = data.dropna()
```

**pandas.dropna() notes:**
- Default is to drop an entire row/column if there is at least one null
- Use the `thresh` integer parameter to indicate the minimum number of nulls that trigger dropping
- Dropping a row with nulls drops a piece of evidence
- Dropping a column with nulls drops an entire feature
- Simple dropping is meaningful when the amount of rows/columns with nulls is small
- Dropping can be substituted or integrated with imputation

### SimpleImputer

Replaces missing values with a constant, mean, median, most frequent, or pre-defined value.

```python
from sklearn.impute import SimpleImputer

# Example dataset with missing values
X = pd.DataFrame({
    'A': [1, 4, 7], 
    'B': ['x', None, 'z'],
    'C': [None, 8, 9]
})

numeric_feat = ['A', 'C']
categ_feat = ['B']

# Impute missing values with 'mean', 'median', 'most_frequent', 'constant'
imputer_num = SimpleImputer(strategy='mean')
X_imputed[numeric_feat] = imputer_num.fit_transform(X[numeric_feat])

imputer_categ = SimpleImputer(strategy='constant', fill_value='unknown')
X_imputed[categ_feat] = imputer_categ.fit_transform(X[categ_feat])
```

### Integration with ColumnTransformer

Combine imputation strategies for different column types:

**Guidelines:**
- Numeric features: 
  - If distribution is not skewed, fill nulls with mean
  - If distribution is skewed, fill nulls with median
- Discrete features: fill nulls with a pre-defined constant (e.g., "unknown")

```python
imputer = ColumnTransformer(
    transformers=[
        ("fill-w-median", SimpleImputer(strategy="median"), ["A", "B", "C"]),
        ("fill-w-mean", SimpleImputer(strategy="mean"), ["D", "E"]),
        ("fill-w-constant", SimpleImputer(strategy="constant", fill_value="unknown"), ["F"])
    ]
)

X_imputed = imputer.fit_transform(X)
```