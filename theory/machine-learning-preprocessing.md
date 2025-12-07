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

# Data pre-processing
## Dealing with missing values in scikit-learn

```python
#imports were omitted in slides but this should be the right ones
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
```

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

## Type conversion

```python
 OneHotEncoder, OrdinalEncoder, Binarizer
```
## Why do we need type conversion?
Many algorithms require numeric features:
- Categorical features must be transformed into numeric
- Ordinal features must be transformed into numeric, and the order must be preserved
Classification requires a target with nominal values:
- A numerical target can be discretised
Discovery of association rules require boolean features:
- A numerical feature can be discretised and transformed into a series of boolean features

### The scikit-learn solution for type conversions
### Binarization of discrete attributes
Attribute $d$ allowing $V$ values → $V$ binary attributes.
Example for "Color" feature:

| Color | Color-Red | Color-Blue | Color-Green | Color-Orange | Color-Yellow |
|-------|-----------|------------|-------------|--------------|--------------|
| Red   | 1         | 0          | 0           | 0            | 0            |
| Blue  | 0         | 1          | 0           | 0            | 0            |
| Green | 0         | 0          | 1           | 0            | 0            |
| Orange| 0         | 0          | 0           | 1            | 0            |
| Yellow| 0         | 0          | 0           | 0            | 1            |

### Nominal to numeric: One-Hot-Encoding
- A feature with $V$ unique values is substituted by $V$ binary features, each corresponding to one of the unique values
- If object $x$ has value $v$ in feature $d$, then the binary feature corresponding to $v$ has True for $x$, all other binary features have value False
- True and False are represented as 1 and 0, therefore can be processed by procedures working only on numeric data
```python
from sklearn.preprocessing import OneHotEncoder
```

### Ordinal to numeric
- The ordered sequence is transformed into consecutive integers
- By default, the lexicographic order is assumed
- The user can specify the proper order of the sequence
Example: `awful, poor, ok, good, great` → `0, 1, 2, 3, 4`

```python
from sklearn.preprocessing import OrdinalEncoder
```



### Numeric to binary with threshold
- Not greater than the threshold becomes zero
- Greater than the threshold becomes one

```python
from sklearn.preprocessing import Binarizer
```
### Discretization/Reduction of the number of distinct values
Some algorithms work better with discrete instead of continuous data:
- A small number of distinct values can let patterns emerge more clearly
- A small number of distinct values lets algorithms be less influenced by noise and random effects
**Discretization:**
- Continuous → Discrete (using thresholds, many options)
- Binarization → single threshold
- Discrete with many values → Discrete with less values (guided by domain knowledge)

![](theory/images/Screenshot%202025-12-07%20alle%2000.44.08.png)

### Numeric to k values
The numbers are discretised into a sequence of integers 0 to $k-1$
Several strategies are available:
- `'uniform'`
- `'quantile'`
- `'kmeans'`
```python
from sklearn.preprocessing import KBinsDiscretizer
```
### Sampling

For both preliminary investigation and final data analysis:
**Statistician perspective:** obtaining the entire data set could be impossible or too expensive  
**Data processing perspective:** processing the entire data set could be too expensive or time consuming

1. Using a sample will work almost as well as using the entire data sets, if the sample is representative
2. A sample is representative if it has approximately the same property (of interest) as the original set of data

### Types of sampling
1. **Simple random:** a single random choice of an object with given probability distribution
2. **With replacement:** repetition of independent extractions of type 1
3. **Without replacement:** repetition of extractions, extracted element is removed from the population
4. **Stratified:** used to split the data set into subsets with homogeneous characteristics, the representativity is guaranteed inside each subset (typically requested in cross-validation)

### Sample size
Statistics provides techniques to assess:
- Optimal sample size
- Sample significativity
Tradeoff between data reduction and precision

### Sampling with/without replacement
- They are nearly equivalent if sample size is a small fraction of the data set size
- With replacement, in a small population a small subset could be underestimated
- Sampling with replacement is:
  - Much easier to implement
  - Much easier to be interpreted from a statistical point of view
  - Extractions are statistically independent
  
![](theory/images/Screenshot%202025-12-07%20alle%2000.48.47.png)

As we can see in the example decreasing the sample size means loss of evidence.

#### Probability of sampling at least one element for each class (with replacement)
This is not related to the size of the dataset
Example: 10 classes (A, B, C, D, E, F, G, H, I, J)

This aspect becomes relevant, for example, in a supervised dataset with a high number of different values of the target.

If the number of data elements is not big enough, it can be difficult to guarantee a stratified partitioning in train/test split or in cross-validation split.

**Example:**  
$N = 1000$, $C = 10$, test-set-size $= 300$, cross-validation-folds $= 10$

The probability of folds without adequate representation of some classes becomes quite high.

When designing the training processes it is necessary to consider those aspects. In the example, one could use only 3 folds in cross-validation.
## Feature creation
Feature creation is a crucial step in data mining - new features can capture more efficiently data characteristics. It involves transforming raw data into meaningful features that can improve predictive models.

**Types of feature creation:**
- **Feature extraction:** pixel picture with a face → eye distance, ...
- **Mapping to a new space:** e.g. signal to frequencies with Fourier transform
- **New features:** e.g. volume and weight to density

### Examples of feature creation in finance
##### Moving averages
Calculate the average closing price of a stock over a specific time window (e.g., 10 days, 50 days).
##### Volatility measures
Compute metrics such as standard deviation or average true range to capture the level of price fluctuations.
##### Relative strength index (RSI)
Derive a momentum oscillator indicating overbought or oversold conditions in the market.
##### Liquidity ratios
Calculate ratios like bid-ask spread or trading volume to assess market liquidity.
##### Fundamental analysis indicators
Include financial metrics such as earnings per share (EPS), price-to-earnings (P/E) ratio, or debt-to-equity ratio.
##### Market sentiment analysis
Utilize sentiment scores from news articles, social media, or analyst reports to gauge market sentiment.
##### Technical analysis patterns
Identify chart patterns such as head and shoulders, double tops, or flags to predict future price movements.

### Feature creation example: closing prices of a stock
```python
data = {
    'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
    'Close': [100, 102, 98, 105, 101]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Feature 1: Moving Average (2 days)
df['MA_2'] = df['Close'].rolling(window=2).mean()

# Feature 2: Daily Price Change
df['Price_Change'] = df['Close'].diff()

# Feature 3: Daily Percentage Change
df['Pct_Change'] = df['Close'].pct_change()

# Feature 4: Relative Strength Index (RSI)
df['RSI'] = 100 - (100 / (1 + (df['Close'].pct_change()).rolling(window=2)\
    .apply(lambda x: x[x > 0].mean() / x[x < 0].mean())))
```

### Feature creation: closing prices of a stock 
**Functions used:**
- `rolling()` aggregates a number of consecutive rows specified with window size
- `diff()` computes the difference of a feature in consecutive rows
- `pct_change()` computes the percentage of change of a feature in consecutive rows
- The aggregations introduce a number of nulls in the initial rows

#### Examples of feature creation in a housing dataset
##### Total area
Calculate the sum of sizes of all rooms in the house.
##### Price per square foot
Divide the price of the house by its total area.
##### Age of the house
Calculate the age of each house by subtracting the year it was built from the current year.
##### Neighborhood median income
Include data on the median income of households in each neighborhood.
##### Distance to city center
Measure the distance of each house from the city center.
##### Renovation status
Create a binary feature indicating whether the house has been recently renovated.
##### School district rating
Include data on the quality of school districts in which the houses are located.
##### Presence of amenities
Create a categorical feature indicating the presence of amenities such as a swimming pool, garden, garage, etc.

##  Data transformation
**Why?**
The features may have different scales - this can alter the results of many learning techniques. Some machine learning algorithms are sensitive to feature scaling while others are virtually invariant to it. There can also be outliers.

### Gradient descent
Machine learning algorithms that use gradient descent as an optimization technique require data to be scaled (e.g., linear regression, logistic regression, neural networks, etc.).
![](theory/images/Screenshot%202025-12-07%20alle%2000.59.39.png)
**Why scaling matters for gradient descent:**
- The presence of feature value $X$ in the formula will affect the step size of the gradient descent
- The difference in ranges of features will cause different step sizes for each feature
- Similar ranges of the various features ensure that gradient descent moves smoothly toward the minima
- Steps for gradient descent are updated at the same rate for all features

### Feature transformation
Map the entire set of values to a new set according to a function:
$x^k$, $\log(x)$, $e^x$, $|x|$

In general, these transformations change the distribution of values.

### Standardization
$$x \rightarrow \frac{x - \mu}{\sigma}$$

**Key points:**
- If original values have a Gaussian distribution, transformed values will have standard Gaussian distribution ($\mu = 0$, $\sigma = 1$)
- This is translation and shrinking/stretching - no change in distribution shape
- Centers data around zero with unit variance
> 
> ### MinMax scaling (a.k.a. Rescaling)
> The domains are mapped to standard ranges:
> 
> **Range 0 to 1:**
> $x \rightarrow \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$
> 
> **Range -1 to 1:**
> $x \rightarrow \frac{x - \frac{x_{\text{max}} + x_{\text{min}}}{2}}{\frac{x_{\text{max}} - x_{\text{min}}}{2}}$
> 
> **Key points:**
> - This is translation and shrinking/stretching - no change in distribution shape
> - Scales data to fit within a specific range
> - Useful when you need bounded values
<div style="display: flex; gap: 10px;">
  <img src="theory/images/Screenshot%202025-12-07%20alle%2001.00.23.png" width="300">
  <img src="theory/images/Screenshot%202025-12-07%20alle%2001.00.30.png" width="300">
</div>

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method=’box-cox’)
X = pd.DataFrame(pt.fit_transform(X0), columns = X0.columns)
```



### Distance-based algorithms
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
```

KNN, K-Means, SVM, and similar algorithms use distances between points to determine their similarity.

### Example: effect of scaling on distances 

**Original data:**

| Student | CGPA | Salary |
|---------|------|--------|
| A       | 3    | 60     |
| B       | 3    | 40     |
| C       | 4    | 40     |
| D       | 4.5  | 50     |
| E       | 4.2  | 52     |

**Scaled data:**

| Student | CGPA    | Salary   |
|---------|---------|----------|
| A       | -1.18431| 1.520013 |
| B       | -1.18431| -1.100699|
| C       | 0.41612 | -1.100699|
| D       | 1.21635 | 0.209657 |
| E       | 0.736212| 0.471728 |

**Distances before scaling:**
- $\text{distance}(A,B) = \sqrt{(40-60)^2 + (3-3)^2} = 20$
- $\text{distance}(B,C) = \sqrt{(40-40)^2 + (4-3)^2} = 1$

**Distances after scaling:**
- $\text{distance}(A_s,B_s) = \sqrt{(-1.1+1.5)^2 + (-1.18+1.18)^2} = 2.6$
- $\text{distance}(B_s,C_s) = \sqrt{(-1.1-1.1)^2 + (0.41+1.18)^2} = 1.59$

Before scaling, the distances seemed very different due to the big numeric difference in the Salary attribute. After scaling, they become comparable.

### Feature rescaling

### Range-based scaling and standardization
Both operate on single features:

**Range-based scaling** stretches/shrinks and translates the range according to the feature's range (with variants):
- Good when data are not Gaussian or we make no distribution assumptions
- MinMaxScaler remaps to [0,1]

**Standardization** subtracts the mean and divides by the standard deviation:
- Resulting distribution has mean zero and unitary standard deviation
- Good when the distribution is Gaussian
- `sklearn.preprocessing.StandardScaler`

### Range-based scalers in Scikit-Learn
**MinMaxScaler** – remaps the feature to $[0,1]$

**RobustScaler** – centering and scaling statistics based on percentiles:
- Not influenced by a few very large marginal outliers
- Resulting range of transformed feature values is larger than MinMaxScaler and StandardScaler

### Normalization
The term "normalization" has different meanings:
- Frequently refers to MinMaxScaler
- In Scikit-learn, `Normalizer` normalizes each data row to unit norm

**RECAP**
![](theory/images/Screenshot%202025-12-07%20alle%2001.31.19.png)

### Workflow for feature transformation
1. Transform the features as required for both train and test data  
2. Fit and optimize the model(s)  
3. Test  
4. Possibly, use the original data to plot relevant views (e.g., to plot cluster assignments)

## Imbalanced data in classification
The performance on the minority class(es) has little impact on standard performance measures. The optimized model could be less effective on minority class(es).

**Solutions:**
- Some estimators allow weighting classes
- Some performance measures account for the contribution of minority class(es)

### Cost-sensitive learning
Several classifiers have the parameter `class_weight`:
- Changes the cost function to account for class imbalance
- Equivalent to oversampling the minority class (repeating random examples) to produce a balanced training set
```python
from sklearn.utils import class_weight
```

### Undersampling
Obtains a balanced training set by randomly reducing the number of examples of the majority class.
**Caution:** Part of the knowledge embedded in the training set is dropped out.
### Oversampling with SMOTE
Synthetic Minority Oversampling Technique – a type of data augmentation for the minority class $c_{min}$:

**Algorithm:**
1. Choose from the training set a random example $x_r$ of class $c_{min}$
2. Find the $k$ nearest neighbors of $x_r$ whose class is $c_{min}$
3. Choose randomly one neighbor $x_{rn}$ from above
4. Create new data element: $m = r \cdot \frac{x_r + x_{rn}}{2}$ (chosen randomly from segment connecting $x_r$ and $x_{rn}$ in feature space)

### Workflow for undersampling/oversampling
1. Resample the training set (apply SMOTE/undersampling)
2. Fit and optimize the estimator
3. Test the fitted estimator on the test set (untouched)

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import Lasso, Ridge
```

## Feature selection

> *"Sometimes less is better" (by Rohan Rao)*

**Benefits of feature selection:**
- Enables machine learning algorithms to train faster
- Reduces model complexity and makes it easier to interpret
- Improves accuracy if the right subset is chosen
- Reduces overfitting

**Note:** A specific selection action may obtain only one of the above effects.

### Supervised or unsupervised feature selection?

**Unsupervised:**
- Many methods available (e.g., for clustering)
- Feature transformation techniques like PCA can reduce the number of features

**Supervised (considering relationship between attributes and class):**
- **Filter methods** (Scheme-Independent Selection)
- **Wrapper methods** (Scheme-Dependent Selection)
- **Embedded methods** (built-in feature selection, e.g., Lasso and Ridge regression)

### Problems with attributes
##### Irrelevant attributes
They can alter results of some mining algorithms, particularly with insufficient control of overfitting.
##### Redundant attributes
Some attributes can be strongly related to other useful attributes. Mining algorithms (e.g., Naive Bayes) are strongly influenced by strong correlations between attributes.
##### Confounding attributes
Some attributes can be misleading with hidden effects on the outcome variable.
**Example:** In a study on weight gain considering physical exercise, age, and sex - sex can be confounding if the ages of males and females have very different ranges in the available data.
##### Mixed effect attributes
One attribute could be strongly related to the class in 65% of cases and random in the other cases.

### The curse of dimensionality
When dimensionality is very high, the occupation of the space becomes very sparse. Discrimination based on distance becomes ineffective.

**Experiment:** Random generation of 500 points - plot the relative difference between the maximum and minimum distance between pairs of points. As dimensions increase, distances become more similar.
### Dimensionality reduction
**Purposes:**
- Avoid the curse of dimensionality
- Noise reduction
- Reduce time and memory complexity of mining algorithms
- Visualization

**Techniques:**
- Principal component analysis (PCA)
- Singular values decomposition (SVD)
- Supervised techniques
- Non-linear techniques

```python
from sklearn.decomposition import PCA
```

### Aggregation
Combining two or more attributes (or objects) into a single attribute (or object).

**Purpose:**
- Data reduction
- Change of scale
- More stable data
- Reduce number of attributes or objects
**Examples:** Cities aggregated into regions, states, countries, etc. Aggregated data tends to have less variability.

### Feature subset selection
#### Redundant attributes
Duplicate most information contained in other attributes (e.g., price and VAT amount).
#### Irrelevant attributes
Do not contain useful information for analysis (e.g., SSN is not relevant to predict wealth).
#### Feature selection approaches:
1. **Brute force:** Try all possible feature subsets as input to data mining algorithm and measure effectiveness
2. **Embedded approach:** Feature selection occurs naturally as part of data mining algorithm (e.g., decision trees)
3. **Filter approach:** Features selected before data mining algorithm runs
4. **Wrapper approaches:** Data mining algorithm chooses best set of attributes (like brute force but without exhaustive search)

#### Filter methods (Scheme-Independent Selection)
Assessment based on general characteristics of data. Select subset of attributes independently from mining model.
**Examples:**
- Build decision tree and consider attributes near the root, then use selected attributes for another classifier
- Select subset of attributes that individually correlate to class but have little intercorrelation
### Some filter methods
#### Pearson's Correlation
A measure for quantifying linear dependence between two continuous variables $X$ and $Y$; value from -1 to +1.
#### LDA (Linear Discriminant Analysis)
Used to find a linear combination of features that characterizes or separates two or more classes.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

#### ANOVA (Analysis of Variance)
Similar to LDA except it's operated using one or more categorical independent features and one continuous dependent feature.
#### Chi-Square
A statistical test applied to groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.
```python
from sklearn.feature_selection import chi2
```

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score
```

### Wrapper methods
Try to use a subset of features and train a model using them. Based on inferences from the previous model, we decide to add or remove features from the subset.
The problem is essentially reduced to a search problem:

**Feature selection process:**
Set of all features → Generate a subset → Learning algorithm → Performance evaluation
#### One wrapper method: Search the Attribute Space
Example: Weather dataset

- Search greedily the space
- For each subset test the performance of the chosen classification model
- Computation intensive

### Difference between Filter and Wrapper methods

**Filter methods:**
- Measure relevance of features by their correlation with dependent variable
- Much faster as they do not involve training models
- Use statistical methods for evaluation of feature subsets
- Might fail to find the best subset of features
- Less prone to overfitting

**Wrapper methods:**
- Measure usefulness of feature subset by actually training a model on it
- Computationally expensive
- Use cross-validation for evaluation
- Can provide the best subset of features
- More prone to overfitting

```python
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
```

### Correlation of quantitative data (Pearson's)
> Measure of the linear relationship between a pair of attributes.
> 
> **Calculation:**
> 1. Standardize the values
> 2. For two attributes $p$ and $q$, consider as vectors the ordered lists of values over all data records
> 3. Compute the dot product of the standardized vectors
> 
> $p = [p_1,...,p_N]$ standardize $\rightarrow p'$  
> $q = [q_1,...,q_N]$ standardize $\rightarrow q'$  
> $\text{corr}(p,q) = p' \cdot q'$
> 
> There's an alternative definition based on covariance and variances.
> 
> ### Correlation - Discussion
> - **Independent variables** → correlation is zero (but inverse is not valid in general)
> - **Correlation zero** → absence of linear relationship between variables
> - **Positive values** imply positive linear relationship
> 
### Role of Correlation in feature selection
#### Identifying redundant features
Features highly correlated with each other contain overlapping information. Retain one feature from such groups to reduce dimensionality.
#### Identifying relevant features
> High correlation with the target variable helps identify features with high predictive power.
> 
>  **Caveat:** Low Pearson's correlation between a feature and the target can sometimes hide a **non-linear correlation**. Therefore, using low Pearson's correlation alone for feature filtering can be dangerous - it only detects linear relationships and may miss important non-linear patterns.

### Computational complexity of correlation-based feature selection
**Calculating correlation**
For $D$ features and $N$ samples, computing correlation $r(f_i, y)$ for all features is:
$O(N \cdot D)$

**Pairwise feature correlation**
Requires computing correlations for $\binom{D}{2}$ feature pairs:
$O(D^2 \cdot N)$

**Overall complexity**
Linear in the number of samples, quadratic in the number of features.

```python
# Complexity demonstration: calculating pairwise correlation
# This operation grows quadratically with the number of columns (features)
correlation_matrix = df.corr() 
```

### Advantages and limitations
- Simple and interpretable method for feature selection
- Reduces overfitting by removing irrelevant/redundant features
- **Limitation:** Only considers linear relationships between variables

### Summary on correlation
Correlation is a simple and effective tool for feature selection. Helps reduce dimensionality by removing redundant and irrelevant features. Should be complemented with other techniques to handle nonlinear relationships.
```python
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
```

## Dimensionality reduction
Instead of ignoring subset of attributes, map dataset into new space with fewer attributes.
#### PCA (Principal Component Analysis)
Find new ordered set of dimensions that better captures data variability:
- First captures most variability
- Second orthogonal to first, captures most remaining variability
- And so on

Fraction of variance captured by each new variable is measured. Small number of new variables can capture most variability.

**Mathematical details:**
- Covariance matrix (positive semidefinite)
- Eigenvalue analysis: eigenvalues positive, sorted decreasing
- Eigenvectors sorted by eigenvalue order

#### PCA computational complexity overview
Involves:
- Data centering
- Covariance matrix computation
- Eigenvalue decomposition or SVD

Depends on $N$ (samples), $D$ (features), method used.

> #### Key steps in PCA
> **Data centering:** Subtract mean of each feature. Complexity: $O(ND)$
> **Covariance matrix computation:**
> - If $D \leq N$: $D \times D$ matrix, $O(ND^2)$
> - If $D > N$: $N \times N$ matrix (dual PCA), $O(N^2D)$
> **Eigenvalue decomposition or SVD:**
> - $D \times D$ covariance: $O(D^3)$
> - SVD of $N \times D$ matrix: $O(N^2D)$ if $N \leq D$; $O(ND^2)$ if $D \leq N$
> 
#### Overall PCA complexity
**Small feature set ($D \ll N$):** $O(D \cdot N^2 + D^3)$
**Large feature set ($D \approx N$):** $O(D^2N + N^3)$

#### Optimizations for PCA
**Truncated SVD:** Computes only top $k$ singular values/vectors. Complexity: $O(kND)$, useful when $k \ll \min(N,D)$

**Sparse data:** Sparse matrix methods exploit sparsity.

#### Practical implications
PCA expensive for large $N,D$ with full SVD.

**Alternatives:** Random projections, iterative methods (power iteration).
**Use cases:**
- Small datasets: Full PCA feasible
- Large datasets: Approximation methods recommended

### MDS (Multi-Dimensional Scaling)
Presentation technique starting from distances among dataset elements. Fits projection into $m$-dimensional space preserving distances. Versions for non-metric/metric spaces.

#### Introduction to MDS
Visualize high-dimensional data in low-dimensional space (2D/3D).

**Goals:**
- Preserve pairwise distances/dissimilarities
- Geometric representation for interpretation
**Applications:** Exploratory analysis, genetics, psychology, marketing.

#### MDS problem setup
Given $n$ objects, dissimilarity matrix $\Delta = (\delta_{ij})$ where $\delta_{ij}$ is distance/dissimilarity between $i,j$.
**Objective:** Find points $x_1, \dots, x_n \in \mathbb{R}^p$ ($p \ll n$) s.t. $d_{ij} \approx \delta_{ij}$ for all $i,j$, where $d_{ij} = \|x_i - x_j\|$.

#### Mathematical formulation
**Types:**
- Classical MDS: Assumes Euclidean distances
- Non-metric MDS: Preserves rank order of distances

**Cost function:** Minimize stress $S(X) = \sum_{i<j} w_{ij} (d_{ij} - \delta_{ij})^2$
- $w_{ij}$ weights (often 1)
- Optimize over coordinates $X$

#### Classical MDS and eigenvalue decomposition
> **Steps:**
> 1. Distance matrix $\Delta$
> 2. Similarity matrix: $B = -\frac{1}{2} H \Delta^2 H$, $H = I - \frac{1}{n} \mathbf{1}\mathbf{1}^T$ centers data
> 3. Eigenvalue decomposition: $B = Q \Lambda Q^T$
> 4. Low-dim embedding: $X = Q_p \Lambda_p^{1/2}$
> 
#### MDS computational complexity
- Pairwise distances: $O(n^2)$
- Eigenvalue decomposition: $O(n^3)$ for dense matrices

**Scalability:** Expensive for large $n$. Use Landmark MDS approximations.

#### MDS advantages and limitations
**Advantages:**
- Interpretable low-dimensional representation
- Works well for approximately Euclidean distances
**Limitations:**
- Sensitive to noise/outliers in distance matrix
- Computationally intensive for large datasets

```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, f_classif
```

### General structure
Main methods (there are more, somewhat different for various estimators):
- `.fit`: Learn empirical variances from X
- `.fit_transform`: Fit to data, then transform it
- `.transform`: Reduce X to selected features

Main argument: X, the dataset → X is the set of features, y is output class

### Baseline estimator: VarianceThreshold
**Removes features with low variance (unsupervised).**

**Example:** Dataset with binary attributes. Eliminate features with proportion $p \geq 0.8$ (80-20 or more).

Bernoullian variance: $p(1-p)$. Threshold: $0.8 \times (1-0.8) = 0.16$

### Univariate, supervised feature selection
Select best set of features based on univariate statistical tests.

Considers original features and target. For each feature, returns score and p-value.

**Methods:**
- **SelectKBest**: Removes all but k highest scoring features
```python
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif

# Load dataset
X, y = load_iris(return_X_y=True)

# Select the 2 best features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

print("Selected feature indices:", selector.get_support(indices=True))
print("Transformed shape:", X_new.shape)
```

- **SelectPercentile**: Removes all but user-specified highest scoring percentage of features

```python
from sklearn.feature_selection import mutual_info_classif, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier  # example estimator
```

### Score functions
Used by feature selector to evaluate how much a feature is useful to predict the target.
- `mutual_info_classif`: Computes Mutual Information (generalization of Information Gain)
- `f_classif`: Fisher test with ANOVA (analysis of variance)

### A wrapper method: Recursive Feature Elimination (RFE)
Feature ranking with recursive feature elimination.

**Process:**
- Uses external estimator to assign weights to features
- Considers smaller and smaller sets of features
- Trains estimator on initial set, obtains feature importance
- Prunes least important features
- Stops when desired number of features reached
**Purpose:** Identify best subset of features contributing most to model's prediction.

### Steps in Recursive Feature Elimination
1. **Train model on all features:** Evaluate feature importance (e.g., coefficients or scores)
2. **Rank features by importance:** Lower importance considered for removal
3. **Remove least important feature(s):** Retrain model with remaining features
4. **Repeat until desired number of features reached**

![](theory/images/Screenshot%202025-12-07%20alle%2002.04.14.png)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Initialize the model
model = LogisticRegression(max_iter=200)

# Initialize the RFE selector
selector = RFE(model, n_features_to_select=2)

# Fit the RFE model
selector.fit(X, y)

# Get the selected features
print("Selected features:", selector.support_)
print("Ranking of features:", selector.ranking_)
```

### Advantages and limitations of RFE
**Advantages:**
- Provides clear method for selecting most important features
- Can be used with any machine learning model
- Reduces overfitting by eliminating irrelevant features
**Limitations:**
- Computationally expensive, especially for large datasets
- May not be effective when feature importance is not clearly defined
- Requires large number of iterations for feature selection