#### TIPS
- w.r.t means "with respect to"

# Data
## Data types

Data can be categorized along several dimensions:

- **Quantitative**: Numerical data (continuous or discrete)
- **Qualitative**: Categorical data (nominal or ordinal)
- **Structured**: Organized in predefined formats (tables, databases)
- **Unstructured**: No predefined organization (text, images, audio)

![](images/machine-learning/data_types_1.png)

"description" and "descriptive statistics" sums while descending (Ordinal has everything in Nominal, plus its own and so on).

**Discrete domains**
- allow a finite number of values (or infinitely countable)
- codes, counts, ...
- special case: binary attributes
- special case: identifier

**Continuous domains**
- floating point variables
- nominals and ordinals are discrete, possibly binary
- intervals and ratio are continuous
- counts are discrete and ratio

## Data quality 
***"Data are never perfect"***
###  Main problems
- **Missing values**: Incomplete data records (lot of nulls)
- **Inconsistent data**: Contradictory information across sources
- **Duplicated entries**: Redundant data instances
- **Wrong values**: Incorrect or inaccurate data
- **Outliers**: Small amounts of data that significantly differ from the rest (can be due to error during measurement, data entry errors, or genuine variability)

### Outliers
- Represent anomalies or errors in the dataset
- Some machine learning techniques are more **robust with respect to errors** than others
- Detection and handling crucial for model performance

**Key point**:  $\text{Better data quality} \Rightarrow \text{Better results}$

**Critical principle**: $Garbage-in–Garbage-out$
- Poor quality input data inevitably leads to poor model performance
- **No algorithm can compensate for fundamentally flawed data**

--- 
## More on data types

### Interval data vs ratio data
Interval **does not preserve relative value upon scale change**

![](images/machine-learning/interval_data_1.png)

### Data transformations
Not all transformations are allowed according to the data type
![](images/machine-learning/transformations_1.png)


## Asymmetric attributes
- ***We only care if it is not null (only interested if the value is present)***

## **General characteristics of data sets**

### **Dimensionality**
*   The diﬀerence between having a small or a large (hundreds, thousands, ...) number  of attribute is also qualitative
### **Sparsity**
*   Sparsity means that there are many zeros or nulls
*   Some databases stores the nulls as zeros or placeholders value (still irrelevant and noisy)

### **Resolution**
*   Greatly influences the results
*   The analysis of too detailed data can be aﬀected by noise
*   The analysis of too general data can hide interesting patterns

### **Record data**

- **Tables**
	*  e.g. relational databases (also dataframes)
- **Transaction**
	*  a row is composed by: TID + set of Items
	* e.g: { id: 1, bought: [hairdryer, facemask...]}
- **Data matrix**
	*  numeric values of the same type
	*  a row is a point in a vector space
- **Sparse data matrix**
	*  asymmetric values of the same type
- Graph data
	- e.g: all XML-like files like html pages

## Noise
- Original data are **modified** or there is a **mix of interesting and uninteresting** data

## Outliers
**Outlier** = data whose characteristics are considerably different from most of the data in the dataset. 

![](images/machine-learning/outliers_1.png)

#### **IQR - InterQuartile Range**

$Q1$: first quartile, $Q3$: third quartile,

$IQR = Q3 - Q1$
$Lower$  $boundary$ $= Q1 - IQR \times 1.5$
$Upper$ $boundary$ $= Q3 + IQR \times 1.5$

*Consider outlier the values out of the whiskers*
The upper whisker will extend to last datum less than $Q3 + 1.5 \times IQR$

---
#  Classification
**Unsupervised Classification**

- The unsupervised mining techniques which can be in some way related to classification are usually known in literature with names different from classification

- **Classification** = supervised classification

## Soybean example

**The Data Set**

- The data set $X$ contains $N$ individuals described by $D$ attribute values each
- We have $Y$ vector which, for each individual $x$ contains the class value $y(x)$ (labels)
- The class allows a finite set of different values (e.g. the diseases), say $C$
- The class values are provided by experts: the supervisors
- We want to learn how to guess the value of the $y(x)$ for individuals which have not been examined by the experts

## Definition of classification model

- **An algorithm which, given an individual for which the class is not known, computes the class**
- The algorithm is parametrized to optimize results for the specific problem

**Development Process:**
1. Choose the learning algorithm
2. Let the algorithm learn its parametrization
3. Assess the quality of the classification model

**Usage:**
- The classification model is used by a run-time classification algorithm with the developed parametrization

### Formal definition

**Decision function:**
$$M(x, \theta) = y(x)_{pred}$$
**Where:**
- $x$: data element with unknown class label $y(x)$
- $\theta$: set of parameter values for the decision function
- $y(x)_{pred}$: predicted class

**Learning Process:**
Given classifier $M(.,.)$, dataset $X$, and supervised labels $Y$, determine $\theta$ to **minimize** prediction error.

##### Example
For simple domain we can use a straight line as decison function.
![](images/machine-learning/classification_example_1.png)
We can see that it makes mistakes.
It's normal, every classifier makes mistakes (even with the best fit for parameters)

## **Vapnik-Chervonenkis Dimension**

*Given a dataset with $N$ elements there are $2^N$ possible different learning problems.*
If a model $M(.,.)$ is able to shatter **all** the possible learning problems with $N$ elements, we say that it has **Vapnik-Chervonenkis** Dimension equal to $N$.

**The straight line has VC dimension 3.**

![](images/machine-learning/classification_workflow.png)
*example of a classification workflow*

### Different types of classificators
- CRISP
	- the classifier assigns to each individual one label
- Probabilistic
	- the classifier assigns a probability for each of the possible labels


## Decision Trees
##### Main strength
Good compromise: decent performance, fast to train and execute, easy to understand

### **Decision Tree Structure**
A run-time classifier structured as a decision tree is a **tree-shaped set of tests**.
The decision tree has:
- **Inner** nodes
- **Leaf** nodes

![](images/machine-learning/decision_tree_algorithm_1.png)

### Learning a decision tree – Model generation

Given a **set $X$ of elements** for which the class is known, grow a decision tree as follows:

- If all the elements belong to class $c$ or $X$ is small: generate a leaf node with label $c$

- Otherwise:
  1. Choose a test based on a single attribute with two or more outcomes
  2. Make this test the root of a tree with one branch for each of the outcomes of the test
  3. Partition $X$ into subsets corresponding to the outcomes
  4. Apply recursively the procedure to the subsets

**The main question is: what feature are more significant and should be used first in the decision process?**

#### We explore the data to find out.
##### Boxplots 
- Good for outliers

```python
plt.figure(figsize=(15,10))

sns.boxplot(data = df);


```
![](images/machine-learning/classification_boxplot_1.png)

![](images/machine-learning/classification_boxplot_2.png)

###### Histograms

```python
# df.hist(figsize=(10.0,9.0)) #using directly pandas
plt.hist(df.quality, color="g", )
```

![](images/machine-learning/classification_hist_1.png)

##### Pairplots

```python
sns.pairplot(df, hue="quality", diag_kind="kde")
```

![](images/machine-learning/classification_pairplot_1.png)

#### Supervised Learning Goals

- Design an algorithm able to forecast the values of an attribute given the values of other attributes
- In our case, guess the class given the other values

**Problem:** if i have a lot of features it is not easy to look at plots
**Answer:** we need a solution based on math 

### Entropy and information gain (IG)

**From information theory we get:**
Given a source $X$ with $V$ possible values, with probability distribution:

$$P(v_1) = p_1, P(v_2) = p_2, ..., P(v_V) = p_V$$

The best coding allows the transmission with an average number of bits given by:

$$H(X) = -\sum_{j} p_j \log_2(p_j)$$

$H(X)$ is the entropy of the information source $X$

#### Meaning

- **High entropy** means that the probabilities are mostly similar
- The histogram would be flat
- **Low entropy** means that some symbols have much higher probability
- The histogram would have peaks
- Higher number of allowed symbols (i.e. of distinct values in an attribute) gives higher entropy

![](images/machine-learning/entropy_1.png)

**If the source is BINARY (probabilities for the outcomes are respectively $p$ and $(1-p)$ when p is 0 or 1 the entropy goes to 0.** -> If i have 2 classes $A$ and $B$ if $P(A) = 0$ then $P(B) = 1$
#### Entropy in Classification

- In classification, low entropy of the class labels of a dataset means that there is low diversity in the labels (i.e. the dataset has high purity, there is a majority class)
- We look for criteria that allow to split a dataset into subsets with higher purity
- With criteria we mean logical formulas to be used as decision function to partition the set elements into the subsets

**After split:**

Splitting the dataset in two parts according to a threshold on a numeric attribute the entropy changes, and becomes the weighted sum of the entropies of the two parts.

The weights are the relative sizes of the two parts.

Let $d \in D$ be a real-valued attribute, let $t$ be a value of the domain of $d$, let $c$ be the class attribute.

We define the entropy of $c$ w.r.t. $d$ with threshold $t$ as:

$$H(c|d : t) = H(c|d < t) \cdot P(d < t) + H(c|d \geq t) \cdot P(d \geq t)$$

#### Information Gain for binary split

It is the reduction of the entropy of a target class obtained with a split of the dataset based on a threshold for a given attribute.
We define:
$$IG(c|d : t) = H(c) - H(c|d : t)$$

It is the information gain provided when we know if, for an individual, $d$ exceeds the threshold $t$ in order to forecast the class value.

We define:
$$IG(c|d) = \max_t IG(c|d : t)$$

### Decision Tree Construction

A decision tree is a tree-structured plan generating a sequence of tests on the known attributes (predicting attributes) to predict the values of an unknown attribute.

**Construction Process:**
- Test the attribute which guarantees the ***maximum IG*** for the class attribute in the current data set $X$
- Partition $X$ according to the test outcomes
- Recursion on the partitioned data

### Train/Test Split

- **Training set**: used to learn the model
- **Test set**: used to evaluate the learned model on fresh data

**Procedure:**
- The split is done randomly
- Assumption: the parts have similar characteristics
- The proportion of the split is decided by the experimenter
- Common solutions: 80-20, 67-33, 50-50

**Example:**
- For a 50-50 split of the Iris dataset
- For this specific split, entropies for the class column in training and test turns out to be both 1.58
```python
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Perform 50-50 train/test split with random_state = 10
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.5, 
    random_state=10,
    stratify=y  # Optional: preserves class distribution
)
```

#### Buildind the decision tree (case with binary split)

### Decision Tree Recursion Step

**Process:**
- Choose the attribute giving the highest IG
- Partition the dataset according to the chosen attribute
- Choose as class label of each partition the majority

**Recursion step:**
- Build a new tree starting from each subset where the minority is non-empty

**Observation:**
The weighted sum of the entropy of the descendant nodes is always smaller than the entropy in the ancestor node, even if one of the descendant has higher entropy (its the sum that counts).

**Termination conditions:**
- Most of the leaves are pure, recursion impossible
- One of the leaves is not pure, but no more tests are able to give positive information gain, recursion impossible
- It is labelled with the majority class, or, in case of tie, with one of the non-empty classes


![](images/machine-learning/decision_tree_algorithm_2.png)
#### Training Set Error

- If we execute the generated decision tree on the training set itself (hiding the class to predict)
- Count the number of discordances between the true and the predicted class
- This is the training set error -> it isn't 0

**why?**

- **The limits of decision trees in general:**
	- A decision tree based on tests on attribute values can fail
- **Insufficient information in the predicting attributes**

##### Meaning
- **Error on the same data used to train the model**
- Represents the **lower limit** of expected error on new data
- We need an upper limit or more significant value for real performance

**It is better to study test set error** -> it tells us about behaviour with unseen data
### Overfitting definition

- Overfitting happens when the learning is affected by noise
- When a learning algorithm is affected by noise, the performance on the test set is (much) worse than that on the training set
#### Formal definition

A decision tree is a hypothesis of the relationship between the predictor attributes and the class.

**Definitions:**
- $h$ = hypothesis
- $error_{train}(h)$ = error of the hypothesis on the training set
- $error_X(h)$ = error of the hypothesis on the entire dataset

**Overfitting Condition:**
$h$ overfits the training set if there is an alternative hypothesis $h_1$ such that:
$$error_{train}(h) < error_{train}(h_1)$$
$$error_X(h) > error_X(h_1)$$
#### Causes for overfitting

1. **Presence of noise**
   - Bad values in predicting attributes or class labels
   - Model influenced by wrong or unusual training data

2. **Lack of representative instances**
   - Some real-world situations underrepresented in training set

A good model has low **generalization error** - works well on examples different from training data.

#### Pruning
Pruning is the way to simplify the model when you are using a decision tree.
It is the tweak of cutting some branches of the decision tree that develop too specifically over less significant features.
### Model Hyperparameters

- Every model generation algorithm can be adjusted by setting specific ***hyperparameters***
- Each model has its own hyperparameters -> decision tree != SVM
- One of the hyperparameters of decision tree generation is the **maximum tree depth**
```python
from sklearn.tree import DecisionTreeClassifier

# Create decision tree with max_depth hyperparameter
model = DecisionTreeClassifier(max_depth=3)
model.fit(x_train, y_train)
```

### Purity Measures for Node Splitting

**We need a measure for the purity of a node** - a node with two classes in the same proportion has low purity. -> Only one class = max purity

**Three common measures:**
- Entropy 
- Gini Index
- Misclassification Error -> (optional)
#### Gini Index Definition

Consider a node $p$ with $C_p$ classes.

**For class $j$:**
- Frequency $f_{p,j}$
- Frequency of other classes $1-f_{p,j}$
- Probability of wrong assignment: $f_{p,j} \cdot (1-f_{p,j})$

**The Gini Index** is the **total probability of wrong classification**:
$$\sum f_{p,j} \cdot (1-f_{p,j}) = \sum f_{p,j} - \sum f_{p,j}^2 = 1 - \sum f_{p,j}^2$$

**Range:**
- Maximum value: when records are uniformly distributed over all classes: $1 - 1/C_p$
- Minimum value: when all records belong to the same class: $0$
#### Splitting Based on Gini Index

When a node $p$ is split into $ds$ descendants $p_1,...,p_{ds}$:

Let $N_{p,i}$ and $N_p$ be the number of records in the $i$-th descendant node and in the root, respectively.

**We choose the split giving the maximum reduction of the Gini Index:**
$$GINI_{split} = GINI_p - \sum \frac{N_{p,i}}{N_p} GINI(p_i)$$
### Conclusion

**Computational Complexity:**
- Overall cost: $O(DN \log N)$
- Run-time classification: $O(h)$ where $h$ is tree height

**Key Properties:**
- Non-parametric approach (no distribution assumptions)
- NP-complete to find optimal DT, heuristics find sub-optimal solutions
- Robust to noise and redundant attributes
- Pruning strategy has high impact on final result

**Practical Advantages:**
- Easy to understand, implement, and use
- Best starting point for supervised learning
- Overfitting controlled by maximum tree depth
- Handles both continuous and discrete predictor attributes