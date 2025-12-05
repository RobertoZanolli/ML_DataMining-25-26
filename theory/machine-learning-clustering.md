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


# Clustering

## What is clustering

**Given:** A set of $N$ objects $x_i$, each described by $D$ values $x_{id}$

**Task:** Find a natural partitioning into $K$ clusters and possibly identify noise objects

**Result:** A clustering scheme - a function mapping each data object to $\{1...K\}$ (or to noise)

**Desired cluster property:**
- Objects in same cluster are similar
- Look for clustering scheme that **maximizes intra-cluster similarity**
## Formal definition for clustering function

>Find a function $clust()$ from $X$ to $\{1..K\}$ such that:
>
>1. $\forall x_1, x_2 \in X$, $clust(x_1) = clust(x_2)$ when $x_1$ and $x_2$ are similar
>2. $\forall x_1, x_2 \in X$, $clust(x_1) \neq clust(x_2)$ when $x_1$ and $x_2$ are not similar
>

### How do we choose a measure to determine clusters

![[centroid_1.png]]
### Centroid

**Definition:** A point whose coordinates are the average coordinates of all points in the cluster

**Calculation:** For each cluster $k$ and dimension $d$, the $d$-th coordinate of the centroid is:

$$centroid_d^k = \frac{1}{|x_i : clust(x_i) = k|} \sum_{x_i : clust(x_i) = k} x_{id}$$