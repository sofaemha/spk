## **Analytic Hierarchy Process (AHP)**
AHP is a structured technique for organizing and analyzing complex decisions, based on mathematics and psychology. The process involves pairwise comparisons of criteria and alternatives, followed by deriving weights and consistency ratios to evaluate the decision.

### Step-by-Step Explanation

---

### 1. **Normalize Pairwise Comparison Matrix**

```python
def normalize_matrix(matrix):
    """Normalize the matrix by dividing each element by the column sum."""
    col_sums = np.sum(matrix, axis=0)
    return matrix / col_sums
```

#### Explanation:

* **Normalization** is a critical step in AHP to ensure that all criteria are measured on a similar scale. In this function:

  * **Column sums** are calculated (`np.sum(matrix, axis=0)`), which gives the sum of each column of the pairwise comparison matrix.
  * Then, each element in the matrix is divided by its respective column sum. This scales the matrix such that the sum of each column is 1.

#### Formula for Normalization:

For a matrix element $(a_{ij})$ in the pairwise comparison matrix, the normalized element $(n_{ij})$ is:

$$
n_{ij}=\frac{a_{ij}}{\sum_{i=1}^{n}a_{ij}}
$$

Where:

* $a_{ij}$ is the element in row $i$, column $j$
* $\sum_{i=1}^{n}a_{ij}$ is the sum of all elements in column $j$

---

### 2. **Calculate Priority Vector**

```python
def calculate_priority_vector(normalized_matrix):
    """Calculate the priority vector by averaging the rows."""
    return np.mean(normalized_matrix, axis=1)
```

#### Explanation:

* After normalization, the **priority vector** (also known as the weight vector) is calculated by averaging each row of the normalized matrix.
* This vector represents the relative importance of each criterion or alternative based on the pairwise comparisons.

#### Formula for Priority Vector:

For each row $i$, the priority vector $w_i$ is calculated as:

$$
w_i = \frac{1}{n} \sum_{j=1}^{n} n_{ij}
$$

Where:

* $n_{ij}$ is the normalized matrix element
* $n$ is the number of rows (or criteria/alternatives)

---

### 3. **Calculate Lambda Max (Maximum Eigenvalue)**

```python
def calculate_lambda_max(matrix, priority_vector):
    """Calculate the maximum eigenvalue (lambda_max)."""
    return np.mean(np.dot(matrix, priority_vector) / priority_vector)
```

#### Explanation:

* The **maximum eigenvalue** $\lambda_{\text{max}}$ is an indicator of how consistent the pairwise comparison matrix is.
* It is calculated by multiplying the original pairwise comparison matrix by the priority vector and then dividing each element of the resulting vector by the corresponding priority vector element.

#### Formula for Lambda Max:

$$
\lambda_{\text{max}} = \frac{1}{n} \sum_{i=1}^{n} \frac{(A \cdot w)_i}{w_i}
$$

Where:

* $A$ is the original pairwise comparison matrix
* $w$ is the priority vector
* $(A \cdot w)_i$ is the element of the vector resulting from multiplying $A$ by $w$

---

### 4. **Calculate Consistency Index (CI)**

```python
def calculate_consistency_index(lambda_max, n):
    """Calculate the Consistency Index (CI)."""
    return (lambda_max - n) / (n - 1)
```

#### Explanation:

* The **Consistency Index (CI)** measures the consistency of the pairwise comparison matrix. If the matrix is perfectly consistent, $\lambda_{\text{max}}$ will equal $n$, and CI will be zero.
* A higher CI indicates more inconsistency in the matrix.

#### Formula for CI:

$$
CI = \frac{\lambda_{\text{max}} - n}{n - 1}
$$

Where:

* $\lambda_{\text{max}}$ is the maximum eigenvalue
* $n$ is the number of criteria or alternatives

---

### 5. **Get Random Index (RI)**

```python
def get_random_index(n):
    """Get the Random Index (RI) for a given matrix size n."""
    ri_values = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    return ri_values.get(n, 1.49)  # Default to 1.49 for n > 10
```

#### Explanation:

* The **Random Index (RI)** is a predefined value based on the number of criteria (or alternatives) used in the pairwise comparisons. It is a statistical value derived from random pairwise comparison matrices.
* This value is used to assess whether the calculated consistency ratio is due to random chance or if the matrix is truly inconsistent.

---

### 6. **Calculate Consistency Ratio (CR)**

```python
def calculate_consistency_ratio(ci, ri):
    """Calculate the Consistency Ratio (CR)."""
    return ci / ri
```

#### Explanation:

* The **Consistency Ratio (CR)** is the ratio of the Consistency Index (CI) to the Random Index (RI). A CR value less than 0.1 is generally considered acceptable.
* This ratio helps in determining whether the pairwise comparison matrix is sufficiently consistent. If CR is greater than 0.1, the matrix might be inconsistent and needs revision.

#### Formula for CR:

$$
CR = \frac{CI}{RI}
$$

Where:

* $CI$ is the Consistency Index
* $RI$ is the Random Index

---

### 7. **AHP Calculation Function**

```python
def ahp(pairwise_matrix):
    """Perform AHP calculation on a pairwise comparison matrix."""
    n = pairwise_matrix.shape[0]

    # Normalize the matrix
    normalized = normalize_matrix(pairwise_matrix)

    # Calculate priority vector (weights)
    weights = calculate_priority_vector(normalized)

    # Calculate lambda_max
    lambda_max = calculate_lambda_max(pairwise_matrix, weights)

    # Calculate CI
    ci = calculate_consistency_index(lambda_max, n)

    # Get RI
    ri = get_random_index(n)

    # Calculate CR
    cr = calculate_consistency_ratio(ci, ri)

    return weights, cr
```

#### Explanation:

* This function brings together all the previous steps to perform the full AHP calculation.
* It takes a pairwise comparison matrix as input, normalizes it, calculates the priority vector, computes $\lambda_{\text{max}}$, calculates the Consistency Index (CI), and determines the Consistency Ratio (CR).

---

##### 1. **Variable: `pairwise_matrix`**

###### Description:

* This is the **input matrix** that represents the pairwise comparisons among the criteria or alternatives. Each element $a_{ij}$ in this matrix indicates the relative importance of criterion $i$ compared to criterion $j$ using the Saaty scale (1 to 9).

  * If $a_{ij} = 3$ it means criterion $i$ is 3 times more important than criterion $j$.
  * If $a_{ij} = 1/3$ criterion $i$ is 3 times less important than criterion $j$.
  * The diagonal elements are always 1 since each criterion is equally important compared to itself.

###### Formula:

* The pairwise comparison matrix $A$ looks something like this:

$$
A=\begin{pmatrix}
1 & a_{12} & a_{13} & \dots & a_{1n} \\
\frac{1}{a_{12}} & 1 & a_{23} & \dots & a_{2n}\\
\frac{1}{a_{13}} & \frac{1}{a_{23}} & 1 & \dots&a_{3n}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
\frac{1}{a_{1n}} & \frac{1}{a_{2n}} & \frac{1}{a_{3n}} & \dots & 1 \\
\end{pmatrix}
$$

Where:

* $a_{ij}$ is the pairwise comparison value.

---

##### 2. **Variable: `n`**

###### Description:

* **`n`** is the number of criteria (or alternatives) being compared. It is derived from the shape of the `pairwise_matrix`, which is an $n \times n$ matrix.
* This variable is important for calculating the consistency index (CI) and the random index (RI).

#### Formula:

* $n = \text{number of rows or columns in the pairwise matrix}$.

For example, if there are 4 criteria, $n = 4$.

---

##### 3. **Variable: `normalized`**

###### Description:

* The **`normalized` matrix** is the result of normalizing the input `pairwise_matrix`. Normalization ensures that all columns in the matrix sum to 1, which is a standard procedure in AHP to make the pairwise comparison matrix comparable across different criteria.

#### Formula:

* Each element $n_{ij}$ of the normalized matrix is calculated as:

$$
n_{ij} = \frac{a_{ij}}{\sum_{i=1}^{n} a_{ij}}
$$

Where:

* $a_{ij}$ is the original pairwise comparison matrix element.
* $\sum_{i=1}^{n} a_{ij}$ is the sum of the $j$-th column in the original matrix.

---

##### 4. **Variable: `weights`**

###### Description:

* **`weights`** represents the **priority vector** or **weights** for each criterion (or alternative) calculated from the normalized matrix. This is the core of the AHP method since these weights represent the relative importance of each criterion or alternative.

###### Formula:

* The **priority vector** is calculated by taking the average of each row in the normalized matrix:

$$
w_i = \frac{1}{n} \sum_{j=1}^{n} n_{ij}
$$

Where:

* $w_i$ is the weight of criterion $i$.
* $n_{ij}$ is the normalized value for the $i$-th row and $j$-th column.

The priority vector gives the relative importance of each criterion (or alternative) in the decision-making process.

---

##### 5. **Variable: `lambda_max`**

###### Description:

* **`lambda_max`** is the **maximum eigenvalue** of the pairwise comparison matrix. It is used to measure the consistency of the decision-making process. A perfectly consistent matrix will have $\lambda_{\text{max}} = n$.

###### Formula:

* The value of $\lambda_{\text{max}}$ is computed by performing a matrix-vector multiplication and dividing each element by the corresponding weight, then averaging the results:

$$
\lambda_{\text{max}} = \frac{1}{n} \sum_{i=1}^{n} \frac{(A \cdot w)_i}{w_i}
$$

Where:

* $A$ is the original pairwise comparison matrix.
* $w$ is the priority vector (weights).
* $(A \cdot w)_i$ is the element of the vector resulting from multiplying matrix $A$ by the weight vector $w$.

This step is an approximation to the **principal eigenvalue** of the matrix, which reflects how well the matrix can be modeled by a consistent set of judgments.

---

### 6. **Variable: `ci` (Consistency Index)**

#### Description:

* **`ci`** is the **Consistency Index**, which quantifies the consistency of the pairwise comparison matrix. If the matrix is perfectly consistent, $\lambda_{\text{max}} = n$ and the CI will be zero. A higher CI indicates more inconsistency in the matrix.

#### Formula:

* The **Consistency Index (CI)** is calculated as:

$$
CI = \frac{\lambda_{\text{max}} - n}{n - 1}
$$

Where:

* $\lambda_{\text{max}}$ is the maximum eigenvalue.
* $n$ is the number of criteria or alternatives.

A CI close to zero suggests a highly consistent matrix, whereas a CI significantly greater than zero indicates potential inconsistency.

---

##### 7. **Variable: `ri` (Random Index)**

###### Description:

* **`ri`** is the **Random Index** associated with the number of criteria or alternatives $n$ This value is derived from statistical analyses of random matrices and is used as a benchmark for assessing the consistency of the matrix.

###### Formula:

$$
\begin{matrix}
Index \\
RI \ Value
\end{matrix}
\quad
\begin{matrix}
\quad
1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 \\
\quad
0.00 & 0.00 & 0.58 & 0.90 & 1.12 & 1.24 & 1.32 & 1.41 & 1.45 & 1.49
\end{matrix}
$$

* **RI values** are predefined based on the size of the matrix. For example, for 3 criteria, the RI is 0.58; for 4 criteria, the RI is 0.90, etc. The function uses a dictionary to return the appropriate RI for a given $n$

```python
ri_values = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
```

---

##### 8. **Variable: `cr` (Consistency Ratio)**

###### Description:

* **`cr`** is the **Consistency Ratio**, which is calculated by dividing the Consistency Index (CI) by the Random Index (RI). This ratio helps assess the quality of the pairwise comparisons. A CR greater than 0.1 indicates that the matrix may be inconsistent, suggesting the need for review.

###### Formula:

* The **Consistency Ratio (CR)** is calculated as:

$$
CR = \frac{CI}{RI}
$$

Where:

* $CI$ is the Consistency Index.
* $RI$ is the Random Index.

---

### Return:

* The function returns two values:

  * **`weights`**: The priority vector (the relative importance of the criteria or alternatives).
  * **`cr`**: The consistency ratio, which tells you whether the pairwise comparisons are consistent.

---

### Overall Process Summary:

1. **Define Criteria and Alternatives**: Enter the number of criteria and alternatives for decision making.
2. **Pairwise Comparisons**: Compare the criteria and alternatives in pairs using the Saaty scale (1 to 9).
3. **Normalize**: Normalize the pairwise comparison matrix.
4. **Calculate Weights**: Calculate the priority vector (weights) by averaging the rows.
5. **Check Consistency**: Calculate $\lambda_{\text{max}}$ CI, and CR to assess the consistency of the matrix.
6. **Rank Alternatives**: Calculate overall scores for each alternative based on the criteria weights and alternative weights, and then rank them.
