# Analytical Gradient Derivations via Chain Rule

**Author:** Tughan Bulut Kurtulus (Student ID: 73454)  
**Architecture:** 2-Hidden-Layer Multi-Layer Perceptron (MLP)  
**Task:** Hand-derived backpropagation gradients for numerical verification  

---

## 1. Network Topology & Forward Formulations

Let batch size be $N$, input dimensionality $D_{\text{in}} = 64$, hidden width $W \in \{16, 64\}$, and class count $C = 10$.

* **Layer 1 (Hidden Affine + ReLU):**
  $$Z_1 = X W_1 + \mathbf{1}_N b_1^T, \quad Z_1 \in \mathbb{R}^{N \times W}, \quad W_1 \in \mathbb{R}^{64 \times W}, \quad b_1 \in \mathbb{R}^W$$
  $$A_1 = \text{ReLU}(Z_1) = \max(0, Z_1), \quad A_1 \in \mathbb{R}^{N \times W}$$

* **Layer 2 (Hidden Affine + ReLU):**
  $$Z_2 = A_1 W_2 + \mathbf{1}_N b_2^T, \quad Z_2 \in \mathbb{R}^{N \times W}, \quad W_2 \in \mathbb{R}^{W \times W}, \quad b_2 \in \mathbb{R}^W$$
  $$A_2 = \text{ReLU}(Z_2) = \max(0, Z_2), \quad A_2 \in \mathbb{R}^{N \times W}$$

* **Layer 3 (Output Affine + Softmax):**
  $$Z_3 = A_2 W_3 + \mathbf{1}_N b_3^T, \quad Z_3 \in \mathbb{R}^{N \times 10}, \quad W_3 \in \mathbb{R}^{W \times 10}, \quad b_3 \in \mathbb{R}^{10}$$
  $$P_{ik} = \frac{e^{Z_{3,ik}}}{\sum_{j=1}^{10} e^{Z_{3,ij}}}, \quad P \in \mathbb{R}^{N \times 10}$$

* **Cross-Entropy Objective Function:**
  $$L = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{10} Y_{ik} \ln(P_{ik})$$

---

## 2. Backward Pass Gradient Derivations

### Step 1: Derivative of Loss w.r.t Output Pre-Activation ($Z_3$)
For sample $i$ and class $k$:
$$\frac{\partial L_i}{\partial Z_{3,ik}} = \sum_{j=1}^{10} \frac{\partial L_i}{\partial P_{ij}} \frac{\partial P_{ij}}{\partial Z_{3,ik}}$$
Given:
$$\frac{\partial L_i}{\partial P_{ij}} = -\frac{Y_{ij}}{P_{ij}}$$
And the Softmax Jacobian:
$$\frac{\partial P_{ij}}{\partial Z_{3,ik}} = \begin{cases} P_{ik}(1 - P_{ik}) & \text{if } j = k \\ -P_{ij} P_{ik} & \text{if } j \neq k \end{cases}$$
Substituting into the summation:
$$\frac{\partial L_i}{\partial Z_{3,ik}} = -\frac{Y_{ik}}{P_{ik}} P_{ik}(1 - P_{ik}) - \sum_{j \neq k} \frac{Y_{ij}}{P_{ij}} (-P_{ij} P_{ik})$$
$$= -Y_{ik}(1 - P_{ik}) + P_{ik} \sum_{j \neq k} Y_{ij} = -Y_{ik} + P_{ik} \sum_{j=1}^{10} Y_{ij}$$
Since $Y_i$ is a one-hot distribution ($\sum_j Y_{ij} = 1$):
$$\frac{\partial L_i}{\partial Z_{3,ik}} = P_{ik} - Y_{ik}$$
Averaging over batch size $N$:
$$\delta_3 = \frac{\partial L}{\partial Z_3} = \frac{1}{N} (P - Y) \in \mathbb{R}^{N \times 10}$$

---

### Step 2: Gradients of Output Parameters ($W_3, b_3$) and Activation ($A_2$)
Applying the chain rule matrix product:
$$\frac{\partial L}{\partial W_3} = A_2^T \delta_3 \in \mathbb{R}^{W \times 10}$$
$$\frac{\partial L}{\partial b_3} = \sum_{i=1}^N \delta_{3,i} = \mathbf{1}_N^T \delta_3 \in \mathbb{R}^{10}$$
$$\frac{\partial L}{\partial A_2} = \delta_3 W_3^T \in \mathbb{R}^{N \times W}$$

---

### Step 3: Backpropagation through ReLU Layer 2
The derivative of ReLU is the Heaviside step function:
$$\frac{\partial A_2}{\partial Z_2} = \mathbf{1}(Z_2 > 0)$$
By Hadamard (element-wise) product:
$$\delta_2 = \frac{\partial L}{\partial Z_2} = \left(\delta_3 W_3^T\right) \odot \mathbf{1}(Z_2 > 0) \in \mathbb{R}^{N \times W}$$
Parameter gradients:
$$\frac{\partial L}{\partial W_2} = A_1^T \delta_2 \in \mathbb{R}^{W \times W}$$
$$\frac{\partial L}{\partial b_2} = \mathbf{1}_N^T \delta_2 \in \mathbb{R}^W$$
$$\frac{\partial L}{\partial A_1} = \delta_2 W_2^T \in \mathbb{R}^{N \times W}$$

---

### Step 4: Backpropagation through ReLU Layer 1 & Input Layer
$$\delta_1 = \frac{\partial L}{\partial Z_1} = \left(\delta_2 W_2^T\right) \odot \mathbf{1}(Z_1 > 0) \in \mathbb{R}^{N \times W}$$
Input layer parameter gradients:
$$\frac{\partial L}{\partial W_1} = X^T \delta_1 \in \mathbb{R}^{64 \times W}$$
$$\frac{\partial L}{\partial b_1} = \mathbf{1}_N^T \delta_1 \in \mathbb{R}^W$$

---

## 3. Numerical Gradient Checking Formula
To verify the analytical calculations above, each parameter $\theta_j$ is perturbed symmetrically:
$$g_{\text{numerical}}(\theta_j) = \frac{L(\theta_j + \epsilon) - L(\theta_j - \epsilon)}{2 \epsilon}$$
Relative error comparison:
$$\text{RelError} = \frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{\max(|g_{\text{analytical}}|, |g_{\text{numerical}}|, 10^{-12})}$$
Empirical verification confirms $\text{RelError} < 10^{-6}$ for all tested parameters.
