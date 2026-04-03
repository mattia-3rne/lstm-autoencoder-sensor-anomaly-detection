# LSTM Autoencoder for Anomaly Detection in Sensor Systems

## 1. Abstract

This repository presents a deep learning framework for unsupervised anomaly detection in univariate sensor time series using a Long Short-Term Memory (LSTM) Autoencoder. The motivating scenario is a continuously operating sensor system that produces a regular, periodic signal under nominal conditions; deviations from learned normal behaviour indicate faults, measurement errors, or physical anomalies. The model is trained exclusively on normal data, learning to compress and reconstruct typical temporal patterns through a fixed-dimensional latent bottleneck. At inference time, the per-window reconstruction error serves as a pointwise anomaly score, and windows whose error exceeds a statistically derived threshold are flagged as anomalous. The framework covers the complete numerical pipeline: sliding-window sequence construction, the explicit LSTM cell dynamics and their matrix formulations, the encoder–decoder bottleneck architecture, the mean-squared-error training objective, backpropagation through time, and the Gaussian-quantile thresholding rule applied to held-out normal data.

---

## 2. Theoretical Background

### 2.1 Time-Series Representation and Sliding-Window Construction

Let $\{y_t\}_{t=1}^{N}$ denote a univariate time series of $N$ scalar sensor readings sampled at a fixed interval $\Delta t$. Before modelling, the series is standardised to zero mean and unit variance using statistics estimated from the anomaly-free training recording only. If the training recording contains $K$ samples:

$$x_t = \frac{y_t - \mu}{\sigma}, \qquad \mu = \frac{1}{K}\sum_{t=1}^{K} y_t, \qquad \sigma = \sqrt{\frac{1}{K}\sum_{t=1}^{K}(y_t - \mu)^2}$$

The standardised series is then segmented into overlapping windows of fixed length $T$ with stride 1. The $i$-th window is the column vector:

```math
\mathbf{w}_i = \begin{bmatrix} x_i \\ x_{i+1} \\ \vdots \\ x_{i+T-1} \end{bmatrix} \in \mathbb{R}^T, \qquad i = 1, 2, \ldots, N - T + 1
```

Each scalar entry $x_{i+t-1}$ constitutes the input at timestep $t$ within the window. The full dataset of windows is partitioned into a training set $\mathcal{X}$ drawn exclusively from the anomaly-free recording, and a test set $\mathcal{Y}$ drawn from the anomalous recording.

### 2.2 The LSTM Cell: Gated State Dynamics

At the core of the autoencoder is the Long Short-Term Memory cell, designed to mitigate the vanishing gradient problem inherent in standard recurrent neural networks. A single LSTM cell maintains two internal state vectors at each timestep $t$: 

1.  **The Cell State** ($\mathbf{c}_t \in \mathbb{R}^p$): This acts as the "conveyor belt" of the network, carrying long-term memory across timesteps with minimal linear interactions, allowing gradients to flow easily during backpropagation.
2.  **The Hidden State** ($\mathbf{h}_t \in \mathbb{R}^p$): This serves as the working memory and the actual output of the cell for the current timestep, derived directly from the cell state.

Here, $p$ represents the number of hidden units in the encoder layer. Because our sensor data is univariate, the input $x_t \in \mathbb{R}^1$ at any timestep is a simple scalar. Consequently, the input weight matrices reduce to column vectors $\mathbf{W}^E_{\square} \in \mathbb{R}^{p \times 1}$. The recurrent weight matrices, which map the previous hidden state to the current gates, are square matrices $\mathbf{U}^E_{\square} \in \mathbb{R}^{p \times p}$, and the bias vectors are $\mathbf{b}^E_{\square} \in \mathbb{R}^p$.

The state dynamics are governed by four neural network layers inside the cell, commonly called gates, that regulate the addition or removal of information to the cell state. Below is the detailed breakdown of the six update equations executed at every timestep $t = 1, \ldots, T$.

#### 1. The Forget Gate ($\mathbf{f}_t$)
The first step is deciding what information from the previous cell state $c_{t-1}$ is no longer relevant and should be discarded. The forget gate looks at the current input $x_t$ and the previous hidden state $h_{t-1}$, and outputs a vector of values between $0$ (completely forget) and $1$ (completely keep).

$$\mathbf{f}_t = \sigma\left(\mathbf{W}^E_f x_t + \mathbf{U}^E_f \mathbf{h}_{t-1} + \mathbf{b}^E_f\right)$$

Where $\sigma(v) = (1 + e^{-v})^{-1}$ is the elementwise sigmoid activation function. Expanding this into its explicit matrix formulation shows how the scalar input and the hidden state vector linearly combine before the activation:

```math
\mathbf{f}_t = \sigma\left( \begin{bmatrix} W^E_{f,1} \\ W^E_{f,2} \\ \vdots \\ W^E_{f,p} \end{bmatrix} x_t + \begin{bmatrix} U^E_{f,11} & U^E_{f,12} & \cdots & U^E_{f,1p} \\ U^E_{f,21} & U^E_{f,22} & \cdots & U^E_{f,2p} \\ \vdots & \vdots & \ddots & \vdots \\ U^E_{f,p1} & U^E_{f,p2} & \cdots & U^E_{f,pp} \end{bmatrix} \begin{bmatrix} h_{t-1,1} \\ h_{t-1,2} \\ \vdots \\ h_{t-1,p} \end{bmatrix} + \begin{bmatrix} b^E_{f,1} \\ b^E_{f,2} \\ \vdots \\ b^E_{f,p} \end{bmatrix} \right)
```

#### 2. The Input Gate ($\mathbf{i}_t$) and Cell Candidate ($\mathbf{g}_t$)
Next, the cell must decide what new information to store in the cell state. This is a two-part process. First, an **input gate** layer determines *which* values we will update:

$$\mathbf{i}_t = \sigma\left(\mathbf{W}^E_i x_t + \mathbf{U}^E_i \mathbf{h}_{t-1} + \mathbf{b}^E_i\right)$$

Expanding the input gate into its full matrix-vector representation:

```math
\mathbf{i}_t = \sigma\left( \begin{bmatrix} W^E_{i,1} \\ W^E_{i,2} \\ \vdots \\ W^E_{i,p} \end{bmatrix} x_t + \begin{bmatrix} U^E_{i,11} & U^E_{i,12} & \cdots & U^E_{i,1p} \\ U^E_{i,21} & U^E_{i,22} & \cdots & U^E_{i,2p} \\ \vdots & \vdots & \ddots & \vdots \\ U^E_{i,p1} & U^E_{i,p2} & \cdots & U^E_{i,pp} \end{bmatrix} \begin{bmatrix} h_{t-1,1} \\ h_{t-1,2} \\ \vdots \\ h_{t-1,p} \end{bmatrix} + \begin{bmatrix} b^E_{i,1} \\ b^E_{i,2} \\ \vdots \\ b^E_{i,p} \end{bmatrix} \right)
```

Second, a **cell candidate** layer $\mathbf{g}_t$ creates a vector of new, raw candidate values that could potentially be added to the state. It uses a hyperbolic tangent function to map values between $-1$ and $1$:

$$\mathbf{g}_t = \tanh\left(\mathbf{W}^E_g x_t + \mathbf{U}^E_g \mathbf{h}_{t-1} + \mathbf{b}^E_g\right)$$

Writing out the explicit matrix operations for the cell candidate vector:

```math
\mathbf{g}_t = \tanh\left( \begin{bmatrix} W^E_{g,1} \\ W^E_{g,2} \\ \vdots \\ W^E_{g,p} \end{bmatrix} x_t + \begin{bmatrix} U^E_{g,11} & U^E_{g,12} & \cdots & U^E_{g,1p} \\ U^E_{g,21} & U^E_{g,22} & \cdots & U^E_{g,2p} \\ \vdots & \vdots & \ddots & \vdots \\ U^E_{g,p1} & U^E_{g,p2} & \cdots & U^E_{g,pp} \end{bmatrix} \begin{bmatrix} h_{t-1,1} \\ h_{t-1,2} \\ \vdots \\ h_{t-1,p} \end{bmatrix} + \begin{bmatrix} b^E_{g,1} \\ b^E_{g,2} \\ \vdots \\ b^E_{g,p} \end{bmatrix} \right)
```

#### 3. The Cell State Update ($\mathbf{c}_t$)
With the gates computed, the actual internal memory of the cell is updated. The previous state $\mathbf{c}_{t-1}$ is multiplied by the forget gate $\mathbf{f}_t$ (dropping information decided in step 1). Then, the new candidate values $\mathbf{g}_t$ are scaled by the input gate $\mathbf{i}_t$ and added to the memory. This represents the new long-term context:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$$

Here, $\odot$ denotes the elementwise Hadamard product. Visually, this is computed row by row:

```math
\mathbf{c}_t = \begin{bmatrix} f_{t,1} \cdot c_{t-1,1} \\ f_{t,2} \cdot c_{t-1,2} \\ \vdots \\ f_{t,p} \cdot c_{t-1,p} \end{bmatrix} + \begin{bmatrix} i_{t,1} \cdot g_{t,1} \\ i_{t,2} \cdot g_{t,2} \\ \vdots \\ i_{t,p} \cdot g_{t,p} \end{bmatrix} = \begin{bmatrix} f_{t,1} c_{t-1,1} + i_{t,1} g_{t,1} \\ f_{t,2} c_{t-1,2} + i_{t,2} g_{t,2} \\ \vdots \\ f_{t,p} c_{t-1,p} + i_{t,p} g_{t,p} \end{bmatrix}
```

#### 4. The Output Gate ($\mathbf{o}_t$) and Hidden State Update ($\mathbf{h}_t$)
Finally, the cell determines what it will output to the next layer in the network (and what it will pass to itself as $\mathbf{h}_t$ for the next timestep). The **output gate** computes a sigmoid-filtered representation of the current inputs:

$$\mathbf{o}_t = \sigma\left(\mathbf{W}^E_o x_t + \mathbf{U}^E_o \mathbf{h}_{t-1} + \mathbf{b}^E_o\right)$$

The new cell state $\mathbf{c}_t$ is then passed through a $\tanh$ function to push its values between $-1$ and $1$, and multiplied by the output gate. This ensures the LSTM only outputs the parts of the cell state that are highly relevant to the current timestep:

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

Expressed explicitly as vectors, the final working memory output is:

```math
\mathbf{h}_t = \begin{bmatrix} o_{t,1} \cdot \tanh(c_{t,1}) \\ o_{t,2} \cdot \tanh(c_{t,2}) \\ \vdots \\ o_{t,p} \cdot \tanh(c_{t,p}) \end{bmatrix}
```

The initial states for the very first timestep of the window are defined as $\mathbf{h}_0 = \mathbf{c}_0 = \mathbf{0}$. By stepping through these equations sequentially from $t=1$ to $t=T$, the encoder compresses the temporal structure of the input window into the terminal hidden state $\mathbf{h}_T$.

### 2.3 The Encoder–Decoder Bottleneck

The autoencoder is composed of two LSTM modules and a single shared linear readout layer.

**Encoder:** The encoder LSTM with $p$ hidden units processes the full input window sequentially from $t = 1$ to $t = T$. Only the terminal hidden state is retained as the latent code:

$$\mathbf{z} = \mathbf{h}_T \in \mathbb{R}^p$$

The terminal cell state $\mathbf{c}_T$ is also forwarded to the decoder as its initial cell state to preserve gradient flow through the bottleneck during backpropagation.

**Bottleneck:** The latent vector $\mathbf{z}$ has dimension $p \ll T$, forcing the encoder to extract a compact summary of the temporal pattern. For a window of length $T = 288$ and $p = 128$, the compression ratio is $288 : 128 \approx 2.25$. Normal windows, which exhibit a consistent periodic structure, compress and reconstruct faithfully; anomalous windows deviate from the learned manifold and incur high reconstruction error.

**Decoder:** The decoder receives the latent code broadcast across all $T$ positions. Define the repeated input matrix:

```math
\mathbf{Z} = \begin{bmatrix} \mathbf{z}^\top \\ \mathbf{z}^\top \\ \vdots \\ \mathbf{z}^\top \end{bmatrix} \in \mathbb{R}^{T \times p}
```

The decoder LSTM with $q$ units is initialized at $(\mathbf{s}_0, \mathbf{m}_0) = (\mathbf{0},\, \mathbf{c}_T)$ and processes $\mathbf{Z}$ from $s = 1$ to $s = T$, yielding hidden states $\mathbf{s}_1, \ldots, \mathbf{s}_T \in \mathbb{R}^q$.

**Readout.** A shared linear layer $\mathbf{W}_R \in \mathbb{R}^{1 \times q}$, $b_R \in \mathbb{R}$ maps each decoder hidden state to a scalar reconstruction:

$$r_s = \mathbf{W}_R\, \mathbf{s}_s + b_R = \sum_{j=1}^{q} W_{R,j}\, s_{s,j} + b_R \in \mathbb{R}$$

The full reconstructed window is then:

```math
\mathbf{r} = \begin{bmatrix} r_1 \\ r_2 \\ \vdots \\ r_T \end{bmatrix} \in \mathbb{R}^T
```

### 2.4 Reconstruction Error as an Anomaly Score

The per-window anomaly score is the mean squared error between the original window and its reconstruction:

$$e(\mathbf{w}) = \frac{1}{T} \sum_{t=1}^{T} (x_t - r_t)^2 = \frac{1}{T}\|\mathbf{w} - \mathbf{r}\|_2^2$$

Because the model is trained only on normal data, the decoder has learned to invert the encoder specifically for nominal temporal patterns. An anomalous window maps to a latent code that the decoder cannot faithfully reconstruct, yielding a large $e(\mathbf{w})$, while normal windows yield a small one.

### 2.5 Gaussian-Quantile Thresholding

After training, the model is evaluated on the held-out normal windows to obtain the empirical distribution of reconstruction errors under nominal conditions:

$$\mathcal{E} = \{ e(\mathbf{w}_i) \mid \mathbf{w}_i \in \mathcal{X} \}$$

Assuming $\mathcal{E}$ is approximately Gaussian, the sample mean and standard deviation are estimated over the $M$ training windows:

$$\mu_e = \frac{1}{M}\sum_{i=1}^{M} e(\mathbf{w}_i), \qquad \sigma_e = \sqrt{\frac{1}{M}\sum_{i=1}^{M}\left(e(\mathbf{w}_i) - \mu_e\right)^2}$$

The anomaly threshold $\tau$ is set at the $(1 - \alpha)$-quantile of this fitted distribution:

$$\tau = \mu_e + \Phi^{-1}(1 - \alpha)\, \sigma_e$$

where $\Phi^{-1}$ is the standard normal quantile function. A window is declared anomalous if and only if its reconstruction error exceeds the threshold:

$$a(\mathbf{w}) = \mathbb{1}\left[e(\mathbf{w}) > \tau\right]$$

---

## 3. Mathematical Framework

### 3.1 The Training Objective

The model is trained to minimise the average reconstruction error over all $M$ training windows. The loss function is:

$$\mathcal{L}(\phi) = \frac{1}{M} \sum_{i=1}^{M} e(w_i) = \frac{1}{M \cdot T} \sum_{i=1}^{M} \sum_{t=1}^{T} (x_{i,t} - r_{i,t})^2$$

where $\phi$ collects all trainable parameters of both LSTMs and the readout layer. The total number of scalar parameters in the encoder LSTM alone is:

$$|\phi_E| = 4(p \cdot 1 + p^2 + p) = 4p(p + 2)$$

since each of the four gates contributes one input weight vector $W^E_{\square} \in \mathbb{R}^{p \times 1}$, one recurrent weight matrix $U^E_{\square} \in \mathbb{R}^{p \times p}$, and one bias $b^E_{\square} \in \mathbb{R}^p$. For the decoder LSTM with input dimension $p$ (the latent code) and hidden dimension $q$, the count is $4q(p + q + 1)$.

### 3.2 Backpropagation Through Time

Minimising $\mathcal{L}$ requires differentiating through the sequential LSTM unrolling. The gradient with respect to the readout weight $W_{R,j}$ follows directly from the chain rule applied to the squared loss:

$$\frac{\partial \mathcal{L}}{\partial W_{R,j}} = -\frac{2}{M \cdot T} \sum_{i=1}^{M} \sum_{s=1}^{T} (x_{i,s} - r_{i,s}) s_{i,s,j}$$

For the recurrent weights, the gradient accumulates across all timesteps. For the encoder forget-gate recurrent weight $U^E_{f,jk}$:

$$\frac{\partial \mathcal{L}}{\partial U^E_{f,jk}} = \sum_{i=1}^{M} \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial h_{i,t,j}} \cdot \frac{\partial h_{i,t,j}}{\partial f_{i,t,j}} \cdot \frac{\partial f_{i,t,j}}{\partial U^E_{f,jk}}$$

The last factor is the previous hidden state:

$$\frac{\partial f_{i,t,j}}{\partial U^E_{f,jk}} = \sigma'(a_{f,t,j}) h_{i,t-1,k}, \qquad \sigma'(v) = \sigma(v)(1 - \sigma(v))$$

where $a_{f,t,j} = W^E_{f,j} x_{i,t} + \sum_k U^E_{f,jk} h_{i,t-1,k} + b^E_{f,j}$ is the pre-activation.

The error signal $\partial \mathcal{L} / \partial h_t$ is computed via the backward recurrence from $t = T$ down to $t = 1$:

$$\delta_t^h = \frac{\partial \mathcal{L}}{\partial h_t} = ({U^E_f}^\top \delta_{t+1}^f + {U^E_i}^\top \delta_{t+1}^i + {U^E_g}^\top \delta_{t+1}^g + {U^E_o}^\top \delta_{t+1}^o) + \frac{\partial \mathcal{L}}{\partial z} 1_{[t = T]}$$

where $\delta_{t}^f = (\delta_t^h \odot o_t \odot (1 - \tanh^2(c_t)) \odot c_{t-1}) \odot \sigma'(a_{f,t})$ is the gate-level error, and $\partial\mathcal{L}/\partial z$ is back-propagated from the decoder through the bottleneck connection. This backward recurrence constitutes the core of BPTT.

### 3.3 The Anomaly Decision Rule

For a chosen false-positive rate $\alpha \in (0, 1)$, the full decision pipeline from raw window to binary label is:

$$\mathbf{w} \xrightarrow{\text{Encoder}} \mathbf{z} \xrightarrow{\text{Decoder}} \mathbf{r} \xrightarrow{\text{MSE}} e(\mathbf{w}) \xrightarrow{\tau} a(\mathbf{w}) = \mathbb{1}\left[e(\mathbf{w}) > \tau\right]$$

where the threshold is:

$$\tau = \mu_e + \Phi^{-1}(1 - \alpha) \sigma_e$$

Common choices are $\alpha = 0.05$ (giving $\Phi^{-1}(0.95) \approx 1.645$) and $\alpha = 0.01$ (giving $\Phi^{-1}(0.99) \approx 2.326$). A more conservative threshold reduces false positives at the cost of increased false negatives. Under the Gaussian assumption, the expected fraction of normal windows incorrectly flagged is exactly $\alpha$.

---

## 4. Pipeline Architecture

The computational framework is structured as a four-phase sequential pipeline operating on the two NAB sensor recordings.

| Phase | Process | Methodological Details |
| :--- | :--- | :--- |
| **1** | **Data Ingestion** | Loads both CSV files directly from the NAB GitHub repository. Parses the timestamp-column as a datetime-index. Saves raw DataFrames to the data-directory. Computes normalisation statistics $\mu$, $\sigma$ from the anomaly-free series only to prevent data leakage. |
| **2** | **Window Construction** | Applies the sliding-window procedure with stride 1 to both standardised series, producing NumPy arrays of shape $(N - T + 1, T, 1)$. Splits the normal-data windows into a training partition and a validation partition. |
| **3** | **Model Training** | Instantiates the encoder–decoder LSTM, compiles with the Adam optimiser and MSE loss $\mathcal{L}(\boldsymbol{\phi})$. Trains for a fixed number of epochs with early stopping monitored on validation loss. |
| **4** | **Anomaly Scoring** | Runs inference on all test windows, computes per-window MSE, estimates $\mu_e$ and $\sigma_e$ from training-set errors, applies the threshold $\tau$, and overlays detected anomaly regions on the original time series. |

### 4.1 Data Ingestion and Normalisation

Both datasets are fetched from their respective raw-content URLs. Each CSV has two columns: timestamp (parsed as pd.Timestamp) and value (float64). Normalisation parameters are estimated from the training series alone, where $K$ is the length of the normal sequence:

$$\mu = \frac{1}{K}\sum_{t=1}^{K} y_t, \qquad \sigma = \sqrt{\frac{1}{K}\sum_{t=1}^{K}\left(y_t - \mu\right)^2}$$

The anomaly series is transformed using the same $\mu$ and $\sigma$, ensuring that the test-set error distribution is directly comparable to the training-set threshold.

### 4.2 Encoder and Decoder Weight Dimensions

For window length $T$, encoder hidden size $p$, and decoder hidden size $q$, the full set of trainable tensors and their shapes is:

| Layer | Parameter | Shape |
| :--- | :--- | :--- |
| Encoder LSTM | $W^E_{\square}$ (×4 gates) | $\mathbb{R}^{p \times 1}$ |
| Encoder LSTM | $U^E_{\square}$ (×4 gates) | $\mathbb{R}^{p \times p}$ |
| Encoder LSTM | $b^E_{\square}$ (×4 gates) | $\mathbb{R}^{p}$ |
| Decoder LSTM | $W^D_{\square}$ (×4 gates) | $\mathbb{R}^{q \times p}$ |
| Decoder LSTM | $U^D_{\square}$ (×4 gates) | $\mathbb{R}^{q \times q}$ |
| Decoder LSTM | $b^D_{\square}$ (×4 gates) | $\mathbb{R}^{q}$ |
| Readout | $W_R$ | $\mathbb{R}^{1 \times q}$ |
| Readout | $b_R$ | $\mathbb{R}^1$ |

---

## 5. Limitations

| Limitation | Description |
| :--- | :--- |
| **Gaussian Error Assumption** | The threshold derivation assumes reconstruction errors on normal data follow a Gaussian distribution. If the empirical error distribution is heavy-tailed or multimodal, the quantile $\Phi^{-1}(1 - \alpha)$ will not correspond to the intended false-positive rate $\alpha$. |
| **Stationarity Requirement** | Normalisation statistics $\mu$ and $\sigma$ are estimated once from the full training series. If the sensor signal exhibits slow drift or regime changes, the fixed normalisation inflates reconstruction errors on later normal windows, raising the false-positive rate over time. |
| **Window Length Sensitivity** | The window length $T$ must be chosen to capture at least one full period of the dominant oscillation. A window that is too short discards temporal context; one that is too long dilutes anomaly signatures by averaging them with many normal timesteps, reducing $e(\mathbf{w})$ and lowering detection sensitivity. |
| **Vanishing Gradients in BPTT** | Despite the gating mechanism, backpropagation through very long sequences can suffer from vanishing gradients. For $T \gg 300$, the error signal $\boldsymbol{\delta}_t^h$ decays in norm for early timesteps, and the encoder may fail to encode long-range dependencies in the latent code $\mathbf{z}$. |

---

## Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository**:
```bash
    git clone https://github.com/mattia-3rne/lstm-autoencoder-sensor-anomaly-detection.git
```

2.  **Install dependencies**:
```bash
    pip install -r requirements.txt
```

3.  **Run the Analysis**:
```bash
    jupyter notebook
```

---

## Repository Structure
* `requirements.txt`: Python dependencies
* `preprocessing.py`: Data loading script
* `weights.npz`: Autoencoder weights
* `main.ipynb`: Primary notebook
* `model.py`: Model architecture
* `train.py`: Training logic
* `utils.py`: Functions
