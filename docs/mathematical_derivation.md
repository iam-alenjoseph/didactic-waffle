# Mathematical Derivation: Generative Diffusion Model for Video (DDPM)

---

## 1. Problem Formulation & Video Tensor Space

Let a continuous video sequence be represented as a spatio-temporal tensor:
$$x_0 \in \mathbb{R}^{C \times T \times H \times W}$$
where:
- $C = 3$ denotes RGB color channels.
- $T$ denotes temporal video length (number of frames).
- $H, W$ denote spatial height and width.

We condition video generation on multimodal geometric priors:
$$c = \{d, n\} \in \mathbb{R}^{4 \times T \times H \times W}$$
where $d \in \mathbb{R}^{1 \times T \times H \times W}$ is the metric/normalized depth sequence and $n \in \mathbb{R}^{3 \times T \times H \times W}$ is the surface normal field generated from Unreal Engine 5.

---

## 2. Forward Diffusion Process (Noising)

The forward process is a Markov chain that gradually injects Gaussian noise into video tensor $x_0$ over discrete timesteps $t \in \{1, \dots, T_{diff}\}$ according to a variance schedule $\beta_1, \beta_2, \dots, \beta_{T_{diff}}$:

$$q(x_1, \dots, x_{T_{diff}} | x_0) = \prod_{t=1}^{T_{diff}} q(x_t | x_{t-1})$$

$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}\right)$$

### 2.1 Marginal Distribution $q(x_t | x_0)$

Defining $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:
$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1}$$
By recursive substitution:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Therefore, the forward marginal at arbitrary timestep $t$ is closed-form:
$$q(x_t | x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$

---

## 3. Analytical Reverse Posterior $q(x_{t-1} | x_t, x_0)$

Conditioned on both $x_t$ and $x_0$, the reverse step is tractable using Bayes' rule:
$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

Substituting Gaussian probability density functions:
$$\ln q(x_{t-1} | x_t, x_0) \propto -\frac{1}{2} \left[ \frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0)^2}{1 - \bar{\alpha}_{t-1}} - \frac{(x_t - \sqrt{\bar{\alpha}_t} x_0)^2}{1 - \bar{\alpha}_t} \right]$$

Expanding and completing the square for $x_{t-1}$:
$$q(x_{t-1} | x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I}\right)$$

where:
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

---

## 4. Parameterization & Model Objective

From the reparameterization $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$, we have:
$$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)$$

Plugging this into $\tilde{\mu}_t$:
$$\mu_\theta(x_t, t, c) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right)$$

where $\epsilon_\theta(x_t, t, c)$ is our Spatio-Temporal Video UNet parameterized by neural network weights $\theta$.

### 4.1 Training Objective with Min-SNR Weighting
The simplified ELBO objective (Ho et al., 2020) with Min-SNR weighting (Hang et al., 2023) is:
$$\mathcal{L}_{video}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ w(t) \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right] + \lambda_{temp} \mathcal{L}_{temp}$$
where:
$$w(t) = \min\left( \text{SNR}(t), \gamma \right) / \text{SNR}(t), \quad \text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$
and $\mathcal{L}_{temp}$ is the inter-frame temporal consistency loss:
$$\mathcal{L}_{temp} = \frac{1}{T-1} \sum_{k=1}^{T-1} \| (\hat{x}_{0, k+1} - \hat{x}_{0, k}) - (x_{0, k+1} - x_{0, k}) \|^2$$

---

## 5. Reverse Ancestral Sampling & Fast DDIM Trajectory

### 5.1 DDPM Ancestral Step (Stochastic, 1000 steps)
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right) + \sigma_t z, \quad z \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### 5.2 DDIM Sampling (Deterministic / Semi-Stochastic, 20–50 steps)
Given a sub-sequence $\tau_1 < \tau_2 < \dots < \tau_S$:
$$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \left( \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \epsilon_\theta(x_{\tau_i}, \tau_i, c)}{\sqrt{\bar{\alpha}_{\tau_i}}} \right) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2} \epsilon_\theta(x_{\tau_i}, \tau_i, c) + \sigma_{\tau_i} \epsilon$$
When $\eta = 0.0 \implies \sigma_{\tau_i} = 0$, the process reduces to a deterministic Ordinary Differential Equation (ODE).

---

## 6. Classifier-Free Guidance (CFG) for Video

During generation, we extrapolate between unconditional and conditional predictions:
$$\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot \left( \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset) \right)$$
where $s \ge 1.0$ is the guidance scale. This forces sharp fidelity alignment with the input depth and normal maps.
