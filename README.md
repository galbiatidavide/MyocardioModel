# Mathematical Modeling and Simulation of Cardiac Function

Scientific Computing Tools for Advanced Mathematical Modelling  
**Supervisor**: Prof. Stefano Pagani  
**Institution**: Politecnico di Milano  
**Academic Year**: 2023–2024  

---

## 🧠 Project Overview

This is a mathematical and computational framework to reconstruct, simulate, and infer electrical propagation in the human heart from sparse electro-anatomical data.  

Developed within the course *Scientific Computing Tools for Advanced Mathematical Modelling*, this project uses **Gaussian Processes**, **Neural Networks**, **Autoencoders**, and **U-Net architectures** to tackle challenges in cardiac activation modeling.  
The codebase is structured in three main checkpoints:

---

## 📍 Checkpoint 1: Activation Time & Velocity Field Reconstruction

### 🎯 Goal  
Reconstruct the **activation time field** and **conduction velocity** from 20 sparse intracardiac recordings.

### ⚙️ Method  
- Gaussian Process Regression (GPR) on 20-point input data  
- Kernel tuning using cross-validated hyperparameter search  
- Velocity estimated via gradient inversion + second-level GPR smoothing

<table>
  <tr>
    <td align="center"><strong>Activation Time Field</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/activation_time_field.png" width="300">
    </td>
    <td align="center"><strong>Velocity Field Estimation</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/velocity_field_estimation.png" width="300">
    </td>
  </tr>
</table>

---

## 📍 Checkpoint 2: Parameter Estimation with Surrogate Modeling

### 🎯 Goal  
Infer three physiological parameters:
- Fiber angle (μ₁ ∈ [−π/10, π/10])
- Anisotropy ratio (μ₂ ∈ [1, 9])
- Activation origin y₀ (μ₃ ∈ [−1.5, 1.5])

### ⚙️ Method  
- Train a Neural Network surrogate model for the Eikonal solver `anisotropic_FMM`
- Inputs: (μ₁, μ₂, μ₃, x, y); Output: activation time t  
- Use grid search with progressive refinement to optimize parameters by minimizing the squared error functional:

$$
C(\mu) = \sum_{i=1}^{20} |t_i - F(x_i, \mu)|^2
$$

<table>
  <tr>
    <td align="center"><strong>Loss Evolution</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/loss_function_evolution.png" width="300">
    </td>
    <td align="center"><strong>NN Prediction vs. Eikonal</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/nn_prediction.png" width="300">
    </td>
  </tr>
</table>

---

## 📍 Checkpoint 3: Speed Field Estimation (Parametric & Deep Learning)

### 🎯 Goal  
Estimate a 2D **speed field** \( c(x,y) \) from only 20 activation time recordings.

### 1. Parametric Latent Space (Autoencoder)

- Compress 151×151 speed fields into 8 latent parameters using an **autoencoder**
- Optimize latent parameters to minimize mismatch between real and simulated activation time fields

<table>
  <tr>
    <td align="center"><strong>Autoencoder Architecture</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/autoencoder_architecture.png" width="300">
    </td>
    <td align="center"><strong>Original vs. Decoded Field</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/speedfield_original.png" width="300">
    </td>
  </tr>
</table>

#### Latent Parameter Distributions

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/param_distribution_1.png" width="150"></td>
    <td><img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/param_distribution_2.png" width="150"></td>
    <td><img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/param_distribution_3.png" width="150"></td>
    <td><img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/param_distribution_4.png" width="150"></td>
    <td><img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/param_distribution_5.png" width="150"></td>
    <td><img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/param_distribution_6.png" width="150"></td>
    <td><img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/param_distribution_7.png" width="150"></td>
    <td><img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/param_distribution_8.png" width="150"></td>
  </tr>
</table>

### 2. Image-to-Image U-Net Regression

An alternative approach was tested using a **U-net architecture** to directly learn a mapping from activation time images to speed fields.

<table>
  <tr>
    <td align="center"><strong>U-net Architecture</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/unet_architecture.png" width="300">
    </td>
    <td align="center"><strong>Reconstructed Speed Field</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/MyocardioModel/main/myocardio_images/unet_speedfield_predicted.png" width="300">
    </td>
  </tr>
</table>

---

## 🧠 Code Architecture

- `Checkpoint_1.ipynb` – GPR for activation time & velocity field
- `Checkpoint_2.ipynb` + `functions_CP2.py` – NN surrogate + parameter optimization
- `Checkpoint_3_autoencoder.ipynb` – speed field compression and estimation
- `Checkpoint3_U-net.ipynb` – U-net speed field regression
- `model_NN_new_training_2.keras` – trained model used in CP2

---

## 🧪 Numerical Highlights

- Activation time prediction accuracy: **0.96 / 1**
- U-net reconstruction score: **1.51 / 2**
- Autoencoder compression: effective despite minor information loss

---

## 📚 References

- Eikonal solvers and Gaussian Processes: Rasmussen & Williams (2006)  
- U-Net: Ronneberger et al. (2015)  
- FMM Solver: Sethian (1996)

---

## 📌 Educational Context

This project was completed as part of the course *Scientific Computing Tools for Advanced Mathematical Modelling* held by **Prof. Stefano Pagani** at **Politecnico di Milano**.

