# TP Part 1: Autoencoders & VAE on MNIST

## Overview
Implementation and comparison of Autoencoder (AE) and Variational Autoencoder (VAE) on MNIST dataset.

## Models

### Autoencoder
- **Encoder**: 784 → 512 → 256 → 128 → 20
- **Decoder**: 20 → 128 → 256 → 512 → 784
- **Loss**: MSE (Reconstruction only)

### Variational Autoencoder
- **Encoder**: 784 → 512 → 256 → 128 → (μ, log σ²)
- **Decoder**: 20 → 128 → 256 → 512 → 784
- **Loss**: ELBO = Reconstruction Loss + KL Divergence

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Latent Dim | 20 |
| Batch Size | 128 |
| Learning Rate | 1e-3 |
| Epochs | 20 |
| Optimizer | Adam |

## Results

| Metric | AE | VAE |
|--------|----|----|
| Final Loss | 0.0076 | 0.0673 |
| KL Divergence | - | 0.0000 |

## Key Conclusions

**AE**: Better reconstruction (loss = 0.0076) but disconnected latent space  
**VAE**: Higher loss (0.0673) but smooth, continuous latent space  
**KL Divergence**: VAE aligns perfectly with N(0,I)  
**Latent Space**: VAE shows better digit clustering and interpolation capability

## Dataset
MNIST: 60,000 training images, 28×28 pixels, 10 classes (0-9)
