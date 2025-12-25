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

# TP Part 2: GANs on Abstract Art Gallery

## Overview
Implementation of Generative Adversarial Network (GAN) to generate abstract art using PyTorch.

## Architecture

### Generator
- **Input**: Random noise (100D latent vector)
- **Structure**: FC → Conv2D Transpose layers
- **Output**: 64×64×3 images
- **Activation**: ReLU (hidden), Tanh (output [-1,1])
- **Parameters**: 2,344,451

### Discriminator
- **Input**: 64×64×3 images
- **Structure**: Conv2D layers with stride 2
- **Output**: Binary classification (real/fake)
- **Activation**: LeakyReLU(0.2)
- **Parameters**: 2,789,217

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Latent Dimension | 100 |
| Batch Size | 64 |
| Learning Rate | 2e-4 |
| Beta1 (Adam) | 0.5 |
| Epochs | 100 |
| Image Size | 64×64 |

## Loss Function
```
D_loss = BCE(D(x), 1) + BCE(D(G(z)), 0)
G_loss = BCE(D(G(z)), 1)
```

## Results

### Training Metrics (Final - Epoch 100)
| Metric | Value |
|--------|-------|
| Generator Loss | 7.0627 |
| Discriminator Loss | 0.0188 |
| D(x) - Real Images | 0.9928 ✓ |
| D(G(z)) - Fake Images | 0.0024  |

## Key Observations

**Discriminator Performance**: Excellent (D(x) = 0.99)  
**Generator Struggle**: D(G(z)) → 0 indicates discriminator too strong (overpowering generator)  
**Mode Collapse Risk**: Generated images are random noise (quality poor)  
**Imbalance**: G_loss increasing while D_loss decreasing = GAN instability

## Generated Output
- Images are noisy/random patterns (8×8 blocks visible)
- No coherent abstract art patterns learned
- Indicates training instability or mode collapse

## Conclusions

**Current Status**: GAN not converging properly
- Generator unable to fool discriminator
- Discriminator too powerful
- Random data caused training failure
