# CUDA Neural Network for MNIST Classification

This project implements a neural network from scratch to classify handwritten digits from the MNIST dataset. The network is trained using backpropagation and is progressively optimized using CUDA across four versions, each adding new layers of GPU-based acceleration and optimization techniques.

## 📁 Project Structure

The project is divided into the following versions:

- **V1/** – The given sequential CPU implementation.
- **V2/** – A naive GPU implementation using CUDA for parallelization.
- **V3/** – An optimized GPU implementation that includes:
  - Tuned kernel **launch configurations**
  - Improved **occupancy** of GPU resources
  - **Communication optimizations** to reduce CPU-GPU data transfer overhead
  - **Memory optimizations**, leveraging the CUDA memory hierarchy (shared memory, constant memory, etc.)
- **V4/** – Builds on V3 with the additional use of **Tensor Cores** for matrix operations (requires compatible hardware).

Each version includes a `Makefile` for compilation and execution.

---

## 🧠 Neural Network Architecture

- **Input Layer**: 784 neurons (28×28 pixels)
- **Hidden Layer**: 128 neurons
- **Output Layer**: 10 neurons (for digits 0 to 9)
- **Activation**: ReLU (hidden), Softmax (output)
- **Loss Function**: Cross-entropy
- **Optimizer**: Stochastic Gradient Descent (SGD)

---

## ⚙️ Requirements

- CUDA Toolkit (11.x or newer recommended)
- Compatible NVIDIA GPU (Tensor Cores required for V4)
- MNIST dataset (in IDX format)

---

## 📦 Dataset Setup

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

---

## 🚀 Compilation & Execution

Navigate into the desired version folder and run:

```bash
make run

make        # Compiles the project
make clean  # Cleans build files


## Example 
cd V3
make run
