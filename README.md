# Simple-NN

A lightweight neural network library for educational purposes and rapid prototyping.

## Description

Simple-NN is a minimal neural network implementation designed to help you understand how neural networks work under the hood. It provides basic building blocks for creating, training, and evaluating neural networks without the complexity of larger frameworks.

## Features

- **Simple Architecture**: Easy-to-understand implementation of neural networks
- **Flexible Layers**: Support for dense (fully connected) layers
- **Activation Functions**: Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)
- **Training**: Basic backpropagation and gradient descent
- **No Dependencies**: Pure Python implementation (NumPy optional for performance)

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Install from source

```bash
# Clone the repository
git clone https://github.com/janis-ozolins/simple-nn.git
cd simple-nn

# Install in development mode
pip install -e .
```

### Install with NumPy (recommended for better performance)

```bash
pip install numpy
pip install -e .
```

## Quick Start

### Basic Usage

```python
from simple_nn.network import NeuralNetwork
from simple_nn.layers import Dense
from simple_nn.activations import ReLU, Softmax

# Create a neural network
network = NeuralNetwork()
network.add(Dense(input_size=4, output_size=8))
network.add(ReLU())
network.add(Dense(input_size=8, output_size=3))
network.add(Softmax())

# Train the network
X_train = [...]  # Your training data
Y_train = [...]  # Your training labels

network.train(X_train, Y_train, epochs=100, learning_rate=0.01)

# Make predictions
predictions = network.predict(X_test)
```

## Project Structure

```
simple-nn/
├── simple_nn/
│   ├── __init__.py
│   ├── network.py      # NeuralNetwork class
│   ├── layers.py       # Layer implementations
│   ├── activations.py  # Activation functions
│   ├── loss.py         # Loss functions
│   └── utils.py        # Utility functions
├── examples/
│   ├── xor.py          # XOR problem example
│   ├── mnist.py        # MNIST classification example
│   └── regression.py   # Regression example
├── tests/
│   └── test_network.py # Unit tests
├── README.md           # This file
├── AGENTS.md          # AI Agent configurations
└── setup.py           # Package configuration
```

## Examples

### XOR Problem

```python
from simple_nn.examples.xor import run_xor_example
run_xor_example()
```

### MNIST Classification

```python
from simple_nn.examples.mnist import run_mnist_example
run_mnist_example()
```

## Configuration

You can configure the network through various parameters:

- `learning_rate`: Step size for gradient descent (default: 0.01)
- `epochs`: Number of training iterations (default: 100)
- `batch_size`: Size of mini-batches (default: 32)
- `verbose`: Whether to print training progress (default: True)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
