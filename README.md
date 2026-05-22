# Simple-NN

A simple neural network implementation in Haskell for learning purposes.

## Description

A minimal feedforward neural network with backpropagation, implemented in Haskell. This project demonstrates the core concepts of neural networks including forward propagation, backward propagation, and training using gradient descent.

## Features

- Feedforward neural network with configurable architecture
- Sigmoid activation function
- Binary cross-entropy loss
- Backpropagation for training
- XOR problem demonstration
- Random weight initialization (Xavier/Glorot)

## Build & Run

### Prerequisites
- [GHC](https://www.haskell.org/ghc/) (Glasgow Haskell Compiler)

### Building

```bash
# Using Makefile
make build

# Or directly with GHC
ghc -O0 -j -o simple-nn src/nn.hs

# Or using the build script
./build.sh
```

### Running

```bash
# Using Makefile
make run

# Or directly
./dist/build/simple-nn

# Or with build script
./build.sh run
```

## Project Structure

```
simple-nn/
├── src/
│   └── nn.hs          # Main neural network implementation
├── Makefile           # Build configuration
├── build.sh           # Build script
├── README.md          # This file
└── AGENTS.md          # Agent documentation (if applicable)
```

## Usage

The program includes a built-in XOR demonstration. When you run it, it will:

1. Test different network architectures ([2,2,1], [2,4,1], [2,6,1])
2. Train each on the XOR problem
3. Report average error for each architecture
4. Perform comprehensive training on the best architecture
5. Display final predictions with error metrics

## Code Overview

### Core Types

- **Neuron**: Contains input weights, bias, activation function, and its derivative
- **ForwardNeuronCal**: Stores calculation results (z and activation)
- **Input**: Training data with features and expected output

### Key Functions

- `createNN`: Creates a neural network with specified layer sizes
- `forwardNNCalInput`: Performs forward propagation
- `backward`: Implements backpropagation
- `train`: Trains the network on a single input
- `predict`: Makes predictions using the trained network

### Training Parameters

- Learning rate: `rate = 0.5` (defined in nn.hs)
- Epsilon (error threshold): Configurable per training run
- Max iterations: Configurable per training run

## Example Output

```
=== XOR Function Learning Demo ===
Testing different network architectures for XOR problem...

XOR Truth Table:
[0,0] -> 0
[0,1] -> 1
[1,0] -> 1
[1,1] -> 0

Testing network architectures:

--- Testing architecture: [2,2,1] ---
Architecture [2,2,1]: Avg Error = 0.2499...

--- Testing architecture: [2,4,1] ---
Architecture [2,4,1]: Avg Error = 0.1234...

--- Testing architecture: [2,6,1] ---
Architecture [2,6,1]: Avg Error = 0.0876...

=== Architecture Comparison ===
Architecture [2,2,1]: Avg Error = ...
Architecture [2,4,1]: Avg Error = ...
Architecture [2,6,1]: Avg Error = ...

=== Comprehensive Training on Best Architecture ===
Using architecture: [2,4,1]

=== Final Evaluation ===
✓ [0,0] -> 0 -> Predicted: 0.0123 (Expected: 0.0, Error: 0.0123)
✓ [0,1] -> 1 -> Predicted: 0.9876 (Expected: 1.0, Error: 0.0123)
✓ [1,0] -> 1 -> Predicted: 0.9821 (Expected: 1.0, Error: 0.0178)
✓ [1,1] -> 0 -> Predicted: 0.0214 (Expected: 0.0, Error: 0.0214)
```

## Customization

To use your own data or modify the network:

1. Edit `src/nn.hs` to add your input data
2. Adjust the `networkSizes` list to test different architectures
3. Modify `epsilon` (error threshold) and `maxIterations` as needed

## Clean

```bash
make clean
# or
rm -rf dist dist-newstyle
```

## License

MIT License
