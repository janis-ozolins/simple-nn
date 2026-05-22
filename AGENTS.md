# AI Agents for Simple-NN

This document describes how to use AI agents with the Simple-NN neural network library for automated training, hyperparameter tuning, and other intelligent operations.

## Overview

Simple-NN supports AI agents that can automate various tasks such as:
- Automated hyperparameter tuning
- Neural architecture search
- Intelligent training monitoring
- Automated model evaluation and selection

## Agent Types

### 1. HyperparameterTuner Agent

Automatically searches for optimal hyperparameters for your neural network.

**Configuration:**
```yaml
agent:
  type: HyperparameterTuner
  parameters:
    learning_rates: [0.001, 0.01, 0.1]
    batch_sizes: [16, 32, 64]
    epochs_list: [50, 100, 200]
    hidden_layer_sizes: [[32], [64], [32, 32], [64, 32]]
  strategy: grid_search  # or random_search, bayesian
  metric: accuracy
  n_trials: 20
```

**Usage:**
```python
from simple_nn.agents import HyperparameterTuner
from simple_nn.network import NeuralNetwork

# Define your search space
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'hidden_units': [32, 64, 128]
}

# Create the tuner agent
tuner = HyperparameterTuner(param_grid, n_trials=10)

# Run the search
best_params, best_score = tuner.search(
    X_train, Y_train, X_val, Y_val,
    network_builder=lambda: NeuralNetwork()
)

print(f"Best parameters: {best_params}")
print(f"Best validation score: {best_score}")
```

### 2. ArchitectureSearch Agent

Searches for optimal neural network architectures.

**Configuration:**
```yaml
agent:
  type: ArchitectureSearch
  parameters:
    max_layers: 5
    max_units_per_layer: 256
    possible_activations: [relu, sigmoid, tanh]
    possible_optimizers: [sgd, adam]
  search_strategy: evolutionary
  population_size: 10
  generations: 5
```

**Usage:**
```python
from simple_nn.agents import ArchitectureSearch

searcher = ArchitectureSearch(
    input_size=784,
    output_size=10,
    max_layers=4,
    population_size=10,
    generations=5
)

best_architecture = searcher.search(X_train, Y_train, X_val, Y_val)
print(f"Best architecture: {best_architecture}")
```

### 3. TrainingMonitor Agent

Monitors training progress and can automatically adjust parameters or stop training early.

**Configuration:**
```yaml
agent:
  type: TrainingMonitor
  parameters:
    early_stopping:
      enabled: true
      patience: 10
      min_delta: 0.001
    learning_rate_adaptation:
      enabled: true
      reduce_on_plateau: true
      factor: 0.5
      patience: 5
    checkpointing:
      enabled: true
      save_best_only: true
      filepath: best_model.pkl
```

**Usage:**
```python
from simple_nn.agents import TrainingMonitor

monitor = TrainingMonitor(
    early_stopping_patience=10,
    reduce_lr_on_plateau=True,
    checkpoint_path='checkpoints/model.pkl'
)

# Attach to your network
network = NeuralNetwork()
monitor.attach(network)

# Train as usual - monitor will handle the rest
network.train(X_train, Y_train, epochs=200, validation_data=(X_val, Y_val))
```

### 4. Ensemble Agent

Creates and manages ensembles of neural networks.

**Configuration:**
```yaml
agent:
  type: Ensemble
  parameters:
    n_models: 5
    aggregation: vote  # or average, weighted
    diversity_strategy: random_init  # or different_architectures
```

**Usage:**
```python
from simple_nn.agents import EnsembleAgent

ensemble = EnsembleAgent(
    n_models=5,
    aggregation='vote',
    base_network_builder=lambda: NeuralNetwork()
)

# Train the ensemble
ensemble.train(X_train, Y_train, epochs=100)

# Use the ensemble for predictions
predictions = ensemble.predict(X_test)
```

## Agent Configuration File

You can define agents in a YAML configuration file:

```yaml
# agents_config.yaml
agents:
  hyperparameter_tuner:
    type: HyperparameterTuner
    parameters:
      learning_rates: [0.001, 0.01, 0.1]
      batch_sizes: [16, 32, 64]
    strategy: random_search
    n_trials: 15

  training_monitor:
    type: TrainingMonitor
    parameters:
      early_stopping_patience: 10
      reduce_lr_on_plateau: true
      checkpoint_path: models/best_model.pkl
```

**Load and use configurations:**
```python
from simple_nn.agents import load_agents_from_config

agents = load_agents_from_config('agents_config.yaml')
tuner = agents['hyperparameter_tuner']
monitor = agents['training_monitor']
```

## Creating Custom Agents

You can create custom agents by subclassing the `BaseAgent` class:

```python
from simple_nn.agents.base import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, custom_param: float):
        super().__init__()
        self.custom_param = custom_param
    
    def setup(self, network):
        """Called when agent is attached to a network"""
        self.network = network
        # Initialize agent-specific state
    
    def on_epoch_end(self, epoch, logs):
        """Called at the end of each epoch"""
        # Implement custom behavior
        if logs['val_loss'] < 0.1:
            print("Early stopping due to low validation loss")
            self.network.stop_training = True
    
    def on_training_end(self, logs):
        """Called when training is complete"""
        # Finalize agent operations
        pass
```

**Register your custom agent:**
```python
from simple_nn.agents.registry import register_agent

@register_agent('my_custom_agent')
class MyCustomAgent(BaseAgent):
    # ... implementation
```

## Agent Lifecycle Methods

All agents support the following lifecycle methods:

- `setup(network)`: Called when agent is attached to a network
- `on_training_begin(logs)`: Called at the start of training
- `on_epoch_begin(epoch, logs)`: Called at the start of each epoch
- `on_batch_begin(batch, logs)`: Called at the start of each batch
- `on_batch_end(batch, logs)`: Called at the end of each batch
- `on_epoch_end(epoch, logs)`: Called at the end of each epoch
- `on_training_end(logs)`: Called at the end of training
- `on_predict_begin(logs)`: Called at the start of prediction
- `on_predict_end(logs)`: Called at the end of prediction

## Best Practices

1. **Start Simple**: Begin with basic agents like TrainingMonitor before using more complex ones
2. **Monitor Resources**: Some agents (especially ArchitectureSearch) can be resource-intensive
3. **Validate Results**: Always validate agent recommendations on a holdout validation set
4. **Combine Agents**: Use multiple agents together for better results (e.g., HyperparameterTuner + TrainingMonitor)
5. **Log Everything**: Enable verbose logging to understand agent decisions

## Example: Complete Training Pipeline

```python
from simple_nn.network import NeuralNetwork
from simple_nn.layers import Dense
from simple_nn.activations import ReLU, Softmax
from simple_nn.agents import TrainingMonitor, HyperparameterTuner

# Create a simple network builder function
def create_network(learning_rate=0.01, hidden_units=64):
    network = NeuralNetwork(learning_rate=learning_rate)
    network.add(Dense(input_size=784, output_size=hidden_units))
    network.add(ReLU())
    network.add(Dense(input_size=hidden_units, output_size=10))
    network.add(Softmax())
    return network

# Set up agents
monitor = TrainingMonitor(
    early_stopping_patience=10,
    checkpoint_path='models/best_model.pkl'
)

tuner = HyperparameterTuner(
    param_grid={
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_units': [32, 64, 128]
    },
    n_trials=10
)

# Run hyperparameter search
best_params, best_score = tuner.search(
    X_train, Y_train, X_val, Y_val,
    network_builder=lambda: create_network(
        learning_rate=0.01,  # Will be overridden
        hidden_units=64       # Will be overridden
    )
)

# Train final model with best parameters
final_network = create_network(**best_params)
monitor.attach(final_network)
final_network.train(X_train, Y_train, epochs=200, validation_data=(X_val, Y_val))

# Load the best checkpoint
final_network.load('models/best_model.pkl')
```

## Troubleshooting

### Common Issues

1. **Agent not working**: Ensure the agent is properly attached to the network using `agent.attach(network)`
2. **Slow performance**: Reduce the search space or number of trials for tuning agents
3. **Memory issues**: Use smaller population sizes for ArchitectureSearch or reduce model complexity
4. **Configuration errors**: Validate your YAML configuration files for syntax errors

### Debug Mode

Enable debug mode for verbose agent logging:

```python
from simple_nn.agents import set_debug_mode

set_debug_mode(True)
# Now all agents will log detailed information
```

## API Reference

### BaseAgent

```python
class BaseAgent:
    def __init__(self, **kwargs):
        """Initialize the agent with configuration"""
    
    def setup(self, network):
        """Attach agent to a network"""
    
    def attach(self, network):
        """Convenience method to setup and register agent"""
    
    def detach(self):
        """Remove agent from network"""
```

### Agent Registry

```python
from simple_nn.agents.registry import (
    register_agent,
    get_agent_class,
    list_available_agents,
    create_agent
)

# List all available agents
available = list_available_agents()

# Create an agent by name
agent = create_agent('HyperparameterTuner', param_grid={...})
```
