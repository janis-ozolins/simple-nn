-- | Simple-NN: a small feed-forward neural network with backpropagation.
--
-- This module re-exports the public API of the library. The implementation
-- is split across the "NN.*" modules:
--
-- * "NN.Activation" - activation functions and derivatives
-- * "NN.Neuron"     - core data types ('Neuron', 'ForwardNeuronCal', 'Input') and 'calcZ'
-- * "NN.Cost"       - loss functions and derivatives
-- * "NN.Network"    - construction and weight initialization
-- * "NN.Forward"    - forward pass and inference
-- * "NN.Backward"   - gradient computation and weight updates
-- * "NN.Train"      - training loop
module NN
  ( -- * Types
    Neuron (..)
  , ForwardNeuronCal (..)
  , Input (..)
    -- * Activation
  , sigmoid
  , sigmoid'
    -- * Construction
  , createSigmoidNeuron
  , createSigmoidLayer
  , createNN
  , createNNwGen
  , randomSigmoid
  , chunkLayers
    -- * Forward pass
  , calcZ
  , forwardNNCalInput
  , forwardNNCal
  , forwardNNLayerCal
  , forwardNeuronCal
  , predict
    -- * Cost
  , cost
  , cost'
    -- * Backward pass / training
  , rate
  , backward
  , backpropagate
  , train
  , trainUl
  ) where

import NN.Activation (sigmoid, sigmoid')
import NN.Backward (backward, rate)
import NN.Cost (cost, cost')
import NN.Forward (forwardNNCal, forwardNNCalInput, forwardNNLayerCal, forwardNeuronCal, predict)
import NN.Network (chunkLayers, createNN, createNNwGen, createSigmoidLayer, createSigmoidNeuron, randomSigmoid)
import NN.Neuron (ForwardNeuronCal (..), Input (..), Neuron (..), calcZ)
import NN.Train (backpropagate, train, trainUl)
