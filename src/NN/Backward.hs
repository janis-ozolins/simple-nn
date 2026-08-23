-- | Backward pass: gradient computation and weight updates.
module NN.Backward
  ( rate
  , backward
  ) where

import Data.Maybe (fromMaybe, listToMaybe)

import NN.Activation (sigmoid')
import NN.Neuron (Neuron (..), ForwardNeuronCal (..))

-- | Learning rate used by 'backward'.
rate :: Double
rate = 0.5

backward :: Bool -> [Double] -> [[Neuron]] -> [[ForwardNeuronCal]] -> [[Neuron]]
backward isOutput da (l:ls) (f:fp:fs) = zipWith updateNeuron (zip dW dB) l : backward False pDa ls (fp:fs)
    where
        -- For output layer with BCE+sigmoid: dL/dz = da (no sigmoid' multiplication)
        -- For hidden layers: dL/dz = da * sigmoid'(a) where a is post-activation, da is error from next layer
        dZ = if isOutput then da else zipWith (*) da (map (sigmoid' . activation) f)

        -- Calculate weight and bias gradients
        prevActivations = map activation fp  -- Activations from previous layer
        dW = zipWith (\z as -> map (* z) as) dZ (repeat prevActivations)
        dB = dZ

        -- Calculate error for previous layer: pDa = W^T * dZ
        weightsT = map inputWeights l  -- shape: (current_layer_size, prev_layer_size)
        pDa = [sum [dZ !! i * weightsT !! i !! j | i <- [0..length dZ - 1]] | j <- [0..length firstWeights - 1]]
          where firstWeights = fromMaybe [] (listToMaybe weightsT)

        -- Update neuron function
        updateNeuron (dw, db) n = Neuron
            (zipWith (\w d -> w - rate * d) (inputWeights n) dw)
            (bias n - rate * db)
            (activate n)
            (activate' n)
backward _ da _ _ = []
