-- | Forward pass: compute activations layer by layer, plus inference.
module NN.Forward
  ( forwardNNCalInput
  , forwardNNCal
  , forwardNNLayerCal
  , forwardNeuronCal
  , predict
  ) where

import Data.Maybe (listToMaybe)

import NN.Activation (sigmoid)
import NN.Neuron (Neuron (..), ForwardNeuronCal (..), Input (..), calcZ)

forwardNNCalInput :: [Double] -> [[Neuron]] -> [[ForwardNeuronCal]]
forwardNNCalInput input neurons = [map (\x -> ForwardNeuronCal x x) input] ++ forwardNNCal (map (\x -> ForwardNeuronCal x x) input) neurons

forwardNNCal :: [ForwardNeuronCal] -> [[Neuron]] -> [[ForwardNeuronCal]]
forwardNNCal forward (n:ns) = cForward : forwardNNCal cForward ns
    where cForward = forwardNNLayerCal forward n
forwardNNCal _ [] = []

forwardNNLayerCal :: [ForwardNeuronCal] -> [Neuron] -> [ForwardNeuronCal]
forwardNNLayerCal prev neurons = map (forwardNeuronCal prev) neurons

forwardNeuronCal :: [ForwardNeuronCal] -> Neuron -> ForwardNeuronCal
forwardNeuronCal forward neuron = ForwardNeuronCal z a
    where
        z = calcZ (inputWeights neuron) (map activation forward) (bias neuron)
        a = sigmoid z

-- | Run the network on an input and return the output layer's activation.
predict :: Input -> [[Neuron]] -> Double
predict i network = maybe 0 activation (listToMaybe (last (forwardNNCalInput (features i) network)))
