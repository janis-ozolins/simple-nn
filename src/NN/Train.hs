-- | Training loop: tie forward, backward, and cost together.
module NN.Train
  ( train
  , trainUl
  , backpropagate
  ) where

import Data.List (find)
import Data.Maybe (listToMaybe)

import NN.Backward (backward)
import NN.Cost (cost')
import NN.Forward (forwardNNCalInput, predict)
import NN.Neuron (Neuron, ForwardNeuronCal (..), Input (..))

train :: Double -> Int -> [[Neuron]] -> Input -> [[Neuron]]
train epsilon maxIterations network input =
    case find (\n -> abs (predict input n - expected input) < epsilon) $ take (maxIterations + 1) $ trainUl network input of
        Just result -> result
        Nothing -> last $ take (maxIterations + 1) $ trainUl network input

trainUl :: [[Neuron]] -> Input -> [[[Neuron]]]
trainUl network samples = iterate (\x -> backpropagate x samples) network

backpropagate :: [[Neuron]] -> Input -> [[Neuron]]
backpropagate network i = reverse $ backward True [cost' (expected i) guess] (reverse network) (reverse forwardNeurons)
    where
        forwardNeurons = forwardNNCalInput (features i) network
        guess = maybe 0 activation (listToMaybe (last forwardNeurons))
