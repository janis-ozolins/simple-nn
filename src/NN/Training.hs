module NN.Training 
    ( rate
    , cost
    , cost'
    , backward
    , backpropagate
    , train
    , trainUl
    , trainOnAll
    , trainComprehensive
    ) where

import NN.Types
import NN.Activation
import NN.Network
import Data.List (find)

-- | Learning rate
rate :: Double
rate = 0.5

-- | Binary cross-entropy cost function
cost :: Double -> Double -> Double
cost expect trueVal = - (expect * log trueVal + (1 - expect) * log (1 - trueVal))

-- | Cost function derivative for binary cross-entropy (dL/dz for output layer with sigmoid)
-- For BCE+sigmoid: dL/dz = prediction - expect
cost' :: Double -> Double -> Double
cost' expect prediction = prediction - expect

-- | Backward propagation
-- isOutput: whether this is the output layer
-- da: error from next layer (or cost derivative for output layer)
-- network: current network layers (in forward order)
-- forward: forward pass results (in forward order)
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
        pDa = [sum [dZ !! i * weightsT !! i !! j | i <- [0..length dZ - 1]] | j <- [0..length (head weightsT) - 1]]
        
        -- Update neuron function
        updateNeuron (dw, db) n = Neuron 
            (zipWith (\w d -> w - rate * d) (inputWeights n) dw)
            (bias n - rate * db)
            (activate n)
            (activate' n)
backward _ _ _ _ = []

-- | Backpropagate through the entire network
backpropagate :: [[Neuron]] -> Input -> [[Neuron]]
backpropagate network i = reverse $ backward True [cost' (expected i) guess] (reverse network) (reverse forwardNeurons)
    where
        forwardNeurons = forwardNNCalInput (features i) network
        guess = activation (head (last forwardNeurons))

-- | Train on a single input for one iteration
train :: Double -> Int -> [[Neuron]] -> Input -> [[Neuron]]
train epsilon maxIterations network input = 
    case find (\n -> abs (predict input n - expected input) < epsilon) $ take (maxIterations + 1) $ trainUl network input of
        Just result -> result
        Nothing -> last $ take (maxIterations + 1) $ trainUl network input

-- | Unlimited training iterator
trainUl :: [[Neuron]] -> Input -> [[[Neuron]]]
trainUl network input = iterate (\x -> backpropagate x input) network

-- | Train on all inputs once
trainOnAll :: Double -> [[Neuron]] -> [Input] -> [[Neuron]]
trainOnAll epsilon network inputs = 
    foldr (\input net -> train epsilon 1 net input) network inputs

-- | Comprehensive training: multiple epochs over all inputs
trainComprehensive :: Double -> Int -> [[Neuron]] -> [Input] -> [[Neuron]]
trainComprehensive epsilon maxIterations network inputs = 
    foldr (\_ net -> trainOnAll epsilon net inputs) network [1..maxIterations `div` length inputs]
