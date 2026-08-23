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

import Data.List(intercalate, transpose, find, foldl')
import Data.List.Split (chunksOf)
import Data.Maybe(fromMaybe, listToMaybe)
import System.Random(StdGen, getStdGen, randomRs, splitGen)

data Neuron = Neuron { inputWeights :: [Double]      -- ^ The input weights
                     , bias :: Double                -- ^ b in W * X + b
                     , activate :: Double -> Double  -- ^ The activation function
                     , activate' :: Double -> Double -- ^ The first derivation of the activation function
                     }

data ForwardNeuronCal = ForwardNeuronCal { calc :: Double, activation :: Double } deriving Show

data Input = Input { features :: [Double], expected :: Double }

instance Show Neuron where
  show (Neuron w b _ _) = "w=[" ++ intercalate "," (map show w) ++ "], b=" ++ show b

sigmoid :: Double -> Double
sigmoid z = 1 / (1 + (exp 1 ** (-z)))

-- sigmoid derivative
sigmoid' :: Double -> Double
sigmoid' a = a * (1 - a)

createSigmoidNeuron :: [Double] -> Neuron
createSigmoidNeuron weights = Neuron weights 0 sigmoid sigmoid'

-- plSize - previous layer size
createSigmoidLayer :: Int -> [Double] -> [Neuron]
createSigmoidLayer plSize weights = map createSigmoidNeuron (chunksOf plSize weights)

calcZ :: [Double] -> [Double] -> Double -> Double
calcZ ws xs b = sum (zipWith (*) ws xs) + b

createNN :: [Int] -> IO [[Neuron]]
createNN layers = createNNwGen layers <$> getStdGen

-- For random initialization of weights W on neuron connections,
-- b doesn't need it, is initialized as 0
-- Xavier/Glorot initialization for sigmoid: range = [-sqrt(6/fan_in), sqrt(6/fan_in)]
randomSigmoid :: Int -> StdGen -> [Double]
randomSigmoid fanIn g = randomRs (-limit, limit) g
    where limit = sqrt (6.0 / fromIntegral fanIn)

createNNwGen :: [Int] -> StdGen -> [[Neuron]]
createNNwGen layers g = map (uncurry sl) $ zip (chunkLayers layers) (splitGens g)
    where
        splitGens gen = gen : splitGens (snd (splitGen gen))
        sl x gen = createSigmoidLayer (fst x) (take (uncurry (*) x) (randomSigmoid (fst x) gen))

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

predict :: Input -> [[Neuron]] -> Double
predict i network = maybe 0 activation (listToMaybe (last (forwardNNCalInput (features i) network)))

cost :: Double -> Double -> Double
cost expect true = - (expect * log true + (1 - expect) * log (1 - true))

-- Cost function derivative for binary cross-entropy (dL/dz for output layer with sigmoid)
cost' :: Double -> Double -> Double
cost' expect pred = pred - expect

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

chunkLayers :: [a] -> [(a,a)]
chunkLayers (p:n:xs) = (p,n) : chunkLayers (n:xs)
chunkLayers _ = []

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
    
