module NN.Network 
    ( calcZ
    , createSigmoidNeuron
    , createSigmoidLayer
    , createNN
    , createNNwGen
    , chunkLayers
    , forwardNeuronCal
    , forwardNNLayerCal
    , forwardNNCal
    , forwardNNCalInput
    , predict
    ) where

import NN.Types
import NN.Activation
import Data.List.Split (chunksOf)
import System.Random (StdGen, getStdGen, randomRs, next)

-- | Calculate weighted sum: W * X + b
calcZ :: [Double] -> [Double] -> Double -> Double
calcZ ws xs b = sum (zipWith (*) ws xs) + b

-- | Create a sigmoid neuron with given weights
createSigmoidNeuron :: [Double] -> Neuron
createSigmoidNeuron weights = Neuron weights 0 sigmoid sigmoid'

-- | Create a layer of sigmoid neurons
-- plSize - previous layer size
createSigmoidLayer :: Int -> [Double] -> [Neuron]
createSigmoidLayer plSize weights = map createSigmoidNeuron (chunksOf plSize weights)

-- | Xavier/Glorot initialization for sigmoid
-- range = [-sqrt(6/fan_in), sqrt(6/fan_in)]
randomSigmoid :: Int -> StdGen -> [Double]
randomSigmoid fanIn g = randomRs (-limit, limit) g
    where limit = sqrt (6.0 / fromIntegral fanIn)

-- | Create neural network with random initialization
createNN :: [Int] -> IO [[Neuron]]
createNN layers = createNNwGen layers <$> getStdGen

-- | Create neural network with given random generator
createNNwGen :: [Int] -> StdGen -> [[Neuron]]
createNNwGen layers g = map (uncurry sl) $ zip (chunkLayers layers) (splitGen g)
    where
        splitGen gen = gen : splitGen (snd (next gen))
        sl x gen = createSigmoidLayer (fst x) (take (uncurry (*) x) (randomSigmoid (fst x) gen))

-- | Helper to create pairs of layer sizes
chunkLayers :: [a] -> [(a,a)]
chunkLayers (p:n:xs) = (p,n) : chunkLayers (n:xs)
chunkLayers _ = []

-- | Forward pass for a single neuron
forwardNeuronCal :: [ForwardNeuronCal] -> Neuron -> ForwardNeuronCal
forwardNeuronCal forward neuron = ForwardNeuronCal z a
    where
        z = calcZ (inputWeights neuron) (map activation forward) (bias neuron)
        a = sigmoid z

-- | Forward pass for a layer
forwardNNLayerCal :: [ForwardNeuronCal] -> [Neuron] -> [ForwardNeuronCal]
forwardNNLayerCal prev neurons = map (forwardNeuronCal prev) neurons

-- | Forward pass through the network (excluding input layer)
forwardNNCal :: [ForwardNeuronCal] -> [[Neuron]] -> [[ForwardNeuronCal]]
forwardNNCal forward (n:ns) = cForward : forwardNNCal cForward ns
    where cForward = forwardNNLayerCal forward n
forwardNNCal _ [] = []

-- | Forward pass starting from input
forwardNNCalInput :: [Double] -> [[Neuron]] -> [[ForwardNeuronCal]]
forwardNNCalInput input neurons = [map (\x -> ForwardNeuronCal x x) input] ++ forwardNNCal (map (\x -> ForwardNeuronCal x x) input) neurons

-- | Make a prediction using the network
predict :: Input -> [[Neuron]] -> Double
predict i network = activation (head (last (forwardNNCalInput (features i) network)))
