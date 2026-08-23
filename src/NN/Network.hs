-- | Network construction and weight initialization.
module NN.Network
  ( createSigmoidNeuron
  , createSigmoidLayer
  , createNN
  , createNNwGen
  , randomSigmoid
  , chunkLayers
  ) where

import Data.List.Split (chunksOf)
import System.Random (StdGen, getStdGen, randomRs, splitGen)

import NN.Activation (sigmoid, sigmoid')
import NN.Neuron (Neuron (..))

createSigmoidNeuron :: [Double] -> Neuron
createSigmoidNeuron weights = Neuron weights 0 sigmoid sigmoid'

-- plSize - previous layer size
createSigmoidLayer :: Int -> [Double] -> [Neuron]
createSigmoidLayer plSize weights = map createSigmoidNeuron (chunksOf plSize weights)

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

-- | Pair up adjacent layer sizes: [2,3,1] -> [(2,3),(3,1)].
chunkLayers :: [a] -> [(a,a)]
chunkLayers (p:n:xs) = (p,n) : chunkLayers (n:xs)
chunkLayers _ = []
