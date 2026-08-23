-- | Core data types for the network and the one primitive computation.
module NN.Neuron
  ( Neuron (..)
  , ForwardNeuronCal (..)
  , Input (..)
  , calcZ
  ) where

import Data.List (intercalate)

-- | A single neuron. The activation function and its derivative are stored
-- per-neuron so different activations could coexist in one network.
data Neuron = Neuron { inputWeights :: [Double]      -- ^ The input weights
                     , bias :: Double                -- ^ b in W * X + b
                     , activate :: Double -> Double  -- ^ The activation function
                     , activate' :: Double -> Double -- ^ The first derivation of the activation function
                     }

-- | Cached forward-pass result for a neuron: the pre-activation (@calc@, i.e. z)
-- and the post-activation (@activation@, i.e. a).
data ForwardNeuronCal = ForwardNeuronCal { calc :: Double, activation :: Double } deriving Show

-- | A labelled training example.
data Input = Input { features :: [Double], expected :: Double }

instance Show Neuron where
  show (Neuron w b _ _) = "w=[" ++ intercalate "," (map show w) ++ "], b=" ++ show b

-- | Weighted sum plus bias: z = W . x + b.
calcZ :: [Double] -> [Double] -> Double -> Double
calcZ ws xs b = sum (zipWith (*) ws xs) + b
