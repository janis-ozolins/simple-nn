module NN.Types where

import Data.List (intercalate)

-- | Represents a neuron with weights, bias, and activation functions
data Neuron = Neuron 
    { inputWeights :: [Double]      -- ^ The input weights
    , bias :: Double                -- ^ b in W * X + b
    , activate :: Double -> Double  -- ^ The activation function
    , activate' :: Double -> Double -- ^ The first derivative of the activation function
    }

-- | Forward calculation result for a neuron
data ForwardNeuronCal = ForwardNeuronCal 
    { calc :: Double      -- ^ Weighted sum (z)
    , activation :: Double -- ^ Activated output (a)
    } deriving Show

-- | Input data with features and expected output
data Input = Input 
    { features :: [Double]
    , expected :: Double
    }

instance Show Neuron where
    show (Neuron w b _ _) = "w=[" ++ intercalate "," (map show w) ++ "], b=" ++ show b
