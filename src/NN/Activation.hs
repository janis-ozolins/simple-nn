-- | Activation functions and their derivatives.
--
-- Add new activations here (e.g. 'relu', 'tanh') without touching the
-- 'Neuron' or network code.
module NN.Activation
  ( sigmoid
  , sigmoid'
  ) where

sigmoid :: Double -> Double
sigmoid z = 1 / (1 + (exp 1 ** (-z)))

-- | sigmoid derivative, expressed in terms of the post-activation value.
sigmoid' :: Double -> Double
sigmoid' a = a * (1 - a)
