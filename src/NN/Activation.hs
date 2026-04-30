module NN.Activation 
    ( sigmoid
    , sigmoid'
    ) where

-- | Sigmoid activation function
sigmoid :: Double -> Double
sigmoid z = 1 / (1 + (exp 1 ** (-z)))

-- | Derivative of sigmoid function
-- For sigmoid: f'(z) = f(z) * (1 - f(z))
sigmoid' :: Double -> Double
sigmoid' a = a * (1 - a)
