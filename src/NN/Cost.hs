-- | Loss functions and their derivatives.
--
-- Currently binary cross-entropy. Add new losses (MSE, hinge, ...) here.
module NN.Cost
  ( cost
  , cost'
  ) where

-- | Binary cross-entropy loss.
cost :: Double -> Double -> Double
cost expect true = - (expect * log true + (1 - expect) * log (1 - true))

-- | Derivative of the BCE loss w.r.t. the output layer pre-activation
-- (combined with sigmoid): dL/dz = pred - expect.
cost' :: Double -> Double -> Double
cost' expect pred = pred - expect
