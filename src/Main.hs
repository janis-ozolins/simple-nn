module Main where

import NN.Types
import NN.Network
import NN.Training
import Data.Foldable (mapM_)

main :: IO ()
main = do
    putStrLn "=== XOR Function Learning Demo ==="
    putStrLn "Testing different network architectures for XOR problem..."
    
    -- Test different network sizes
    let networkSizes = [[2, 2, 1], [2, 4, 1], [2, 6, 1]]
    let xorInputs = [
            (Input [0,0] 0, "[0,0] -> 0"),
            (Input [0,1] 1, "[0,1] -> 1"),
            (Input [1,0] 1, "[1,0] -> 1"),
            (Input [1,1] 0, "[1,1] -> 0")
            ]
    
    let epsilon = 0.1  -- Target error threshold
    let maxIterations = 100000  -- More iterations for better learning
    
    putStrLn "XOR Truth Table:"
    mapM_ (putStrLn . snd) xorInputs
    putStrLn "\nTesting network architectures:"
    
    -- Test each network size
    results <- mapM (testNetworkSize epsilon maxIterations xorInputs) networkSizes
    
    -- Compare results
    putStrLn "\n=== Architecture Comparison ==="
    mapM_ putStrLn results
    
    -- Test the best architecture with more comprehensive training
    putStrLn "\n=== Comprehensive Training on Best Architecture ==="
    let bestNetworkSize = [2, 4, 1]  -- Start with this as best
    bestNetwork <- createNN bestNetworkSize
    putStrLn $ "Using architecture: " ++ show bestNetworkSize
    
    -- Train on all patterns multiple times (simple batch training)
    let trainedNetwork = trainComprehensive epsilon 100000 bestNetwork (map fst xorInputs)
    
    -- Final evaluation
    putStrLn "\n=== Final Evaluation ==="
    mapM_ (testXORInput trainedNetwork) xorInputs

    where
        testNetworkSize epsilon maxIterations xorInputs size = do
            putStrLn $ "\n--- Testing architecture: " ++ show size ++ " ---"
            network <- createNN size
            
            -- Train on all patterns
            let trainedNetwork = trainOnAll epsilon maxIterations network (map fst xorInputs)
            let totalError = sum [abs (predict input trainedNetwork - expected input) | (input, _) <- xorInputs]
            let avgError = totalError / fromIntegral (length xorInputs)
            
            let result = "Architecture " ++ show size ++ ": Avg Error = " ++ show avgError
            putStrLn result
            return result

        testXORInput network (input, description) = do
            let prediction = predict input network
            let error = abs (prediction - expected input)
            let success = if error < 0.2 then "[OK]" else "[FAIL]"
            putStrLn $ success ++ " " ++ description ++ " -> Predicted: " ++ show prediction ++ 
                      " (Expected: " ++ show (expected input) ++ ", Error: " ++ show error ++ ")"
