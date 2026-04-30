
import Data.List(intercalate, transpose, find, foldl')
import Data.List.Split (chunksOf)
import System.Random(StdGen, getStdGen, randomRs, next)
import Data.Maybe      (fromJust)

import Debug.Trace     (trace)

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
createNNwGen layers g = map (uncurry sl) $ zip (chunkLayers layers) (splitGen g)
    where
        splitGen gen = gen : splitGen (snd (next gen))
        sl x gen = createSigmoidLayer (fst x) (take (uncurry (*) x) (randomSigmoid (fst x) gen))

-- forwardNNCalMultiInput :: [[Double]] -> [[Neuron]] -> [[ForwardNeuronCal]]
-- forwardNNCalMultiInput inputs neurons = map (\x -> forwardNNCalInput x neurons) inputs

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
        pDa = [sum [dZ !! i * weightsT !! i !! j | i <- [0..length dZ - 1]] | j <- [0..length (head weightsT) - 1]]
        
        -- Update neuron function
        updateNeuron (dw, db) n = Neuron 
            (zipWith (\w d -> w - rate * d) (inputWeights n) dw)
            (bias n - rate * db)
            (activate n)
            (activate' n)
backward _ da _ _ = []

repli ::  [a] -> Int -> [a]
repli xs n = concat (replicate n xs)

calcNet :: [Double] -> [Double] -> Double
calcNet xs ws = sum $ zipWith (*) xs ws

predict :: Input -> [[Neuron]] -> Double
predict i network = activation (head (last (forwardNNCalInput (features i) network)))

cost :: Double -> Double -> Double
cost expect true = -1 * ((expect * log true) + (1 - expect) * log (1 - true))

-- Cost function derivative for binary cross-entropy (dL/dz for output layer with sigmoid)
cost' :: Double -> Double -> Double
cost' expect pred = pred - expect

forwardNNCalInput :: [Double] -> [[Neuron]] -> [[ForwardNeuronCal]]
forwardNNCalInput input neurons = [map (\x -> ForwardNeuronCal x x) input] ++ (forwardNNCal (map (\x -> ForwardNeuronCal x x) input) neurons)

forwardNNCal :: [ForwardNeuronCal] -> [[Neuron]] -> [[ForwardNeuronCal]]
-- forwardNNCal input neurons = forwardNNCalx (map (\x -> ForwardNeuronCal x 0) input) neurons
forwardNNCal forward (n:ns) = cForward : forwardNNCal cForward ns
    where
        cForward = forwardNNLayerCal forward n
forwardNNCal forward [] = []

forwardNNLayerCal :: [ForwardNeuronCal] -> [Neuron] -> [ForwardNeuronCal]
forwardNNLayerCal prev neurons = map (forwardNeuronCal prev) neurons

forwardNeuronCal :: [ForwardNeuronCal] -> Neuron -> ForwardNeuronCal
forwardNeuronCal forward neuron = ForwardNeuronCal z a
    where
        z = calcZ (inputWeights neuron) (map activation forward) (bias neuron)
        a = sigmoid z

-- backpropagation
-- backpropagate :: [[ForwardNeuronCal]] -> [[Neuron]] -> [[Neuron]]
-- backpropagate f n = 

chunkLayers :: [a] -> [(a,a)]
chunkLayers (p:n:xs) = (p,n) : chunkLayers (n:xs)
chunkLayers _ = []

mmult :: Num a => [[a]] -> [[a]] -> [[a]]
mmult a b = [ [ sum $ zipWith (*) ar bc | bc <- (transpose b) ] | ar <- a ]

-- mmultI :: Num a => [a] -> [a] -> [[a]]
-- mmultI a b = mmult (repeat (length a) b) (repeat (length b) a)

-- network = createNN [2,2,1]
-- guess = predict (Input [1,1] 1) <$> network
-- forwardNeurons = forwardNNCalInput [1,1] <$> network
-- backward <$> (fmap (\x -> [cost' 1 x]) guess) <*> (fmap reverse network) <*> (fmap reverse forwardNeurons)

-- backward <$> (fmap (\x -> [x]) error) <*> (fmap reverse new_network) <*> (fmap reverse forwardNeurons)

-- nameReturn :: IO Double
-- nameReturn = do network <- createNN [2,2,1]
--                 return predict (Input [1,1] 1) network

train :: Double -> Int -> [[Neuron]] -> Input -> [[Neuron]]
train epsilon maxIterations network input = 
    case find (findGradient epsilon input) $ take (maxIterations + 1) $ trainUl network input of
        Just result -> result
        Nothing -> last $ take (maxIterations + 1) $ trainUl network input

findGradient :: Double -> Input -> [[Neuron]] -> Bool
findGradient epsilon input network = abs (predict input network - expected input) < epsilon

trainUl :: [[Neuron]] -> Input -> [[[Neuron]]]
trainUl network samples = iterate (\x -> backpropagate x samples) network

backpropagate :: [[Neuron]] -> Input -> [[Neuron]]
-- backpropagation starts from end so that is why reversing is needed
backpropagate network i = reverse $ backward True [cost' (expected i) guess] (reverse network) (reverse forwardNeurons)
    where
        forwardNeurons = forwardNNCalInput (features i) network
        guess = activation (head (last forwardNeurons))


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
    let trainedNetwork = trainComprehensive epsilon 100000 bestNetwork xorInputs
    
    -- Final evaluation
    putStrLn "\n=== Final Evaluation ==="
    mapM_ (testXORInput trainedNetwork) xorInputs

    where
        testNetworkSize epsilon maxIterations xorInputs size = do
            putStrLn $ "\n--- Testing architecture: " ++ show size ++ " ---"
            network <- createNN size
            
            -- Train on all patterns
            let trainedNetwork = trainOnAll epsilon maxIterations network xorInputs
            let totalError = sum [abs (predict input trainedNetwork - expected input) | (input, _) <- xorInputs]
            let avgError = totalError / fromIntegral (length xorInputs)
            
            let result = "Architecture " ++ show size ++ ": Avg Error = " ++ show avgError
            putStrLn result
            return result
        
        trainOnAll epsilon maxIterations network inputs = 
            foldl' (\net _ -> trainAllOnce epsilon net inputs) network [1..maxIterations]
        
        trainAllOnce epsilon network inputs = 
            foldl' (\net (input, _) -> train epsilon 1 net input) network inputs
        
        trainComprehensive epsilon maxIterations network xorInputs = 
            foldl' (\net _ -> trainOnAll epsilon 1 net xorInputs) network [1..maxIterations `div` length xorInputs]
            where
                trainOnAll epsilon iterations net inputs = 
                    foldl' (\n (input, _) -> train epsilon iterations n input) net inputs
        
        testXORInput network (input, description) = do
            let prediction = predict input network
            let error = abs (prediction - expected input)
            let success = if error < 0.2 then "✓" else "✗"
            putStrLn $ success ++ " " ++ description ++ " -> Predicted: " ++ show prediction ++ 
                      " (Expected: " ++ show (expected input) ++ ", Error: " ++ show error ++ ")"
    
