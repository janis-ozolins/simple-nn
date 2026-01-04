
import Data.List(intercalate, transpose, find)
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
randomSigmoid :: StdGen -> [Double]
randomSigmoid = randomRs (-1, 1)

createNNwGen :: [Int] -> StdGen -> [[Neuron]]
createNNwGen layers g = map sl $ chunkLayers layers
    where
        sl x = createSigmoidLayer (fst x) (take (uncurry (*) x) rs)
        rs = randomSigmoid g

-- forwardNNCalMultiInput :: [[Double]] -> [[Neuron]] -> [[ForwardNeuronCal]]
-- forwardNNCalMultiInput inputs neurons = map (\x -> forwardNNCalInput x neurons) inputs

rate :: Double
rate = 0.1

backward :: [Double] -> [[Neuron]] -> [[ForwardNeuronCal]] -> [[Neuron]]
backward da (l:ls) (f:fs) | trace ("backward " ++ show da ++ ", " ++ show l ++ ", " ++ show f ++ "\n") False = undefined
backward da (l:ls) (f:fp:fs) = zipWith (\u n -> Neuron (zipWith (\w d -> w - rate * d) (inputWeights n) (fst u)) (bias n - rate * snd u) (activate n) (activate' n)) (zip dW dB) l : backward pDa ls (fp:fs)
    where
        dZ = zipWith (*) da (map (sigmoid' . calc) f)
        dW = zipWith (\z as -> map (* z) as) dZ (repeat (map activation fp))
        dB = dZ
        pDa = zipWith (\z ws -> sum (map (* z) ws)) (repli dZ 5) (transpose (map inputWeights l))
backward da _ _ = []

repli ::  [a] -> Int -> [a]
repli xs n = concat (replicate n xs)

calcNet :: [Double] -> [Double] -> Double
calcNet xs ws = sum $ zipWith (*) xs ws

predict :: Input -> [[Neuron]] -> Double
predict i network = activation (head (last (forwardNNCalInput (features i) network)))

cost :: Double -> Double -> Double
cost expect true = -1 * ((expect * log true) + (1 - expect) * log (1 - true))

cost' :: Double -> Double -> Double
cost' expect true = -1 * (expect / true) - (1 - expect) / (1 - true)

forwardNNCalInput :: [Double] -> [[Neuron]] -> [[ForwardNeuronCal]]
forwardNNCalInput input neurons = [map (\x -> ForwardNeuronCal 0 x) input] ++ (forwardNNCal (map (\x -> ForwardNeuronCal x 0) input) neurons)

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
        z = calcZ (inputWeights neuron) (map calc forward) (bias neuron)
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

train :: Double -> [[Neuron]] -> Input -> [[Neuron]]
train epsilon network input = fromJust $ find (findGradient epsilon input) $ trainUl network input

findGradient :: Double -> Input -> [[Neuron]] -> Bool
findGradient _ input network | trace ("findGradient " ++ (show $ predict input network) ++ "\n") False = undefined
findGradient epsilon input network = predict input network > 1 - epsilon

trainUl :: [[Neuron]] -> Input -> [[[Neuron]]]
trainUl network samples = iterate (\x -> backpropagate x samples) network

backpropagate :: [[Neuron]] -> Input -> [[Neuron]]
-- backpropagation starts from end so that is why rersing is needed
backpropagate network _ | trace ("backpropagate " ++ show network ++ "\n") False = undefined
backpropagate network i = reverse $ backward [cost' 1 guess] (reverse network) (reverse forwardNeurons)
    where
        forwardNeurons = forwardNNCalInput (features i) network
        guess = activation (head (last forwardNeurons))


main :: IO ()
main = do
    putStrLn "Creating neural network with structure [2, 2, 1]..."
    network <- createNN [2, 2, 1]
    putStrLn "Original network:"
    mapM_ (putStrLn . show) network
    
    let epsilon = 1 - 0.7310585786300048
    let testInput = Input [1,1] 1
    putStrLn $ "Training with epsilon: " ++ show epsilon
    putStrLn $ "Training input: " ++ show (features testInput) ++ " -> " ++ show (expected testInput)
    
    let trainedNetwork = train epsilon network testInput
    putStrLn "\nTraining complete!"
    putStrLn "Trained network:"
    mapM_ (putStrLn . show) trainedNetwork
    
    let prediction = predict testInput trainedNetwork
    putStrLn $ "\nPrediction for input [1,1]: " ++ show prediction
    putStrLn $ "Expected: " ++ show (expected testInput)
    putStrLn $ "Cost: " ++ show (cost (expected testInput) prediction)