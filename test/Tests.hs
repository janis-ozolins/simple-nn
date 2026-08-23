module Main (main) where

import NN
import System.Random (StdGen, mkStdGen, randomRs, splitGen)
import System.Exit (exitWith, ExitCode(..))
import Control.Monad (when)
import Data.List (foldl', genericLength)

-- ---------------------------------------------------------------------------
-- Minimal test harness (no external test dependencies)
-- ---------------------------------------------------------------------------

type TestName = String

assertEqual :: (Eq a, Show a) => TestName -> a -> a -> IO Bool
assertEqual name expected actual =
  if expected == actual
    then ok name
    else fail_ name $ "expected " ++ show expected ++ " got " ++ show actual

assertBool :: TestName -> Bool -> IO Bool
assertBool name cond =
  if cond then ok name else fail_ name "condition was False"

assertClose :: TestName -> Double -> Double -> Double -> IO Bool
assertClose name tol expected actual =
  if abs (expected - actual) <= tol
    then ok name
    else fail_ name $ "expected " ++ show expected ++ " within " ++ show tol
                ++ " got " ++ show actual

ok :: TestName -> IO Bool
ok name = putStrLn ("  ok   " ++ name) >> return True

fail_ :: TestName -> String -> IO Bool
fail_ name why = putStrLn ("  FAIL " ++ name ++ ": " ++ why) >> return False

-- ---------------------------------------------------------------------------
-- Unit tests: exact values for pure functions
-- ---------------------------------------------------------------------------

unitTests :: [IO Bool]
unitTests =
  [ assertClose "sigmoid 0 == 0.5" 1e-12 0.5 (sigmoid 0)
  , assertClose "sigmoid' 0.5 == 0.25" 1e-12 0.25 (sigmoid' 0.5)
  , assertClose "calcZ [1,1] [2,3] 0 == 5" 1e-12 5.0 (calcZ [1,1] [2,3] 0)
  , assertClose "calcZ with bias" 1e-12 7.5 (calcZ [1,1] [2,3] 2.5)
  , assertClose "cost 1 0.9 == -log 0.9" 1e-9 (-log 0.9) (cost 1 0.9)
  , assertClose "cost 0 0.1 == -log 0.9" 1e-9 (-log 0.9) (cost 0 0.1)
  , assertClose "cost' 1 0.5 == -0.5" 1e-12 (-0.5) (cost' 1 0.5)
  , assertClose "cost' 0 0.5 == 0.5" 1e-12 0.5 (cost' 0 0.5)
  , assertClose "cost' 1 0.9 == -0.1" 1e-12 (-0.1) (cost' 1 0.9)
  , assertEqual "chunkLayers [2,3,1]" [(2,3),(3,1)] (chunkLayers [2,3,1])
  , assertEqual "chunkLayers [5] == []" [] (chunkLayers [5 :: Int])
  , assertEqual "chunkLayers [] == []" [] (chunkLayers [] :: [(Int,Int)])
  , let layer = createSigmoidLayer 2 [1,2,3,4]
    in assertBool "createSigmoidLayer: 2 neurons" (length layer == 2)
  , let layer = createSigmoidLayer 2 [1,2,3,4]
    in assertBool "createSigmoidLayer: each neuron has 2 weights"
                 (map (length . inputWeights) layer == [2,2])
  , let layer = createSigmoidLayer 2 [1,2,3,4]
    in assertBool "createSigmoidLayer: biases initialized to 0"
                 (all (== 0) (map bias layer))
  ]

-- ---------------------------------------------------------------------------
-- Property tests: randomized, seeded for determinism
-- ---------------------------------------------------------------------------

-- | Run a predicate over n random values drawn from a range using a seed.
property :: TestName -> Int -> (Double -> Bool) -> Double -> Double -> StdGen -> IO Bool
property name n pred_ lo hi g0 =
  let xs = take n (randomRs (lo, hi) g0 :: [Double])
      bad = filter (not . pred_) xs
  in if null bad
       then ok name
       else fail_ name $ "counterexamples: " ++ show (take 3 bad)

propTests :: [IO Bool]
propTests =
  [ property "sigmoid x in (0,1)" 1000 (\x -> sigmoid x > 0 && sigmoid x < 1) (-20) 20 (mkStdGen 1)
  , property "sigmoid' a == a*(1-a)" 1000 (\a -> abs (sigmoid' a - a*(1-a)) < 1e-12) 0 1 (mkStdGen 2)
  , property "cost y p >= 0" 1000 (\p -> cost 1 p >= 0 && cost 0 p >= 0) 0.001 0.999 (mkStdGen 3)
  , property "cost' y p == p - y" 1000 (\p -> abs (cost' 1 p - (p - 1)) < 1e-12
                                          && abs (cost' 0 p - p) < 1e-12) 0.001 0.999 (mkStdGen 4)
  , forwardShapeProp
  , predictRangeProp
  , backwardShapeProp
  ]

-- | Forward pass preserves layer sizes: result has length (net layers + 1),
-- and each row's width matches the spec (input width, then each layer's neuron count).
forwardShapeProp :: IO Bool
forwardShapeProp =
  let layers = [2,3,1]
      g = mkStdGen 10
      net = createNNwGen layers g
      input = [0.5, 0.5]
      fwd = forwardNNCalInput input net
      widths = map length fwd
      -- createNNwGen builds one layer per adjacent pair (chunkLayers),
      -- each layer's neuron count is the second element of the pair.
      expectedWidths = length input : map snd (chunkLayers layers)
  in assertEqual "forward preserves layer widths" expectedWidths widths

-- | predict output is always in [0,1] (sigmoid output layer).
predictRangeProp :: IO Bool
predictRangeProp =
  let layers = [2,4,1]
      g0 = mkStdGen 11
      net = createNNwGen layers g0
      g1 = snd (splitGen g0)
      xs = take 200 (randomRs (-5, 5) g1 :: [Double])
      preds = [predict (Input [x, 1 - x] 0) net | x <- xs]
  in assertBool "predict in [0,1]" (all (\p -> p >= 0 && p <= 1) preds)

-- | backward preserves the network shape (layer count, neuron count, weight count).
-- backward is called on (reverse net) and returns layers in that reversed order,
-- so we compare against (reverse updated) to match the original ordering.
backwardShapeProp :: IO Bool
backwardShapeProp =
  let layers = [2,3,1]
      g = mkStdGen 12
      net = createNNwGen layers g
      input = [0.5, 0.5]
      fwd = forwardNNCalInput input net
      da = [0.1]  -- output layer size = 1
      updated = backward True da (reverse net) (reverse fwd)
      shape net_ = (length net_, map length net_, map (length . inputWeights) (concat net_))
  in assertEqual "backward preserves network shape" (shape net) (shape (reverse updated))

-- ---------------------------------------------------------------------------
-- Integration: deterministic training actually learns
-- ---------------------------------------------------------------------------

integrationTests :: [IO Bool]
integrationTests =
  [ singleInputLearns
  , xorConverges
  ]

-- | Training a single input must reduce the prediction error.
singleInputLearns :: IO Bool
singleInputLearns =
  let g = mkStdGen 42
      net = createNNwGen [2,4,1] g
      input = Input [0,1] 1
      initialErr = abs (predict input net - expected input)
      trained = train 0.01 5000 net input
      finalErr = abs (predict input trained - expected input)
  in do
    r1 <- assertBool "single-input: initial error is non-trivial" (initialErr > 0.01)
    r2 <- assertBool "single-input: training reduces error" (finalErr < initialErr)
    return (r1 && r2)

-- | Full XOR pipeline: after batch training, average error drops below threshold.
xorConverges :: IO Bool
xorConverges =
  let g = mkStdGen 7
      net0 = createNNwGen [2,4,1] g
      patterns = [ Input [0,0] 0, Input [0,1] 1, Input [1,0] 1, Input [1,1] 0 ]
      -- 1500 sweeps over all 4 patterns, 100 gradient steps each
      trained = foldl' (\n i -> train 0.05 100 n i)
                       net0
                       (concat (replicate 1500 patterns))
      avgErr = sum [abs (predict i trained - expected i) | i <- patterns]
               / genericLength patterns
  in assertBool ("XOR converges: avg error < 0.2 (got " ++ show avgErr ++ ")")
                (avgErr < 0.2)

-- ---------------------------------------------------------------------------
-- Runner
-- ---------------------------------------------------------------------------

main :: IO ()
main = do
  putStrLn "Running NN tests..."
  let groups :: [(String, [IO Bool])]
      groups =
        [ ("Unit tests", unitTests)
        , ("Property tests", propTests)
        , ("Integration tests", integrationTests)
        ]
  results <- mapM runGroup groups
  let totalPass = sum [p | (p, _) <- results]
      totalFail = sum [f | (_, f) <- results]
  putStrLn "\n----------------------------------------"
  putStrLn $ "Total: " ++ show (totalPass + totalFail)
                 ++ "  Passed: " ++ show totalPass
                 ++ "  Failed: " ++ show totalFail
  when (totalFail > 0) (exitWith (ExitFailure 1))
  where
    runGroup (name, tests) = do
      putStrLn $ "\n[" ++ name ++ "]"
      rs <- sequence tests
      let p = length (filter id rs)
          f = length (filter not rs)
      return (p, f)
