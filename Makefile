.PHONY: all clean build run

# Fast build using ghc directly (no cabal overhead)
all: build

build: dist/build/simple-nn

dist/build/simple-nn: src/nn.hs
	mkdir -p dist/build
	ghc -O2 -j -o dist/build/simple-nn src/nn.hs

# Alternative: cabal build (slower but more standard)
cabal-build:
	cabal build

run: build
	./dist/build/simple-nn

clean:
	rm -rf dist dist-newstyle simple-nn.cabal
	find . -name "*.o" -delete
	find . -name "*.hi" -delete
