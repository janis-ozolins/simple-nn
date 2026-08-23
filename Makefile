.PHONY: all clean build run test

# Fast build using ghc directly (no cabal overhead)
all: build

SOURCES = $(wildcard src/NN.hs) $(wildcard src/NN/*.hs) src/Main.hs

build: dist/build/simple-nn

dist/build/simple-nn: $(SOURCES)
	mkdir -p dist/build
	ghc -O0 -j -isrc -o dist/build/simple-nn src/Main.hs

# Alternative: cabal build (slower but more standard)
cabal-build:
	cabal build

run: build
	./dist/build/simple-nn

test: dist/build/nn-tests
	./dist/build/nn-tests

dist/build/nn-tests: $(SOURCES) test/Tests.hs
	mkdir -p dist/build
	ghc -O0 -j -isrc -o dist/build/nn-tests test/Tests.hs

clean:
	rm -rf dist dist-newstyle simple-nn.cabal
	find . -name "*.o" -delete
	find . -name "*.hi" -delete
