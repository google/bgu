# Bilateral Guided Upsampling

This is not an official Google product.

## Overview

This is an implementation of **Bilateral Guided Upsampling** as outlined in the SIGGRAPH Asia 2016 [paper](https://people.csail.mit.edu/jiawen/bgu/bgu.pdf) by Jiawen Chen, Andrew Adams, Neal Wadhwa, and Samuel W. Hasinoff.

## Code structure

We include a MATLAB implementation of the slow global optimization algorithm and a [Halide](http://halide-lang.org/) implementation of the fast approximation algorithm. We also provide a trivial GLSL shader for the performing slicing on the GPU. A full OpenGL demo application in on our roadmap.

We thank Elena Adams for the Parrot photo.

### Build instructions (MATLAB)

1. Run MATLAB.
2. `cd src/matlab`
3. `demo`

#### Main driver files:

- `bguFit` Given a (low-resolution) input/output pair, fits an affine model.
- `bguSlice` Given an affine model and a (high-resolution) image, applies the model, producing a (high-resolution) result.
- `testBGU` Test harness that runs `bguFit` followed by `bguSlice`. Stores the results along with the passed-in ground truth into a result struct.
- `showTestResults` Displays the result struct as image figures.
- `runOnFilesnames` Run `testBGU` and `showTestResults` on filenames instead of matrices.
- `demo.m` Runs `runOnFilenames` on the Parrot example in `images`.

------

### Build instructions (Halide, Linux and MacOS)

Our code should build and run on Windows but we have not tested it.

1. `cd src/halide`
2. Download a [Halide distribution](https://github.com/halide/Halide/releases) and unzip it such that you have a directory called `src/halide/halide`.
3. Install **libpng** and **zlib**. On MacOS, we used MacPorts and installed to the default location under `/opt/local`. If you use a different prefix location, edit `Makefile` and change `MACOS_PREFIX_PATH` appropriately.
4. `make`
5. Look at `high_res_out.png` and `high_res_out_gray.png`.

### License

Apache 2.0.