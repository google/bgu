// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http ://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "fit_and_slice_1x4.h"
#include "fit_and_slice_3x4.h"
#include "fit_only_1x4.h"
#include "fit_only_3x4.h"

#include "benchmark.h"
#include "HalideBuffer.h"
#include "halide_image_io.h"

using namespace Halide::Tools;

int main(int argc, char **argv) {

    if (argc < 7) {
        printf("Usage: %s low_res_in.png low_res_out_gray.png high_res_in.png "
                "high_res_out_gray.png spatial_sigma range_bins\n",
                argv[0]);
        return 1;
    }

    Halide::Buffer<float> low_res_in = load_image(argv[1]);
    printf("Loaded low_res_in: %d x %d x %d\n",
            low_res_in.width(),
            low_res_in.height(),
            low_res_in.channels());

    if (low_res_in.channels() != 3) {
        // TODO: write bgu_1x2 to support grayscale input.
        fprintf(stderr, "low_res_in must have 3 channels.\n");
        return 2;
    }

    Halide::Buffer<float> low_res_out = load_image(argv[2]);
    printf("Loaded low_res_out: %d x %d x %d\n",
            low_res_out.width(),
            low_res_out.height(),
            low_res_out.channels());

    if (low_res_out.channels() != 1 &&
        low_res_out.channels() != 3) {
        fprintf(stderr, "low_res_out must have only 1 or 3 channels.\n");
        return 3;
    }

    if (low_res_in.width() != low_res_out.width() ||
        low_res_in.height() != low_res_out.height()) {
        fprintf(stderr, "low_res_in and low_res_out must have the same "
                "extent.\n");
        return 4;
    }

    Halide::Buffer<float> high_res_in = load_image(argv[3]);
    printf("Loaded high_res_in: %d x %d x %d\n",
            high_res_in.width(),
            high_res_in.height(),
            high_res_in.channels());

    if (high_res_in.channels() != low_res_in.channels()) {
        fprintf(stderr, "low_res_in and high_res_in must have the same number "
                "of channels.\n");
        return 5;
    }

    Halide::Buffer<float> high_res_out(high_res_in.width(),
                                       high_res_in.height(),
                                       low_res_out.channels());
    float s_sigma = atoi(argv[5]);
    float r_sigma = 1.0f / atoi(argv[6]);

    // Fit the curves and slice out the result.
    if (high_res_out.channels() == 1) {
        fit_and_slice_1x4(r_sigma, s_sigma,
                          low_res_in, low_res_out,
                          high_res_in, high_res_out);
    } else {
        fit_and_slice_3x4(r_sigma, s_sigma,
                          low_res_in, low_res_out,
                          high_res_in, high_res_out);
    }

    save_image(high_res_out, argv[4]);

    // You'd normally slice out the result using a shader. Check the
    // runtime of curve fitting alone.
    int grid_w = low_res_in.width() / s_sigma;
    grid_w = ((grid_w+7)/8)*8;
    int grid_h = low_res_in.height() / s_sigma;
    int grid_z = round(1.0f/r_sigma);
    int grid_c = low_res_out.channels() * (low_res_in.channels() + 1);
    Halide::Buffer<float> coeffs(grid_w, grid_h, grid_z, grid_c);

    double min_t;
    if (high_res_out.channels() == 1) {
        min_t = benchmark(10, 10, [&]() {
            fit_only_1x4(r_sigma, s_sigma,
                         low_res_in, low_res_out, high_res_in,
                         coeffs);
        });
    } else {
        min_t = benchmark(10, 10, [&]() {
            fit_only_3x4(r_sigma, s_sigma,
                         low_res_in, low_res_out, high_res_in,
                         coeffs);
        });
    }
    printf("Time for fitting: %g ms.\n", min_t * 1e3);

    return 0;
}
