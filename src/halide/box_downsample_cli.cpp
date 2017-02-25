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
#include "box_downsample_halide.h"

#include <cassert>
#include <cmath>
#include <string>

#include "HalideBuffer.h"
#include "halide_image_io.h"

using namespace Halide::Tools;

inline int roundUp(float x) {
    return static_cast<int>(ceilf(x));
}

// Given input image extents and an integer downsample factor, compute the
// extents of the downsampled image such that every pixel in the downsampled
// image covers *at least* a downsample_factor x downsample_factor area of the
// input.
//
// If the input extents do not evenly divide downsample_factor, then the right
// and bottom edges of the downsampled output will extend past the edge of the
// input.
inline void ConservativeExtentsForDownsampling(
    int input_width, int input_height,
    int downsample_factor,
    int& output_width, int& output_height) {
    assert(downsample_factor >= 1);
    output_width = roundUp(static_cast<float>(input_width) / downsample_factor);
    output_height = roundUp(static_cast<float>(input_height) / downsample_factor);
}

int main(int argc, char* argv[]) {

    if (argc < 4) {
        printf("Usage: %s <input.png> <downsample_factor> <output.png> [u8]\n",
                argv[0]);
        printf("\n");
        printf("box_downsample_cli conservativesly downsamples the input by "
                "an integer downsample factor such that every pixel in the "
                "output image covers an area *at least* downsample_factor x "
                "downsample_factor in the input."
                "\n\n"
                "The input png will be converted to unsigned 16-bit, box "
                "downsampled, and saved as unsigned 16-bit. "
                "\n\n"
                "If the optional last argument is \"u8\", it will be saved "
                "as unsigned 8-bit instead.\n");
        return 1;
    }

    bool save_u8 = false;
    if (argc > 4) {
        std::string arg4 = argv[4];
        if (arg4 == "u8") {
            save_u8 = true;
        }
    }
    if (save_u8) {
        printf("Writing output as 8-bit.\n");
    } else {
        printf("Writing output as 16-bit.\n");
    }

    Halide::Buffer<uint16_t> input = load_image(argv[1]);
    printf("Loaded input: %d x %d x %d\n",
            input.width(),
            input.height(),
            input.channels());

    int downsample_factor = atoi(argv[2]);
    printf("downsample_factor = %d\n", downsample_factor);
    if (downsample_factor < 2 || downsample_factor > 16) {
        fprintf(stderr,
                "downsample_factor must be an integer in [2, 16].\n");
    }

    int output_width;
    int output_height;
    ConservativeExtentsForDownsampling(input.width(),
                                       input.height(),
                                       downsample_factor,
                                       output_width,
                                       output_height);
    printf("Downsampled output will be %d x %d x %d\n",
            output_width, output_height, input.channels());
    Halide::Buffer<uint16_t> output_u16(output_width,
                                        output_height,
                                        input.channels());

    int ret = box_downsample_halide(input, downsample_factor, output_u16);
    if (ret != 0) {
        fprintf(stderr, "Error executing box_downsample_halide kernel.\n");
        return 2;
    }

    Halide::Buffer<uint8_t> output_u8;
    if (save_u8) {
        printf("Converting downsampled image to uint8_t.\n");
#if 0
    // TODO: once a new release of Halide is available, use this simpler
    // version.
        output_u8 = Halide::Buffer<uint8_t>::make_with_shape_of(output_u16);
        output_u8.for_each_value([&](uint8_t& dst, int16_t src) {
                Halide::Tools::Internal::convert(src, dst);
        }, output_u16);
#else
        output_u8 = Halide::Buffer<uint8_t>(output_u16.width(),
                                            output_u16.height(),
                                            output_u16.channels());
        output_u8.for_each_element([&](int x, int y, int c) {
            Halide::Tools::Internal::convert(output_u16(x, y, c),
                                             output_u8(x, y, c));
        });
    }
#endif

    printf("Writing: %s...", argv[3]);

    bool save_ok;
    if (save_u8) {
        save_ok = save_png(output_u8, argv[3]);
    } else {
        save_ok = save_png(output_u16, argv[3]);
    }

    if (save_ok) {
        printf("done.\n");
    } else {
        printf("FAILED.\n");
    }


    return 0;
}
