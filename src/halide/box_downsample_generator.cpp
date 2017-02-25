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
#include "Halide.h"

using Halide::_;
using Halide::ConciseCasts::i16;
using Halide::ConciseCasts::i32;

class BoxDownsample : public Halide::Generator<BoxDownsample> {
 public:
  Halide::ImageParam input_{Halide::UInt(16), 3, "input"};
  Halide::Param<int> downsample_factor_{"downsample_factor", 8};

  Func build() {
    // Reduce the resolution of the input image, low-pass filtering with a
    // simple box filter.
    Func clamped_input("clamped_input");
    clamped_input = Halide::BoundaryConditions::repeat_edge(input_);

    Expr patch_area = downsample_factor_ * downsample_factor_;

    Var x("x"), y("y");
    Func output("output");
    RDom r(0, downsample_factor_, 0, downsample_factor_);
    Expr patch_sum = sum(i32(clamped_input(x * downsample_factor_ + r.x,
                                           y * downsample_factor_ + r.y, _)));
    Expr patch_mean = (patch_sum + patch_area / 2) / patch_area;
    output(x, y, _) = i16(patch_mean);

    // TODO: This code can use further optimization:
    // For larger downsample factors, can vectorize over r.x.
    constexpr int kParallelTaskSize = 8;
    int vec_size = get_target().natural_vector_size<int16_t>();
    output.parallel(y, kParallelTaskSize).vectorize(x, vec_size);

    // Specialize the pipeline for reasonable powers of 2.
    for (int i = 2; i <= 16; i *= 2) {
      output.specialize(downsample_factor_ == i);
    }

    return output;
  }
};

auto register_BoxDownsample =
    Halide::RegisterGenerator<BoxDownsample>("box_downsample_halide");
