#include <vector>

#include "third_party/halide/halide.h"

using Halide::Expr;
using Halide::Func;
using Halide::max;
using Halide::undef;
using Halide::Var;
using Halide::BoundaryConditions::repeat_edge;
using Halide::ConciseCasts::f32;
using Halide::ConciseCasts::i16_sat;
using Halide::ConciseCasts::i32;

namespace bgu {

Var x("x"), y("y"), z("z"), c("c");

// A class to hold a matrix of Halide Exprs.
template <int rows, int cols>
struct Matrix {
  Expr exprs[rows][cols];

  Expr operator()(int i, int j) const { return exprs[i][j]; }

  Expr& operator()(int i, int j) { return exprs[i][j]; }

  void dump() {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        std::cout << exprs[i][j];
        if (j < cols - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "\n";
    }
  }
};

// Matrix-matrix multiply.
template <int R, int S, int T>
Matrix<R, T> mat_mul(const Matrix<R, S>& A, const Matrix<S, T>& B) {
  Matrix<R, T> result;
  for (int r = 0; r < R; r++) {
    for (int t = 0; t < T; t++) {
      result(r, t) = 0.0f;
      for (int s = 0; s < S; s++) {
        result(r, t) += A(r, s) * B(s, t);
      }
    }
  }
  return result;
}

// Solve Ax = b at each x, y, z. Compute the result at the given Func and Var.
template <int M, int N>
Matrix<M, N> solve(Matrix<M, M> A, Matrix<M, N> b, Func compute, Var at,
                   bool apply_schedule) {
  // Put the input matrices in a Func to do the Gaussian elimination.
  Var vi, vj;
  Func f;
  f(x, y, z, vi, vj) = undef<float>();
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      f(x, y, z, i, j) = A(i, j);
    }
    for (int j = 0; j < N; j++) {
      f(x, y, z, i, j + M) = b(i, j);
    }
  }

  // Eliminate lower left.
  for (int k = 0; k < M - 1; k++) {
    for (int i = k + 1; i < M; i++) {
      f(x, y, z, -1, 0) = f(x, y, z, i, k) / f(x, y, z, k, k);
      for (int j = k + 1; j < M + N; j++) {
        f(x, y, z, i, j) -= f(x, y, z, k, j) * f(x, y, z, -1, 0);
      }
      f(x, y, z, i, k) = 0.0f;
    }
  }

  // Eliminate upper right.
  for (int k = M - 1; k > 0; k--) {
    for (int i = 0; i < k; i++) {
      f(x, y, z, -1, 0) = f(x, y, z, i, k) / f(x, y, z, k, k);
      for (int j = k + 1; j < M + N; j++) {
        f(x, y, z, i, j) -= f(x, y, z, k, j) * f(x, y, z, -1, 0);
      }
      f(x, y, z, i, k) = 0.0f;
    }
  }

  // Divide by diagonal and put it in the output matrix.
  for (int i = 0; i < M; i++) {
    f(x, y, z, i, i) = 1.0f / f(x, y, z, i, i);
    for (int j = 0; j < N; j++) {
      b(i, j) = f(x, y, z, i, j + M) * f(x, y, z, i, i);
    }
  }

  if (apply_schedule) {
    for (int i = 0; i < f.num_update_definitions(); i++) {
      f.update(i).vectorize(x);
    }

    f.compute_at(compute, at);
  }
  return b;
}

template <int N, int M>
Matrix<M, N> transpose(const Matrix<N, M>& in) {
  Matrix<M, N> out;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      out(j, i) = in(i, j);
    }
  }
  return out;
}

Expr pack_channels(Var c, std::vector<Expr> exprs) {
  Expr e = exprs.back();
  for (int i = static_cast<int>(exprs.size()) - 2; i >= 0; i--) {
    e = select(c == i, exprs[i], e);
  }
  return e;
}

// Clamps the input to [input_black_level, input_white_level], then maps it to a
// float in [0,1].
Expr normalize_to_float(Expr input,
                        Expr input_black_level, Expr input_white_level) {
  Expr inverse_input_range = 1.0f / (input_white_level - input_black_level);
  Expr input_clamped = clamp(input, input_black_level, input_white_level);
  return (input_clamped - input_black_level) * inverse_input_range;
}

// Clamps the input to [0,1], then maps it to a signed 16-bit integer in
// [output_black_level, output_white_level].
Expr normalized_float_to_i16(Expr input,
                             Expr output_black_level, Expr output_white_level) {
  Expr input_clamped = clamp(input, 0.0f, 1.0f);
  Expr output_range = output_white_level - output_black_level;
  return i16_sat(round(output_black_level + input_clamped * output_range));
}

class FitAndSliceAffineGrid : public Halide::Generator<FitAndSliceAffineGrid> {
 public:
  static constexpr int kVecWidth = 4;

  Input<int> s_sigma_{"s_sigma"};  // Size of each spatial bin
                                   // in the grid. Typically 4.

  Input<float> r_sigma_{"r_sigma"};  // Size of each luma bin in
                                     // the grid. Typically 1/8.

  // levels_ is a 2x2x3 set of parameters.
  // levels_(channel)(black | white)(input | output)
  Input<Buffer<int16_t>> levels_{"levels", 3};

  Input<float> color2luma_red_{"color2luma_red"};
  Input<float> color2luma_green_{"color2luma_green"};
  Input<float> color2luma_blue_{"color2luma_blue"};

  Input<float> curve_alpha_{"curve_alpha"};
  Input<float> lambda_{"lambda"};

  // Low-resolution input image.
  Input<Buffer<int16_t>> splat_loc_i16_{"splat_loc_i16", 3};

  // Low-resolution output image.
  Input<Buffer<int16_t>> values_i16_{"values_i16", 3};

  // High-resolution input image.
  Input<Buffer<int16_t>> slice_loc_i16_{"slice_loc_i16", 3};

  Output<Buffer<int16_t>> sliced_{"sliced", 3};

  void generate() {
    const bool apply_schedule = !auto_schedule;

    // Figure out how much we're upsampling by.
    Expr upsample_factor_x =
        i32(round(f32(slice_loc_i16_.width()) / splat_loc_i16_.width()));
    Expr upsample_factor_y =
        i32(round(f32(slice_loc_i16_.height()) / splat_loc_i16_.height()));
    Expr upsample_factor = max(upsample_factor_x, upsample_factor_y);

    // Convert all inputs to float and add boundary conditions.
    Func clamped_splat_loc("clamped_splat_loc");
    clamped_splat_loc(x, y, c) = normalize_to_float(
        repeat_edge(splat_loc_i16_)(x, y, c),
        levels_(c, 0, 0), levels_(c, 1, 0));

    Func clamped_values("clamped_values");
    clamped_values(x, y, c) = normalize_to_float(
        repeat_edge(values_i16_)(x, y, c),
        levels_(c, 0, 1), levels_(c, 1, 1));

    Func slice_loc("slice_loc");
    slice_loc(x, y, c) = normalize_to_float(
        repeat_edge(slice_loc_i16_)(x, y, c),
        levels_(c, 0, 0), levels_(c, 1, 0));

    Func gray_splat_loc("gray_splat_loc");
    gray_splat_loc(x, y) = (color2luma_red_   * clamped_splat_loc(x, y, 0) +
                            color2luma_green_ * clamped_splat_loc(x, y, 1) +
                            color2luma_blue_  * clamped_splat_loc(x, y, 2));

    Func curved_gray_splat_loc("curved_gray_splat_loc");
    curved_gray_splat_loc(x, y) = gray_splat_loc(x, y) *
        fast_inverse(lerp(1.0f, gray_splat_loc(x, y), curve_alpha_));

    Func gray_slice_loc("gray_slice_loc");
    gray_slice_loc(x, y) = (color2luma_red_   * slice_loc(x, y, 0) +
                            color2luma_green_ * slice_loc(x, y, 1) +
                            color2luma_blue_  * slice_loc(x, y, 2));

    Func curved_gray_slice_loc("curved_gray_slice_loc");
    curved_gray_slice_loc(x, y) = gray_slice_loc(x, y) *
        fast_inverse(lerp(1.0f, gray_slice_loc(x, y), curve_alpha_));

    // Construct the affine bilateral grid.
    Func histogram("histogram");
    RDom r(0, s_sigma_, 0, s_sigma_);
    {
      histogram(x, y, z, c) = 0.0f;

      Expr sx = x * s_sigma_ + r.x - s_sigma_ / 2;
      Expr sy = y * s_sigma_ + r.y - s_sigma_ / 2;
      Expr pos = curved_gray_splat_loc(sx, sy);
      pos = clamp(pos, 0.0f, 1.0f);
      Expr zi = cast<int>(round(pos * (1.0f / r_sigma_)));

      // Sum all the terms needed to fit a 3x4 matrix relating low-res input to
      // low-res output within this bilateral grid cell.
      Expr vr = clamped_values(sx, sy, 0);
      Expr vg = clamped_values(sx, sy, 1);
      Expr vb = clamped_values(sx, sy, 2);
      Expr sr = clamped_splat_loc(sx, sy, 0);
      Expr sg = clamped_splat_loc(sx, sy, 1);
      Expr sb = clamped_splat_loc(sx, sy, 2);

      histogram(x, y, zi, c) +=
          pack_channels(c,
                     {sr * sr, sr * sg, sr * sb,   sr,
                               sg * sg, sg * sb,   sg,
                                        sb * sb,   sb,
                                                 1.0f,
                      vr * sr, vr * sg, vr * sb,   vr,
                      vg * sr, vg * sg, vg * sb,   vg,
                      vb * sr, vb * sg, vb * sb,   vb});
    }

    // Convolution pyramids (Farbman et al.) suggests convolving by something
    // 1/d^3-like to get an interpolating membrane, so we do that. We could also
    // just use a convolution pyramid here, but these grids are really small, so
    // it's OK for the filter to drop sharply and truncate early.
    Expr t0 = 1.0f;
    Expr t1 = 1.0f / 8;
    Expr t2 = 1.0f / 27;
    Expr t3 = 1.0f / 64;

    // Blur the grid using a seven-tap filter.
    Func blurx("blurx");
    Func blury("blury");
    Func blurz("blurz");

    blurz(x, y, z, c) =
        t0 *  histogram(x, y, z    , c) +
        t1 * (histogram(x, y, z - 1, c) + histogram(x, y, z + 1, c)) +
        t2 * (histogram(x, y, z - 2, c) + histogram(x, y, z + 2, c)) +
        t3 * (histogram(x, y, z - 3, c) + histogram(x, y, z + 3, c));
    blury(x, y, z, c) =
        t0 *  blurz(x, y    , z, c) +
        t1 * (blurz(x, y - 1, z, c) + blurz(x, y + 1, z, c)) +
        t2 * (blurz(x, y - 2, z, c) + blurz(x, y + 2, z, c)) +
        t3 * (blurz(x, y - 3, z, c) + blurz(x, y + 3, z, c));
    blurx(x, y, z, c) =
        t0 *  blury(x    , y, z, c) +
        t1 * (blury(x - 1, y, z, c) + blury(x + 1, y, z, c)) +
        t2 * (blury(x - 2, y, z, c) + blury(x + 2, y, z, c)) +
        t3 * (blury(x - 3, y, z, c) + blury(x + 3, y, z, c));

    // Do the solve: convert accumulated Gram matrices (A) and right hand
    // sides (b) into affine color matrices.
    Func matrix("matrix");
    {
      // Pull out the 4x4 symmetric matrix.
      Matrix<4, 4> A;
      A(0, 0) = blurx(x, y, z, 0);
      A(0, 1) = blurx(x, y, z, 1);
      A(0, 2) = blurx(x, y, z, 2);
      A(0, 3) = blurx(x, y, z, 3);
      A(1, 0) = A(0, 1);
      A(1, 1) = blurx(x, y, z, 4);
      A(1, 2) = blurx(x, y, z, 5);
      A(1, 3) = blurx(x, y, z, 6);
      A(2, 0) = A(0, 2);
      A(2, 1) = A(1, 2);
      A(2, 2) = blurx(x, y, z, 7);
      A(2, 3) = blurx(x, y, z, 8);
      A(3, 0) = A(0, 3);
      A(3, 1) = A(1, 3);
      A(3, 2) = A(2, 3);
      A(3, 3) = blurx(x, y, z, 9);

      // Pull out b transposed.
      Matrix<4, 3> b;
      for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 4; ++i) {
          b(i, j) = blurx(x, y, z, 10 + 4 * j + i);
        }
      }

      // Regularize by pushing the solution towards the average gain
      // in this cell = (output_luma + eps) / (input_luma + eps).
      // TODO(jiawen,abadams): potentially expose this constant.
      constexpr float kEpsilon = 0.1f;
      Expr N = A(3, 3);
      Expr output_luma = b(3, 0) + 2 * b(3, 1) + b(3, 2) + kEpsilon * (N + 1);
      Expr input_luma = A(3, 0) + 2 * A(3, 1) + A(3, 2) + kEpsilon * (N + 1);
      Expr gain = output_luma / input_luma;

      Expr weighted_lambda = (N + 1) * lambda_;

      A(0, 0) += weighted_lambda;
      A(1, 1) += weighted_lambda;
      A(2, 2) += weighted_lambda;
      A(3, 3) += weighted_lambda;

      b(0, 0) += weighted_lambda * gain;
      b(1, 1) += weighted_lambda * gain;
      b(2, 2) += weighted_lambda * gain;

      // Now solve Ax = b.
      Matrix<3, 4> result = transpose(solve(A, b, matrix, x, apply_schedule));

      // Pack the resulting matrix into the output Func.
      matrix(x, y, z, c) =
          pack_channels(c, {
              result(0, 0), result(0, 1), result(0, 2), result(0, 3),
              result(1, 0), result(1, 1), result(1, 2), result(1, 3),
              result(2, 0), result(2, 1), result(2, 2), result(2, 3)});
    }

    // If using the shader we stop there, and the Func "matrix" is the output.
    // This monolithic version also compiles a more convenient but slower
    // version that does the trilerp and evaluates the model inside the same
    // Halide pipeline.
    Func interpolated("interpolated");
    Func slice_loc_z("slice_loc_z");
    Func interpolated_matrix_x("interpolated_matrix_x");
    Func interpolated_matrix_y("interpolated_matrix_y");
    Func interpolated_matrix_z("interpolated_matrix_z");
    {
      // Spatial bin size in the high-res image.
      Expr big_sigma = s_sigma_ * upsample_factor;

      // Upsample the matrices in x and y
      Expr yf = cast<float>(y) / big_sigma;
      Expr yi = cast<int>(floor(yf));
      yf -= yi;
      interpolated_matrix_y(x, y, z, c) =
          lerp(matrix(x, yi, z, c),
               matrix(x, yi + 1, z, c),
               yf);

      Expr xf = cast<float>(x) / big_sigma;
      Expr xi = cast<int>(floor(xf));
      xf -= xi;
      interpolated_matrix_x(x, y, z, c) =
          lerp(interpolated_matrix_y(xi, y, z, c),
               interpolated_matrix_y(xi + 1, y, z, c),
               xf);

      // Sample it along the z direction using intensity
      Expr num_intensity_bins = cast<int>(1.0f / r_sigma_);
      Expr val = clamp(curved_gray_slice_loc(x, y), 0.0f, 1.0f);
      Expr zv = val * num_intensity_bins;
      Expr zi = cast<int>(zv);
      Expr zf = zv - zi;
      slice_loc_z(x, y) = {zi, zf};

      interpolated_matrix_z(x, y, c) =
          lerp(interpolated_matrix_x(x, y, slice_loc_z(x, y)[0], c),
               interpolated_matrix_x(x, y, slice_loc_z(x, y)[0] + 1, c),
               slice_loc_z(x, y)[1]);

      interpolated(x, y, c) =
          (interpolated_matrix_z(x, y, 4 * c + 0) * slice_loc(x, y, 0) +
           interpolated_matrix_z(x, y, 4 * c + 1) * slice_loc(x, y, 1) +
           interpolated_matrix_z(x, y, 4 * c + 2) * slice_loc(x, y, 2) +
           interpolated_matrix_z(x, y, 4 * c + 3));
    }

    // Normalize
    sliced_(x, y, c) = normalized_float_to_i16(
        interpolated(x, y, c), levels_(c, 0, 1), levels_(c, 1, 1));

    // ----- Estimates
    {
      constexpr int kWidthBig = 4032;
      constexpr int kHeightBig = 3024;
      constexpr int kWidthSmall = kWidthBig / 8;
      constexpr int kHeightSmall = kHeightBig / 8;
      constexpr int kChannels = 3;

      // Estimates based on default values of flags in bgu_cli.cc
      // and the default downsampling in downsample_cli.cc (8)
      s_sigma_.set_estimate(4.f);
      r_sigma_.set_estimate(0.125f);
      levels_.set_estimates({{0, 3}, {0, 2}, {0, 2}});
      color2luma_red_.set_estimate(0.25f);
      color2luma_green_.set_estimate(0.5f);
      color2luma_blue_.set_estimate(0.25f);
      curve_alpha_.set_estimate(0.8f);
      lambda_.set_estimate(0.001f);
      splat_loc_i16_.set_estimates(
          {{0, kWidthSmall}, {0, kHeightSmall}, {0, kChannels}});
      values_i16_.set_estimates(
          {{0, kWidthSmall}, {0, kHeightSmall}, {0, kChannels}});
      slice_loc_i16_.set_estimates(
          {{0, kWidthBig}, {0, kHeightBig}, {0, kChannels}});
      sliced_.set_estimates({{0, kWidthBig}, {0, kHeightBig}, {0, kChannels}});
    }

    // ----- Schedule
    if (apply_schedule) {
      // Fitting schedule:

      // Compute the per tile histograms and splatting locations within rows of
      // the blur in z.
      histogram.compute_at(blurz, y);
      histogram.update().reorder(c, r.x, r.y, x, y).unroll(c);

      curved_gray_splat_loc.compute_at(blurz, y).vectorize(x, kVecWidth);

      // Compute the blur in z at root.
      blurz.compute_root()
          .reorder(c, z, x, y)
          .parallel(y)
          .vectorize(x, kVecWidth);

      // The blurs of the Gram matrices across x and y will be computed within
      // the outer loops of the matrix solve.
      blury.compute_at(matrix, z).vectorize(x, kVecWidth);

      blurx.compute_at(matrix, x).vectorize(x);

      // The matrix solve. Store c innermost, because subsequent stages will do
      // vectorized loads from this across c.
      // TODO(jiawen,abadams): In decoupled version, remove reorder storage
      // call.
      matrix.compute_root()
          .reorder_storage(c, x, y, z)
          .reorder(c, x, z, y)
          .parallel(y)
          .vectorize(x, kVecWidth)
          .bound(c, 0, 12)  // 3x4 matrix = 12 channels.
          .unroll(c);

      // Applying the curves.

      // Instead of doing a full trilerp at each output pixel, for each scanline
      // of output we first slice out a 2D array of matrices that we'll bilerp
      // into. We'll be accessing it in vectors across the c dimension, so store
      // c innermost.
      interpolated_matrix_y.compute_at(sliced_, y)
          .reorder_storage(c, x, y, z)
          .bound(c, 0, 12)
          .vectorize(c);

      // Compute the output in vectors across x.
      sliced_.compute_root()
          .parallel(y)
          .vectorize(x, kVecWidth)
          .reorder(c, x, y)
          .bound(c, 0, 3)
          .unroll(c);

      // Computing where to slice vectorizes nicely across x.
      curved_gray_slice_loc.compute_at(sliced_, x).vectorize(x);

      slice_loc_z.compute_at(sliced_, x).vectorize(x);

      // But sampling the matrix vectorizes better across c.
      interpolated_matrix_z.compute_at(sliced_, c).vectorize(c).unroll(x);
    }
  }
};

}  // namespace bgu

HALIDE_REGISTER_GENERATOR(bgu::FitAndSliceAffineGrid,
                          FitAndSliceAffineGridHalide)
