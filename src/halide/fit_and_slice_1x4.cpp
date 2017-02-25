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

// This is a slightly simplified version of bilateral_grid_3x4.cpp for
// tasks that output a single channel (e.g. matting). This still
// requires a 4x4 matrix solve, so most of the code is the same.

#include "Halide.h"
#include <cstdio>

using namespace Halide;
using Halide::ConciseCasts::f32;
using Halide::ConciseCasts::i32;

Var x("x"), y("y"), z("z"), c("c");

// A class to hold a matrix of Halide Exprs.
template<int rows, int cols>
struct Matrix {
    Expr exprs[rows][cols];

    Expr operator()(int i, int j) const {
        return exprs[i][j];
    }

    Expr &operator()(int i, int j) {
        return exprs[i][j];
    }

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
template<int R, int S, int T>
Matrix<R, T> mat_mul(const Matrix<R, S> &A, const Matrix<S, T> &B) {
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
template<int M, int N>
Matrix<M, N> solve(Matrix<M, M> A, Matrix<M, N> b, Func compute, Var at) {
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

    // eliminate lower left
    for (int k = 0; k < M-1; k++) {
        for (int i = k+1; i < M; i++) {
            f(x, y, z, -1, 0) = f(x, y, z, i, k) / f(x, y, z, k, k);
            for (int j = k+1; j < M+N; j++) {
                f(x, y, z, i, j) -= f(x, y, z, k, j) * f(x, y, z, -1, 0);
            }
            f(x, y, z, i, k) = 0.0f;
        }
    }

    // eliminate upper right
    for (int k = M-1; k > 0; k--) {
        for (int i = 0; i < k; i++) {
            f(x, y, z, -1, 0) = f(x, y, z, i, k) / f(x, y, z, k, k);
            for (int j = k+1; j < M+N; j++) {
                f(x, y, z, i, j) -= f(x, y, z, k, j) * f(x, y, z, -1, 0);
            }
            f(x, y, z, i, k) = 0.0f;
        }
    }

    // Divide by diagonal and put it in the output matrix.
    for (int i = 0; i < M; i++) {
        f(x, y, z, i, i) = 1.0f/f(x, y, z, i, i);
        for (int j = 0; j < N; j++) {
            b(i, j) = f(x, y, z, i, j+M) * f(x, y, z, i, i);
        }
    }

    for (int i = 0; i < f.num_update_definitions(); i++) {
        f.update(i).vectorize(x);
    }

    f.compute_at(compute, at);

    return b;
};


template<int N, int M>
Matrix<M, N> transpose(const Matrix<N, M> &in) {
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
    for (int i = (int)exprs.size() - 2; i >= 0; i--) {
        e = select(c == i, exprs[i], e);
    }
    return e;
}

int main(int argc, char **argv) {

    // The low resolution output
    ImageParam values(Float(32), 2);

    // The low resolution input
    ImageParam splat_loc(Float(32), 3);

    // The high resolution output
    ImageParam slice_loc(Float(32), 3);

    // Size of each luma bin in the grid. Typically 1/8.
    Param<float> r_sigma("r_sigma");

    // Size of each spatial bin in the grid. Typically 16.
    Param<int> s_sigma("s_sigma");

    // Add a boundary condition to the inputs.
    Func clamped_values = BoundaryConditions::repeat_edge(values);
    Func clamped_splat_loc = BoundaryConditions::repeat_edge(splat_loc);

    // Figure out how much we're upsampling by. Not relevant if we're
    // just fitting curves.
    Expr upsample_factor_x =
        i32(ceil(f32(slice_loc.width()) / splat_loc.width()));
    Expr upsample_factor_y =
        i32(ceil(f32(slice_loc.height()) / splat_loc.height()));
    Expr upsample_factor = max(upsample_factor_x, upsample_factor_y);

    Func gray_splat_loc;
    gray_splat_loc(x, y) = (0.25f * clamped_splat_loc(x, y, 0) +
                            0.5f * clamped_splat_loc(x, y, 1) +
                            0.25f * clamped_splat_loc(x, y, 2));

    Func gray_slice_loc;
    gray_slice_loc(x, y) = (0.25f * slice_loc(x, y, 0) +
                            0.5f * slice_loc(x, y, 1) +
                            0.25f * slice_loc(x, y, 2));

    // Construct the bilateral grid
    Func histogram("histogram");
    RDom r(0, s_sigma, 0, s_sigma);
    {
        histogram(x, y, z, c) = 0.0f;

        Expr sx = x * s_sigma + r.x - s_sigma/2, sy = y * s_sigma + r.y - s_sigma/2;
        Expr pos = gray_splat_loc(sx, sy);
        pos = clamp(pos, 0.0f, 1.0f);
        Expr zi = cast<int>(round(pos * (1.0f/r_sigma)));

        // Sum all the terms we need to fit a line relating
        // low-res input to low-res output within this bilateral grid
        // cell
        Expr v = clamped_values(sx, sy);
        Expr sr = clamped_splat_loc(sx, sy, 0), sg = clamped_splat_loc(sx, sy, 1), sb = clamped_splat_loc(sx, sy, 2);

        histogram(x, y, zi, c) +=
            pack_channels(c,
                       {sr*sr, sr*sg, sr*sb, sr,
                               sg*sg, sg*sb, sg,
                                      sb*sb, sb,
                                           1.0f,
                         v*sr,  v*sg,  v*sb,  v});

    }

    // Convolution pyramids (Farbman et al.) suggests convolving by
    // something 1/d^3-like to get an interpolating membrane, so we do
    // that. We could also just use a convolution pyramid here, but
    // these grids are really small, so it's OK for the filter to drop
    // sharply and truncate early.
    Expr t0 = 1.0f/64, t1 = 1.0f/27, t2 = 1.0f/8, t3 = 1.0f;

    // Blur the grid using a seven-tap filter
    Func blurx("blurx"), blury("blury"), blurz("blurz");

    blurz(x, y, z, c) = (histogram(x, y, z-3, c)*t0 +
                         histogram(x, y, z-2, c)*t1 +
                         histogram(x, y, z-1, c)*t2 +
                         histogram(x, y, z  , c)*t3 +
                         histogram(x, y, z+1, c)*t2 +
                         histogram(x, y, z+2, c)*t1 +
                         histogram(x, y, z+3, c)*t0);
    blury(x, y, z, c) = (blurz(x, y-3, z, c)*t0 +
                         blurz(x, y-2, z, c)*t1 +
                         blurz(x, y-1, z, c)*t2 +
                         blurz(x, y  , z, c)*t3 +
                         blurz(x, y+1, z, c)*t2 +
                         blurz(x, y+2, z, c)*t1 +
                         blurz(x, y+3, z, c)*t0);
    blurx(x, y, z, c) = (blury(x-3, y, z, c)*t0 +
                         blury(x-2, y, z, c)*t1 +
                         blury(x-1, y, z, c)*t2 +
                         blury(x  , y, z, c)*t3 +
                         blury(x+1, y, z, c)*t2 +
                         blury(x+2, y, z, c)*t1 +
                         blury(x+3, y, z, c)*t0);

    // Do the solve, to convert the accumulated values to the affine
    // matrices.
    Func line("line");
    {
        // Pull out the 4x4 symmetric matrix from the values we've
        // accumulated.
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

        // Pull out the rhs
        Matrix<4, 1> b;
        b(0, 0) = blurx(x, y, z, 10);
        b(1, 0) = blurx(x, y, z, 11);
        b(2, 0) = blurx(x, y, z, 12);
        b(3, 0) = blurx(x, y, z, 13);

        // Regularize by pushing the solution towards the average gain
        // in this cell = (average output luma + eps) / (average input luma + eps).
        const float lambda = 1e-6f;
        const float epsilon = 1e-6f;

        // The bottom right entry of A is a count of the number of
        // constraints affecting this cell.
        Expr N = A(3, 3);

        // The last row of each matrix is the sum of input RGB values
        // and output luma values for the pixels affecting this
        // cell. Instead of dividing them by N+1 to get averages,
        // we'll multiply epsilon by N+1. This saves two
        // divisions. We'll also need to multiply the luma values by
        // four, because using 1 2 1 for the input weights actually
        // computes luma*4.
        Expr output_luma = 4 * b(3, 0) + epsilon * (N + 1);
        Expr input_luma = A(3, 0) + 2 * A(3, 1) + A(3, 2) + epsilon * (N + 1);
        Expr gain = output_luma / input_luma;

        // Add lambda and lambda*gain to the diagonal of the matrices.
        // In the rgb -> rgb case we regularized the transform to be
        // close to gain * identity. In the rgb -> gray case we
        // regularize the transform to be close to gain * (some
        // reasonable conversion of the input to luma).
        Expr weighted_lambda = lambda * (N + 1);
        A(0, 0) += weighted_lambda;
        A(1, 1) += weighted_lambda;
        A(2, 2) += weighted_lambda;
        A(3, 3) += weighted_lambda;

        b(0, 0) += weighted_lambda * gain * 0.25f;
        b(1, 0) += weighted_lambda * gain * 0.5f;
        b(2, 0) += weighted_lambda * gain * 0.25f;

        // Now solve Ax = b
        Matrix<1, 4> result = transpose(solve(A, b, line, x));

        // Pack the resulting matrix into the output Func.
        line(x, y, z, c) = pack_channels(c, {
                    result(0, 0),
                    result(0, 1),
                    result(0, 2),
                    result(0, 3)});
    }

    // If using the shader we stop there, and the Func "line" is the
    // output. We also compile a more convenient but slower version
    // that does the trilerp and evaluates the model inside the same
    // Halide pipeline.

    // We'll take trilinear samples to compute the output. We factor
    // this into several stages to make better use of SIMD.
    Func interpolated("interpolated");
    Func slice_loc_z("slice_loc_z");
    Func interpolated_matrix_x("interpolated_matrix_x");
    Func interpolated_matrix_y("interpolated_matrix_y");
    Func interpolated_matrix_z("interpolated_matrix_z");
    {
        // Spatial bin size in the high-res image.
        Expr big_sigma = s_sigma * upsample_factor;

        // Upsample the matrices in x and y.
        Expr yf = cast<float>(y) / big_sigma;
        Expr yi = cast<int>(floor(yf));
        yf -= yi;
        interpolated_matrix_y(x, y, z, c) =
            lerp(line(x, yi, z, c),
                 line(x, yi + 1, z, c),
                 yf);

        Expr xf = cast<float>(x) / big_sigma;
        Expr xi = cast<int>(floor(xf));
        xf -= xi;
        interpolated_matrix_x(x, y, z, c) =
            lerp(interpolated_matrix_y(xi, y, z, c),
                 interpolated_matrix_y(xi + 1, y, z, c),
                 xf);

        // Sample it along the z direction using intensity.
        Expr num_intensity_bins = cast<int>(1.0f / r_sigma);
        Expr val = gray_slice_loc(x, y);
        val = clamp(val, 0.0f, 1.0f);
        Expr zv = val * num_intensity_bins;
        Expr zi = cast<int>(zv);
        Expr zf = zv - zi;
        slice_loc_z(x, y) = {zi, zf};

        interpolated_matrix_z(x, y, c) =
            lerp(interpolated_matrix_x(x, y, slice_loc_z(x, y)[0], c),
                 interpolated_matrix_x(x, y, slice_loc_z(x, y)[0]+1, c),
                 slice_loc_z(x, y)[1]);

        // Multiply by 3x4 by 4x1.
        interpolated(x, y) =
            (interpolated_matrix_z(x, y, 0) * slice_loc(x, y, 0) +
             interpolated_matrix_z(x, y, 1) * slice_loc(x, y, 1) +
             interpolated_matrix_z(x, y, 2) * slice_loc(x, y, 2) +
             interpolated_matrix_z(x, y, 3));
    }

    // Normalize
    Func slice("slice");
    slice(x, y) = clamp(interpolated(x, y), 0.0f, 1.0f);

    // The schedule. Based on the schedule for bilateral grid in the
    // Halide repo.

    // Fitting the curves.

    // Compute the per tile histograms and splatting locations within
    // rows of the blur in z.
    histogram
        .compute_at(blurz, y);
    histogram.update()
        .reorder(c, r.x, r.y, x, y)
        .unroll(c);

    gray_splat_loc
        .compute_at(blurz, y)
        .vectorize(x, 8);

    // Compute the blur in z at root
    blurz
        .compute_root()
        .reorder(c, z, x, y)
        .parallel(y)
        .vectorize(x, 8);

    // The blurs of the Gram matrices across x and y will be computed
    // within the outer loops of the matrix solve.
    blury
        .compute_at(line, z)
        .vectorize(x, 4);

    blurx
        .compute_at(line, x)
        .vectorize(x);

    // The matrix solve. Store c innermost, because subsequent stages
    // will do vectorized loads from this across c. If you just want
    // the matrices, you probably want to remove this reorder storage
    // call.
    line
        .compute_root()
        .reorder_storage(c, x, y, z)
        .reorder(c, x, z, y)
        .parallel(y)
        .vectorize(x, 8)
        .bound(c, 0, 4)
        .unroll(c);

    // Applying the curves.

    // You should really do the trilerp on the GPU in a shader. We can
    // make the CPU implementation a little faster though by factoring
    // it into a few stages. At scanline of output we first slice out
    // a 2D array of matrices that we'll bilerp into. We'll be
    // accessing it in vectors across the c dimension, so store c
    // innermost.
    interpolated_matrix_y
        .compute_root()
        .reorder_storage(c, x, y, z)
        .bound(c, 0, 4)
        .vectorize(c);

    // Compute the output in vectors across x.
    slice
        .compute_root()
        .parallel(y)
        .vectorize(x, 8);

    // Computing where to slice vectorizes nicely across x.
    gray_slice_loc
        .compute_root()
        .vectorize(x, 8);

    slice_loc_z
        .compute_root()
        .vectorize(x, 8);

    // But sampling the matrix vectorizes better across c.
    interpolated_matrix_z
        .compute_root()
        .vectorize(c, 4);

    // Compile a version that does the slicing.
    slice.compile_to_file("fit_and_slice_1x4", {r_sigma, s_sigma, splat_loc,
        values, slice_loc}, "fit_and_slice_1x4");

    // Compile a version that just does the fitting. Use this with the GL
    // shader.
    line.compile_to_file("fit_only_1x4", {r_sigma, s_sigma, splat_loc, values,
        slice_loc}, "fit_only_1x4");

    return 0;
}
