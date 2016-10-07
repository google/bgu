MATLAB source for the global version of the algorithm in the directory "matlab".
The demo script "demo.m" runs Bilateral Guided Upsampling on the Parrot example,
approximating a large detail enhancement using Local Laplacian Filters. To
generate the gallery, download the complete image set then remove the line
containing the error().

Halide source for the fast approximate version of the algorithm in the directory
"fast".

To compile this yourself you will need to download a Halide distribution from
https://github.com/halide/Halide/releases and untar it into the "fast"
directory. These apps fit the local curves and report the timings, and they also
apply the local curves (slowly on the CPU) so that you can inspect the output.

Precompiled versions for linux, os x and windows are available in the
fast/precompiled subdirectory. Run one of them with no arguments for usage
instructions.

For mapping color images to color outputs (e.g. local laplacian filters) see
bilateral_grid_3x4.cpp and filter_3x4.cpp. For mapping color images to grayscale
outputs (e.g. style transfer) see bilateral_grid_1x4.cpp and filter_1x4.cpp

The fast version of the algorithm contains images for a milder version of detail
enhancement. Note that low_res_out.png is a 16-bit PNG and may not display
correctly in some viewers.

3) The OpenGL shader that can apply the local curves at display-time:
apply_local_curves_fs.glsl
