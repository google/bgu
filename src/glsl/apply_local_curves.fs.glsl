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
#version 310 es

precision mediump float;

// The high-resolution input.
uniform sampler2D uInput;

// Rows of the 3D array of 3x4 matrices.
uniform sampler3D uAffineGridRow0;
uniform sampler3D uAffineGridRow1;
uniform sampler3D uAffineGridRow2;

layout(location = 0) in vec4 vTexcoord;
layout(location = 0) out vec4 colorOut;

void main() {
    vec4 colorIn = texture(uInput, vTexcoord.xy);
    colorIn.w = 1.0;

    // Compute input luma.
    float luma = dot(vec3(0.25, 0.5, 0.25), colorIn.xyz);

    // Sample the grid at (x, y, luma) to get an affine matrix.
    vec3 gridLoc = vec3(vTexcoord.xy, luma);
    vec4 row0 = texture(uAffineGridRow0, gridLoc);
    vec4 row1 = texture(uAffineGridRow1, gridLoc);
    vec4 row2 = texture(uAffineGridRow2, gridLoc);

    // Apply the matrix.
    colorOut = vec4(dot(colorIn, row0),
		    dot(colorIn, row1),
		    dot(colorIn, row2),
		    1.0);
    colorOut = clamp(colorOut, vec4(0.0), vec4(1.0));
}
