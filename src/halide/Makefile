# Point to the bin directory of a halide distro from here:
# https://github.com/halide/Halide/releases

# This should work if you unpack a Halide distribution in this folder.
HALIDE_PATH ?= halide

# On MacOS with MacPorts, zlib and libpng gets installed into /opt/local. Change
# as necessary for Homebrew or Fink.
MACOS_PREFIX_PATH ?= /opt/local

# Change this to arm-32-android or arm-32-ios or similar to compile
# object files usable in an Android or iOS app.
HALIDE_TARGET ?= host

all: box_downsample_cli high_res_out.png high_res_out_gray.png

# Compile Halide source into an executable called box_downsample_generate.
box_downsample_generate: box_downsample_generator.cpp
	c++ -std=c++11 -g -fno-rtti \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools \
		-L $(HALIDE_PATH)/bin \
		-lHalide -lpthread -ldl \
		-o box_downsample_generate \
		box_downsample_generator.cpp $(HALIDE_PATH)/tools/GenGen.cpp

# Run box_downsample_generator to create box_downsample_halide.{h,a}.
# Then link against it to produce box_downsample_cli.
box_downsample_cli: box_downsample_generate box_downsample_cli.cpp
	DYLD_LIBRARY_PATH=$(HALIDE_PATH)/bin LD_LIBRARY_PATH=$(HALIDE_PATH)/bin ./box_downsample_generate -o . target=host
	c++ -std=c++11 -O3 -ffast-math -Wall -Werror \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools -I $(MACOS_PREFIX_PATH)/include \
		-L $(MACOS_PREFIX_PATH)/lib \
		-lpthread -ldl -lpng -lz \
		-o box_downsample_cli \
		box_downsample_halide.a \
		box_downsample_cli.cpp

# Compile Halide source into an executable called fit_and_slice_3x4.
fit_and_slice_3x4: fit_and_slice_3x4.cpp
	c++ -std=c++11 -g \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools \
		-L $(HALIDE_PATH)/bin \
		-lHalide -lpthread -ldl -lz \
		-o fit_and_slice_3x4 fit_and_slice_3x4.cpp

# Compile Halide source into an executable called fit_and_slice_1x4.
fit_and_slice_1x4: fit_and_slice_1x4.cpp
	c++ -std=c++11 -g \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools \
		-L $(HALIDE_PATH)/bin \
		-lHalide -lpthread -ldl -lz \
		-o fit_and_slice_1x4 \
		fit_and_slice_1x4.cpp

# Run ./fit_and_slice_3x4 to produce fit_and_slice_3x4.o and filter_only_3x4.o.
fit_and_slice_3x4.o: fit_and_slice_3x4
	HL_TARGET=$(HALIDE_TARGET) DYLD_LIBRARY_PATH=$(HALIDE_PATH)/bin LD_LIBRARY_PATH=$(HALIDE_PATH)/bin ./fit_and_slice_3x4

# Run ./fit_and_slice_1x4 to produce fit_and_slice_1x4.o and filter_only_1x4.o.
fit_and_slice_1x4.o: fit_and_slice_1x4
	HL_TARGET=$(HALIDE_TARGET) DYLD_LIBRARY_PATH=$(HALIDE_PATH)/bin LD_LIBRARY_PATH=$(HALIDE_PATH)/bin ./fit_and_slice_1x4

# Build regular C++ command line program bgu_cli.cpp to produce the
# bgu_cli executable.
#
# It links in fit_and_slice_mxn.o and fit_only_mxn.o.
bgu_cli: fit_and_slice_1x4.o fit_and_slice_3x4.o bgu_cli.cpp
	c++ -std=c++11 -O3 -ffast-math -Wall -Werror \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools -I $(MACOS_PREFIX_PATH)/include \
		-L $(MACOS_PREFIX_PATH)/lib \
		-lpthread -ldl -lpng -lz \
		-o bgu_cli \
		fit_and_slice_1x4.o fit_only_1x4.o \
		fit_and_slice_3x4.o fit_only_3x4.o \
		bgu_cli.cpp

# Static-linkage version for easier distribution.
# Note that MacOS does not support static linkage.
bgu_cli_static: fit_and_slice_1x4.o fit_and_slice_3x4.o bgu_cli.cpp
	c++ -static -std=c++11 -O3 -ffast-math -Wall -Werror \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools \
		-lpthread -ldl -lpng -lz \
		-o bgu_cli_static \
		fit_and_slice_1x4.o fit_only_1x4.o \
		fit_and_slice_3x4.o fit_only_3x4.o \
		bgu_cli.cpp

high_res_out.png: bgu_cli
	./bgu_cli ../../images/low_res_in.png ../../images/low_res_out.png ../../images/high_res_in.png high_res_out.png 16 8

high_res_out_gray.png: bgu_cli
	./bgu_cli ../../images/low_res_in.png ../../images/low_res_out_gray.png ../../images/high_res_in.png high_res_out_gray.png 16 8

clean:
	rm -f fit_and_slice_*.h fit_and_slice_*.o fit_only_*.h fit_only_*.o
	rm -f high_res_out*.png
	rm -f box_downsample_halide.a box_downsample_halide.h
	rm -f box_downsample_cli bgu_cli
	rm -rf box_downsample_generate.dSYM
	rm -rf fit_and_slice_3x4.dSYM fit_and_slice_1x4.dSYM
