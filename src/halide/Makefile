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

all: high_res_out.png high_res_out_gray.png

# Compile Halide source into an executable called bgu_3x4.
bgu_3x4: bgu_3x4.cpp
	c++ -std=c++11 -g \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools \
		-L $(HALIDE_PATH)/bin \
		-lHalide -lpthread -ldl -lz \
		-o bgu_3x4 bgu_3x4.cpp

# Compile Halide source into an executable called bgu_1x4.
bgu_1x4: bgu_1x4.cpp
	c++ -std=c++11 -g \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools \
		-L $(HALIDE_PATH)/bin \
		-lHalide -lpthread -ldl -lz \
		-o bgu_1x4 bgu_1x4.cpp

# Run ./bgu_3x4 to produce bgu_3x4.o and filter_only_3x4.o.
bgu_3x4.o: bgu_3x4
	HL_TARGET=$(HALIDE_TARGET) DYLD_LIBRARY_PATH=$(HALIDE_PATH)/bin LD_LIBRARY_PATH=$(HALIDE_PATH)/bin ./bgu_3x4

# Run ./bgu_1x4 to produce bgu_1x4.o and filter_only_1x4.o.
bgu_1x4.o: bgu_1x4
	HL_TARGET=$(HALIDE_TARGET) DYLD_LIBRARY_PATH=$(HALIDE_PATH)/bin LD_LIBRARY_PATH=$(HALIDE_PATH)/bin ./bgu_1x4

# Compile regular C++ source in filter_3x4.cpp and link with .o files to produce
# filter_3x4 executable.
filter_3x4: bgu_3x4.o filter_3x4.cpp
	c++ -std=c++11 -O3 -ffast-math -Wall -Werror \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools -I $(MACOS_PREFIX_PATH)/include \
		-L $(MACOS_PREFIX_PATH)/lib \
		-lpthread -ldl -lpng -lz \
		-o filter_3x4 \
		bgu_3x4.o fit_only_3x4.o filter_3x4.cpp

# Compile regular C++ source in filter_1x4.cpp and link with .o files to produce
# filter_1x4 executable.
filter_1x4: bgu_1x4.o filter_1x4.cpp
	c++ -std=c++11 -O3 -ffast-math -Wall -Werror \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools -I $(MACOS_PREFIX_PATH)/include \
		-L $(MACOS_PREFIX_PATH)/lib \
		-lpthread -ldl -lpng -lz \
		-o filter_1x4 \
		bgu_1x4.o fit_only_1x4.o filter_1x4.cpp

# Static-linkage versions for easier distribution.
# Note that MacOS does not support static linkage.
filter_3x4_static: bgu_3x4.o filter_3x4.cpp
	c++ -static -std=c++11 -O3 -ffast-math -Wall -Werror \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools \
		-lpthread -ldl -lpng -lz \
		-o filter_3x4_static \
		filter_3x4.cpp bgu_3x4.o fit_only_3x4.o

filter_1x4_static: bgu_1x4.o filter_1x4.cpp
	c++ -static -std=c++11 -O3 -ffast-math -Wall -Werror \
		-I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools \
		-lpthread -ldl -lpng -lz \
		-o filter_3x4_static \
		filter_1x4.cpp bgu_1x4.o fit_only_1x4.o

high_res_out.png: filter_3x4
	./filter_3x4 ../../images/low_res_in.png ../../images/low_res_out.png ../../images/high_res_in.png high_res_out.png 16 8

high_res_out_gray.png: filter_1x4
	./filter_1x4 ../../images/low_res_in.png ../../images/low_res_out_gray.png ../../images/high_res_in.png high_res_out_gray.png 16 8

clean:
	rm -f bgu*.o bgu_3x4 bgu_1x4 bgu*.h filter_1x4 filter_3x4 fit_only_*.o fit_only_*.h high_res_out*.png
	rm -rf bgu_3x4.dSYM bgu_1x4.dSYM
