# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nuc/slambook/Test/ch5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nuc/slambook/Test/ch5/build

# Include any dependencies generated for this target.
include CMakeFiles/imageBasics.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/imageBasics.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imageBasics.dir/flags.make

CMakeFiles/imageBasics.dir/imageBasics.cpp.o: CMakeFiles/imageBasics.dir/flags.make
CMakeFiles/imageBasics.dir/imageBasics.cpp.o: ../imageBasics.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuc/slambook/Test/ch5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/imageBasics.dir/imageBasics.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/imageBasics.dir/imageBasics.cpp.o -c /home/nuc/slambook/Test/ch5/imageBasics.cpp

CMakeFiles/imageBasics.dir/imageBasics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imageBasics.dir/imageBasics.cpp.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CXX_CREATE_PREPROCESSED_SOURCE

CMakeFiles/imageBasics.dir/imageBasics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imageBasics.dir/imageBasics.cpp.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CXX_CREATE_ASSEMBLY_SOURCE

CMakeFiles/imageBasics.dir/imageBasics.cpp.o.requires:

.PHONY : CMakeFiles/imageBasics.dir/imageBasics.cpp.o.requires

CMakeFiles/imageBasics.dir/imageBasics.cpp.o.provides: CMakeFiles/imageBasics.dir/imageBasics.cpp.o.requires
	$(MAKE) -f CMakeFiles/imageBasics.dir/build.make CMakeFiles/imageBasics.dir/imageBasics.cpp.o.provides.build
.PHONY : CMakeFiles/imageBasics.dir/imageBasics.cpp.o.provides

CMakeFiles/imageBasics.dir/imageBasics.cpp.o.provides.build: CMakeFiles/imageBasics.dir/imageBasics.cpp.o


# Object files for target imageBasics
imageBasics_OBJECTS = \
"CMakeFiles/imageBasics.dir/imageBasics.cpp.o"

# External object files for target imageBasics
imageBasics_EXTERNAL_OBJECTS =

imageBasics: CMakeFiles/imageBasics.dir/imageBasics.cpp.o
imageBasics: CMakeFiles/imageBasics.dir/build.make
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_objdetect.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_ml.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_shape.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_dnn.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_viz.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_stitching.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_videostab.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_superres.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_photo.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_calib3d.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_features2d.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_highgui.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_flann.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_video.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_videoio.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_imgcodecs.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_imgproc.so.4.0.0
imageBasics: /home/nuc/libs/opencv-trunk/release/installed/lib/libopencv_core.so.4.0.0
imageBasics: CMakeFiles/imageBasics.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nuc/slambook/Test/ch5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable imageBasics"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imageBasics.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imageBasics.dir/build: imageBasics

.PHONY : CMakeFiles/imageBasics.dir/build

CMakeFiles/imageBasics.dir/requires: CMakeFiles/imageBasics.dir/imageBasics.cpp.o.requires

.PHONY : CMakeFiles/imageBasics.dir/requires

CMakeFiles/imageBasics.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imageBasics.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imageBasics.dir/clean

CMakeFiles/imageBasics.dir/depend:
	cd /home/nuc/slambook/Test/ch5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nuc/slambook/Test/ch5 /home/nuc/slambook/Test/ch5 /home/nuc/slambook/Test/ch5/build /home/nuc/slambook/Test/ch5/build /home/nuc/slambook/Test/ch5/build/CMakeFiles/imageBasics.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/imageBasics.dir/depend

