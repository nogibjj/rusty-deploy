# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspaces/rusty-deploy/rtorchdist

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspaces/rusty-deploy/rtorchdist/build

# Include any dependencies generated for this target.
include CMakeFiles/check_pytorch.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/check_pytorch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/check_pytorch.dir/flags.make

CMakeFiles/check_pytorch.dir/check_pytorch.cpp.o: CMakeFiles/check_pytorch.dir/flags.make
CMakeFiles/check_pytorch.dir/check_pytorch.cpp.o: ../check_pytorch.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspaces/rusty-deploy/rtorchdist/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/check_pytorch.dir/check_pytorch.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/check_pytorch.dir/check_pytorch.cpp.o -c /workspaces/rusty-deploy/rtorchdist/check_pytorch.cpp

CMakeFiles/check_pytorch.dir/check_pytorch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/check_pytorch.dir/check_pytorch.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspaces/rusty-deploy/rtorchdist/check_pytorch.cpp > CMakeFiles/check_pytorch.dir/check_pytorch.cpp.i

CMakeFiles/check_pytorch.dir/check_pytorch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/check_pytorch.dir/check_pytorch.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspaces/rusty-deploy/rtorchdist/check_pytorch.cpp -o CMakeFiles/check_pytorch.dir/check_pytorch.cpp.s

# Object files for target check_pytorch
check_pytorch_OBJECTS = \
"CMakeFiles/check_pytorch.dir/check_pytorch.cpp.o"

# External object files for target check_pytorch
check_pytorch_EXTERNAL_OBJECTS =

check_pytorch: CMakeFiles/check_pytorch.dir/check_pytorch.cpp.o
check_pytorch: CMakeFiles/check_pytorch.dir/build.make
check_pytorch: /usr/local/lib/libtorch/lib/libtorch.so
check_pytorch: /usr/local/lib/libtorch/lib/libc10.so
check_pytorch: /usr/local/lib/libtorch/lib/libkineto.a
check_pytorch: /usr/local/lib/libtorch/lib/libc10.so
check_pytorch: CMakeFiles/check_pytorch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspaces/rusty-deploy/rtorchdist/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable check_pytorch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/check_pytorch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/check_pytorch.dir/build: check_pytorch

.PHONY : CMakeFiles/check_pytorch.dir/build

CMakeFiles/check_pytorch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/check_pytorch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/check_pytorch.dir/clean

CMakeFiles/check_pytorch.dir/depend:
	cd /workspaces/rusty-deploy/rtorchdist/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspaces/rusty-deploy/rtorchdist /workspaces/rusty-deploy/rtorchdist /workspaces/rusty-deploy/rtorchdist/build /workspaces/rusty-deploy/rtorchdist/build /workspaces/rusty-deploy/rtorchdist/build/CMakeFiles/check_pytorch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/check_pytorch.dir/depend

