# CUDA RayTracing Project

This project implemented ray tracing on GPU (NVIDIA CUDA). It also includes our implementations of various optimization techniques for benchmarking performance and identifying best practices.

<table>
  <tr>
    <td align="center">
      <img src="gif/spheres.gif" alt="Spheres with circulating light source" width="100%">
      <p>Rendering spheres with circulating light source</p>
    </td>
    <td align="center">
      <img src="gif/cat.gif" alt="Playing with cat" width="100%">
      <p>Real-time interactive with TriangleMesh rendering</p>
    </td>
  </tr>
</table>

## Table of Contents
- [Prerequisites](#prerequisites)
- [Build Instructions](#build-instructions)
- [Targets](#targets)
- [Usage](#usage)

## Prerequisites

Before building the project, ensure you have the following installed:

- NVIDIA CUDA Toolkit
- GCC (for compiling the CPU code)
- NVCC (NVIDIA CUDA Compiler)

## Build Instructions

To build the project, you can use the provided `Makefile`. The `Makefile` includes various targets for building different versions of the rendering programs.

To build all targets, simply run:

```sh
make all
```

# Targets

The `Makefile` defines several targets for building different versions of the CUDA rendering and optimization programs:

- `realtime`: Compiles `realtime_render.cu` for real-time rendering.
- `global`: Compiles `global_launcher.cu` for global memory access.
- `array_bvh`: Compiles `different-versions/array_bvh.cu` for BVH with array storage.
- `optimized`: Compiles `optimized.cu` for general optimizations.
- `optimized_vertices-in-shared`: Compiles `different-versions/optimized_vertices-in-shared.cu` for shared memory optimization.
- `optimized_non-coalesced`: Compiles `different-versions/optimized_non-coalesced.cu` for non-coalesced memory access optimization.
- `optimized_bvh-tree`: Compiles `different-versions/optimized_bvh-tree.cu` for optimized BVH tree.
- `optimized_bvh-texture`: Compiles `different-versions/optimized_bvh-texture.cu` for optimized BVH using textures.
- `optimized_recursive`: Compiles `different-versions/optimized_recursive.cu` for optimized recursive BVH.
- `cpu`: Compiles `cpu_launcher.cpp` for CPU rendering using OpenMP.

## Usage

After building the targets using the `make` command (e.g., `make all` or `make <target>`), you can run the respective executables. Here are examples of how to run some of the programs:

```sh
./realtime      # Run the real-time rendering program
./global        # Run the global memory access program
./array_bvh     # Run the BVH with array storage program
./optimized     # Run the general optimizations program
```

## Notes
The code has only been tested on Linux machines (Ubuntu) with CUDA and OpenGL installed.

## Authors

Project Members: Cong VU, Minh PHAM, The NGUYEN
