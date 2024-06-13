NVCC = /usr/local/cuda/bin/nvcc

LDLIBS = -lglut -lGL -lGLU -lm -lGLEW
CUDAFLAGS = -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include --use_fast_math

all: global optimized realtime array_bvh cpu optimized_vertices-in-shared optimized_non-coalesced optimized_bvh-tree optimized_bvh-texture

realtime:
	$(NVCC) realtime_render.cu -o realtime $(CUDAFLAGS) $(LDLIBS)

global:
	$(NVCC) global_launcher.cu -o global $(CUDAFLAGS)

array_bvh:
	$(NVCC) different-versions/array_bvh.cu -o array_bvh $(CUDAFLAGS)

optimized:
	$(NVCC) optimized.cu -o optimized $(CUDAFLAGS)

optimized_vertices-in-shared:
	$(NVCC) different-versions/optimized_vertices-in-shared.cu -o optimized_vertices-in-shared $(CUDAFLAGS)

optimized_non-coalesced:
	$(NVCC) different-versions/optimized_non-coalesced.cu -o optimized_non-coalesced $(CUDAFLAGS)

optimized_bvh-tree:
	$(NVCC) different-versions/optimized_bvh-tree.cu -o optimized_bvh-tree $(CUDAFLAGS)

optimized_bvh-texture:
	$(NVCC) different-versions/optimized_bvh-texture.cu -o optimized_bvh-texture $(CUDAFLAGS)

optimized_bvh-texture:
	$(NVCC) different-versions/optimized_bvh-texture.cu -o optimized_bvh-texture $(CUDAFLAGS)

optimized_recursive:
	$(NVCC) different-versions/optimized_recursive.cu -o optimized_recursive $(CUDAFLAGS)
cpu:
	g++ cpu_launcher.cpp -o cpu -O3 -fopenmp -std=c++17

clean:
	rm -f *.out realtime global optimized array_bvh optimized_vertices-in-shared optimized_non-coalesced optimized_bvh-tree optimized_bvh-texture