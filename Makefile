NVCC = /usr/local/cuda/bin/nvcc

LDLIBS = -lglut -lGL -lGLU -lm -lGLEW

all: global optimized realtime array_bvh cpu optimized_vertices-in-shared optimized_non-coalesced optimized_bvh-tree optimized_bvh-texture

realtime:
	$(NVCC) realtime_render.cu -o realtime -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include $(LDLIBS)

global:
	$(NVCC) global_launcher.cu -o global -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

array_bvh:
	$(NVCC) array_bvh.cu -o array_bvh -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

optimized:
	$(NVCC) optimized.cu -o optimized -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

optimized_vertices-in-shared:
	$(NVCC) optimized_vertices-in-shared.cu -o optimized_vertices-in-shared -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

optimized_non-coalesced:
	$(NVCC) optimized_non-coalesced.cu -o optimized_non-coalesced -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

optimized_bvh-tree:
	$(NVCC) optimized_bvh-tree.cu -o optimized_bvh-tree -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

optimized_bvh-texture:
	$(NVCC) optimized_bvh-texture.cu -o optimized_bvh-texture -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

cpu:
	g++ cpu_launcher.cpp -o cpu -O3 -fopenmp -std=c++17

clean:
	rm -f *.out realtime global optimized array_bvh optimized_vertices-in-shared optimized_non-coalesced optimized_bvh-tree optimized_bvh-texture