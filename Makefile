NVCC = /usr/local/cuda/bin/nvcc

LDLIBS = -lglut -lGL -lGLU -lm -lGLEW

all: global shared realtime array_bvh cpu

realtime:
	$(NVCC) realtime_render.cu -o realtime -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include $(LDLIBS)

global:
	$(NVCC) global_launcher.cu -o global -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

array_bvh:
	$(NVCC) array_bvh.cu -o array_bvh -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

shared:
	$(NVCC) shared_memory.cu -o shared -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

cpu:
	g++ cpu_launcher.cpp -o cpu -O3 -fopenmp -std=c++17

clean:
	rm -f *.out realtime global shared array_bvh