NVCC = /usr/local/cuda/bin/nvcc

raytracer:
	$(NVCC) raytracer.cu -o raytracer -O3 -arch=sm_86 -std=c++17 -I/usr/local/cuda/include 

clean:
	rm -f raytracer