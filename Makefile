NVCC = /usr/local/cuda/bin/nvcc

raytracer:
	$(NVCC) raytracer.cu -o raytracer -O3 -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

clean:
	rm -f raytracer