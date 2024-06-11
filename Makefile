NVCC = /usr/local/cuda/bin/nvcc

all: global shared

global:
	$(NVCC) global_launcher.cu -o global -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

shared:
	$(NVCC) shared_launcher.cu -o shared -O3 -arch=sm_75 -std=c++17 -I/usr/local/cuda/include

clean:
	rm -f *.out