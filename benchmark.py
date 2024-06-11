import subprocess
import time
import numpy as np
import sys

# Check number of arguments
if len(sys.argv) != 2:
    print("Please include the file name to benchmark")
    sys.exit()

# Arguments range (num_rays is power of 2 for better "shared" performance)
num_rays_list = [2**i for i in range(9)]  # [1, 2, 4, 8, 16, 32, 64, 128, 256]
num_bounces_list = range(1, 11)

# Matrix to store the results
results = np.zeros((len(num_rays_list), len(num_bounces_list)))

def run_benchmark(num_rays, num_bounces):
    start_time = time.time()
    subprocess.run([f"./{sys.argv[1]}", str(num_rays), str(num_bounces)], check=True)
    end_time = time.time()
    return end_time - start_time

for i, num_rays in enumerate(num_rays_list):
    for j, num_bounces in enumerate(num_bounces_list):
        runtimes = []
        for _ in range(5):  # Run 5 times for averaging
            runtime = run_benchmark(num_rays, num_bounces)
            runtimes.append(runtime)
        avg_runtime = sum(runtimes) / len(runtimes)
        results[i, j] = avg_runtime
        print(f"num_rays: {num_rays}, num_bounces: {num_bounces}, avg_runtime: {avg_runtime:.4f} seconds")

print(f"\nBenchmark Results of file {sys.argv[1]} (in seconds):")
print(f"The horizontal axis is number of bounces and the vertical axis is number of rays.")
print("    " + " ".join([f"{b:>6}" for b in num_bounces_list]))
for i, num_rays in enumerate(num_rays_list):
    print(f"{num_rays:>8} " + " ".join([f"{results[i, j]:>6.4f}" for j in range(len(num_bounces_list))]))
