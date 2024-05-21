#include <stdio.h>
#include <math.h>
#include <vector>
#include <random>
#include <iostream>
#include <thrust/device_vector.h>
#include <curand_kernel.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define SQR(X) ((X)*(X))
#define NORMED_VEC(X) ((X) / (X).norm())
#ifndef PI
    #define PI 3.14159265358979323846
#endif
#define PRINT_VEC(v) (printf("%s: (%lf %lf %lf)\n", #v, (v)[0], (v)[1], (v)[2]))
#define INF (1e9+9)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ inline double uniform(curandState *rand_states, unsigned int tid) {
    curandState local_state = rand_states[tid];
    double RANDOM = curand_uniform( &local_state );
    rand_states[tid] = local_state;
	return RANDOM;
}

class Vector {
public:
	__device__ Vector(double x = 0, double y = 0, double z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	__device__ double norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	__device__ double norm() const {
		return sqrt(norm2());
	}
	__device__ void normalize() {
		double n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
	}
	__device__ double operator[](int i) const { return data[i]; };
	__device__ double& operator[](int i) { return data[i]; };
	double data[3];
};

__device__ Vector operator+(const Vector& a, const Vector& b) {
	return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
__device__ Vector operator-(const Vector& a, const Vector& b) {
	return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
__device__ Vector operator-(const Vector& a) {
	return Vector(-a[0], -a[1], -a[2]);
}
__device__ Vector operator*(const double a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
__device__ Vector operator*(const Vector& a, const double b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
// Element wise vector multiplication
__device__ Vector operator*(const Vector& a, const Vector& b) {
	return Vector(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
__device__ Vector operator/(const Vector& a, const double b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
__device__ double dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
__device__ Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

class Ray {
public:
	__device__ Ray(const Vector &O, const Vector &u, double refraction_index = 1.) : O(O), u(u), refraction_index(refraction_index) {};
	// ...
	Vector O, u;
	double refraction_index;
};

class Geometry {
public:
	__device__ Geometry(const Vector &albedo, int id, bool mirror, double in_refraction_index, double out_refraction_index): albedo(albedo), id(id),
	mirror(mirror), in_refraction_index(in_refraction_index), out_refraction_index(out_refraction_index) {}
	__device__ Geometry(): mirror(0), in_refraction_index(1), out_refraction_index(1) {};

	Vector albedo;
	int id;
	bool mirror;
	double in_refraction_index;
	double out_refraction_index;
	__device__ virtual bool intersect(const Ray& r, double &t, Vector &N) { return 0; };
};

class Sphere: public Geometry {
public:
	__device__ Sphere(const Vector &C, double R, const Vector& albedo, bool mirror = 0, double in_refraction_index = 1., double out_refraction_index = 1.) : 
	C(C), R(R), Geometry(albedo, id, mirror, in_refraction_index, out_refraction_index) {};
    Vector C;
    double R;
	__device__ bool intersect(const Ray &r, double &t, Vector &N) override {
		double delta = SQR(dot(r.u, r.O - C)) - ((r.O - C).norm2() - R*R);
		if (delta < 0)
			return 0;
		double t1 = dot(r.u, C - r.O) - sqrt(delta); // first intersection
		double t2 = dot(r.u, C - r.O) + sqrt(delta); // second intersection
		if (t2 < 0)
			return 0;
		t = t1 < 0 ? t2 : t1;
		N = r.O + t * r.u - C;
		N.normalize();
		return 1;
	}
};

class Scene {
public:
	__device__ void addObject(Geometry* s) {
		s->id = objects_size;
		objects[objects_size++] = s;
	}
	
	__device__ bool intersect_all(const Ray& r, Vector &P, Vector &N, int &objectId) {
		double t_min = INF;
		int id_min = -1;
		Vector N_min;
        for (int i = 0; i < objects_size; i++) {
            Geometry* object_ptr = objects[i];
			double t;
			double id = object_ptr->id;
			Vector N_tmp;
			bool ok = object_ptr->intersect(r, t, N_tmp);
			if (ok && t < t_min) {
				t_min = t;
				id_min = id;
				N_min = N_tmp;
			}
		}
		P = r.O + t_min * r.u;
		objectId = id_min;
		N = N_min;
		return id_min != -1;
	}

	__device__ Vector getColor(const Ray& ray, int ray_depth) {
		if (ray_depth < 0) return Vector(0., 0., 0.); // terminates recursion at some <- point
		Vector P, N;
		int sphere_id = -1;
		bool inter = intersect_all(ray, P, N, sphere_id);
		Vector color;
		if (inter) {
			if (objects[sphere_id]->mirror) {
				// Reflection
				double epsilon = 1e-6;
				Vector P_adjusted = P + epsilon * N;
				Vector new_direction = ray.u - 2 * dot(ray.u, N) * N;
				Ray reflected_ray(P_adjusted, new_direction, ray.refraction_index);
				return getColor(reflected_ray, ray_depth - 1);
			} else if (objects[sphere_id]->in_refraction_index != objects[sphere_id]->out_refraction_index) {
				// Refraction
				double epsilon = 1e-6;
				double refract_ratio;
				bool out2in = ray.refraction_index == objects[sphere_id]->out_refraction_index;
				if (out2in) { 
					// outside to inside
					refract_ratio = objects[sphere_id]->out_refraction_index / objects[sphere_id]->in_refraction_index;
				} else { 
					// inside to outside
					refract_ratio = objects[sphere_id]->in_refraction_index / objects[sphere_id]->out_refraction_index;
					N = -N;
				}
				if (((out2in && ray.refraction_index > objects[sphere_id]->in_refraction_index) ||
					(!out2in && ray.refraction_index > objects[sphere_id]->out_refraction_index)) &&
					SQR(refract_ratio) * (1 - SQR(dot(ray.u, N))) > 1) { 
					// total internal reflection
					return getColor(Ray(P + epsilon * N, ray.u - 2 * dot(ray.u, N) * N, ray.refraction_index), ray_depth - 1);
				}
				Vector P_adjusted = P - epsilon * N;
				Vector N_component = - sqrt(1 - SQR(refract_ratio) * (1 - SQR(dot(ray.u, N)))) * N;
				Vector T_component = refract_ratio * (ray.u - dot(ray.u, N) * N);
				Vector new_direction = N_component + T_component;
				if (out2in) {
					return getColor(Ray(P_adjusted, new_direction, objects[sphere_id]->in_refraction_index), ray_depth - 1);
				} else {
					return getColor(Ray(P_adjusted, new_direction, objects[sphere_id]->out_refraction_index), ray_depth - 1);
				}
			} else {
				// 	handle diffuse surfaces
				// 	Get shadow
				Vector P_prime;
				int sphere_id_shadow;
				double epsilon = 1e-6;
				Vector P_adjusted = P + epsilon * N;
				Vector direct_color, indirect_color;
				Vector N_prime;
				bool _ = intersect_all(Ray(P_adjusted, NORMED_VEC(L - P_adjusted)), P_prime, N_prime, sphere_id_shadow);
				
				if ((P_prime - P_adjusted).norm2() <= (L - P_adjusted).norm2()) {
					// Is shadow
					direct_color = Vector(0, 0, 0);
				} else {
					// Get direct color
					Geometry* S = objects[sphere_id];
					Vector wlight = L - P;
					wlight.normalize();
					double l = intensity / (4 * PI * (L - P).norm2()) * max(dot(N, wlight), 0.);
					direct_color = l * S->albedo / PI;
				}
				// Get indirect color by launching ray
				unsigned int seed = threadIdx.x;
				double r1 = uniform(rand_states, seed);
				double r2 = uniform(rand_states, seed);
				double x = cos(2 * PI * r1) * sqrt(1 - r2);
				double y = sin(2 * PI * r1) * sqrt(1 - r2);
				double z = sqrt(r2);
				Vector T1;
				if (abs(N[1]) != 0 && abs(N[0]) != 0) {
					T1 = Vector(-N[1], N[0], 0);
				} else {
					T1 = Vector(-N[2], 0, N[0]);
				}
				T1.normalize();
				Vector T2 = cross(N, T1);
				Vector random_direction = x * T1 + y * T2 + z * N;
				indirect_color = ((Geometry *)objects[sphere_id])->albedo * getColor(Ray(P_adjusted, random_direction), ray_depth - 1);
				color = direct_color + indirect_color;
			}
		}
		return color;
}
	Geometry* objects[100];
    int objects_size = 0;
	double intensity = 3e10;
	Vector L;
	curandState* rand_states;
};

__global__ void KernelInit(Scene *s) {
 	auto id = threadIdx.x + blockIdx.x * blockDim.x;
	if (!id) {
		s->L = Vector(-10., 20., 40.);
		s->objects_size = 0;
		s->intensity = 3e10;
		s->addObject(new Sphere(Vector(0, 0, 0), 10, Vector(1., 1., 1.))); // white sphere
		s->addObject(new Sphere(Vector(0, 0, -1000), 940, Vector(0., 1., 0.))); // green fore wall
		s->addObject(new Sphere(Vector(0, -1000, 0), 990, Vector(0., 0., 1.))); // blue floor
		s->addObject(new Sphere(Vector(0, 1000, 0), 940, Vector(1., 0., 0.))); // red ceiling
		s->addObject(new Sphere(Vector(-1000, 0, 0), 940, Vector(0., 1., 1.))); // cyan left wall
		s->addObject(new Sphere(Vector(1000, 0, 0), 940, Vector(1., 1., 0.))); // yellow right wall
		s->addObject(new Sphere(Vector(0, 0, 1000), 940, Vector(1., 0., 1.))); // magenta back wall
		s->addObject(new Sphere(Vector(-20, 0, 0), 10, Vector(0., 0., 0.), 1)); // mirror sphere
		s->addObject(new Sphere(Vector(20, 0, 0), 9, Vector(0., 0., 0.), 0, 1, 1.5)); // inner nested ssphere
		s->addObject(new Sphere(Vector(20, 0, 0), 10, Vector(0., 0., 0.), 0, 1.5, 1)); // outer nested sphere
		s->rand_states = new curandState[blockDim.x];
	}
	__syncthreads();
  	curand_init(123456, id, 0, s->rand_states + id);
}

__global__ void KernelLaunch(Scene *s, char *image, int W, int H, int num_rays, int num_bounce) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / W, j = index % W;
	Vector C(0, 0, 55);
	double alpha = PI/3;
	double z = -W / (2 * tan(alpha/2));
    unsigned int seed = threadIdx.x;
    Vector u_center((double)j - (double)W / 2 + 0.5, (double)H / 2 - i - 0.5, z);
    Vector color_total(0, 0, 0);
    for (int t = 0; t < num_rays; t++) {
        // Box-muller for anti-aliasing
        double sigma = 2 * pow(10, -1);
        double r1 = uniform(s->rand_states, seed);
        double r2 = uniform(s->rand_states, seed);
        Vector u = u_center + Vector(sigma * sqrt(-2 * log(r1)) * cos(2 * PI * r2), sigma * sqrt(-2 * log(r1)) * sin(2 * PI * r2), 0);
        u.normalize();
        Ray r(C, u);
        Vector color = s->getColor(r, num_bounce);
        color_total = color_total + color;
    }
    Vector color_avg = color_total / num_rays;
    image[(i * W + j) * 3 + 0] = min(std::pow(color_avg[0], 1./2.2), 255.);
    image[(i * W + j) * 3 + 1] = min(std::pow(color_avg[1], 1./2.2), 255.);
    image[(i * W + j) * 3 + 2] = min(std::pow(color_avg[2], 1./2.2), 255.);
}

__global__ void KernelDelete(Scene *s) {
 	for (auto x: s->objects) {
		delete x;
	}
	delete s->rand_states;
}

int main(int argc, char **argv) {
    if (argc != 3) {
		std::cout << "Invalid number of arguments!\nThe first argument is number of rays and the second argument is number of bounces.\n";
		return 0;
	}
	auto start_time = std::chrono::system_clock::now();

	const int num_rays = atoi(argv[1]), num_bounce = atoi(argv[2]);
	int W = 512;
	int H = 512;
	int image_size = sizeof(char) * H * W * 3;
	const int BLOCK_DIM = 128;
	int GRID_DIM = W * H / BLOCK_DIM;
	
	Scene *d_s;
    char *h_image, *d_image;
    h_image = new char[H * W * 3];

	// Increase stack size to 16KB per thread (Should be reduced in the future)
	gpuErrchk( cudaDeviceSetLimit(cudaLimitStackSize, 1<<14) );
	
	// Malloc & transfer to GPU
    gpuErrchk( cudaMalloc((void**)&d_s, sizeof(Scene)) );
    gpuErrchk( cudaMalloc((void**)&d_image, image_size) );

	// Init scene in kernel
	KernelInit<<<1, BLOCK_DIM>>>(d_s);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

	// Launch kernel
    KernelLaunch<<<GRID_DIM, BLOCK_DIM>>>(d_s, d_image, W, H, num_rays, num_bounce);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

	// Free objects in scene
	KernelDelete<<<1, 1>>>(d_s);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(d_s) );
    gpuErrchk( cudaFree(d_image) );

	stbi_write_png("image.png", W, H, 3, &h_image[0], 0);
    delete h_image;

    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> run_time = end_time-start_time;
    std::cout << "Rendering time: " << run_time.count() << " s\n";
}