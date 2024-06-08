#include <stdio.h>
#include <math.h>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
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
#define MAX_RAY_DEPTH 10

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
	__device__ __host__ Vector(double x = 0, double y = 0, double z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	__device__ __host__ double norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	__device__ __host__ double norm() const {
		return sqrt(norm2());
	}
	__device__ __host__ void normalize() {
		double n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
	}
	__device__ __host__ double operator[](int i) const { return data[i]; };
	__device__ __host__ double& operator[](int i) { return data[i]; };
	double data[3];

	__device__ void print(){
		printf("%d %d %d\n", data[0], data[1], data[2]);
	}
};

__device__ __host__ Vector operator+(const Vector& a, const Vector& b) {
	return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
__device__ __host__ Vector operator-(const Vector& a, const Vector& b) {
	return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
__device__ __host__ Vector operator-(const Vector& a) {
	return Vector(-a[0], -a[1], -a[2]);
}
__device__ __host__ Vector operator*(const double a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
__device__ __host__ Vector operator*(const Vector& a, const double b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
// Element wise vector multiplication
__device__ __host__ Vector operator*(const Vector& a, const Vector& b) {
	return Vector(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
__device__ __host__ Vector operator/(const Vector& a, const double b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
__device__ __host__ double dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
__device__ __host__ Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

class Ray {
public:
	__device__ Ray(const Vector &O, const Vector &u, double refraction_index = 1.) : O(O), u(u), refraction_index(refraction_index) {};
	// ...
	Vector O, u;
	double refraction_index;
};

// class Geometry {
// public:
// 	__device__ Geometry(const Vector &C, double R, const Vector &albedo, bool mirror = 0, double in_refraction_index = 1, double out_refraction_index = 1): C(C), R(R), albedo(albedo), id(-1),
// 	mirror(mirror), in_refraction_index(in_refraction_index), out_refraction_index(out_refraction_index) {}
// 	__device__ Geometry(): id(-1), mirror(0), in_refraction_index(1), out_refraction_index(1) {};
	
// 	Vector C;
//     double R;
// 	Vector albedo;
// 	int id;
// 	bool mirror;
// 	double in_refraction_index;
// 	double out_refraction_index;
// 	__device__ virtual bool intersect(const Ray& r, double &t, Vector &N) { return 0; };
// };
	// __device__ bool intersect(const Ray &r, double &t, Vector &N) {
	// 	double delta = SQR(dot(r.u, r.O - C)) - ((r.O - C).norm2() - R*R);
	// 	if (delta < 0)
	// 		return 0;
	// 	double t1 = dot(r.u, C - r.O) - sqrt(delta); // first intersection
	// 	double t2 = dot(r.u, C - r.O) + sqrt(delta); // second intersection
	// 	if (t2 < 0)
	// 		return 0;
	// 	t = t1 < 0 ? t2 : t1;
	// 	N = r.O + t * r.u - C;
	// 	N.normalize();
	// 	return 1;
	// }
// };

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
		// printf("Intersect!\n");
		return 1;
	}
};


/* Start of code derived from Prof Bonnel's code */
class TriangleIndices {
public:
	__device__ __host__ TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1, bool added = false) : vtxi(vtxi), vtxj(vtxj), vtxk(vtxk), uvi(uvi), uvj(uvj), uvk(uvk), ni(ni), nj(nj), nk(nk), group(group) {
	};
	int vtxi, vtxj, vtxk; // indices within the vertex coordinates array
	int uvi, uvj, uvk;  // indices within the uv coordinates array
	int ni, nj, nk;  // indices within the normals array
	int group;       // face group
};

template <typename T> __device__ void swap ( T& a, T& b ) {
  T c(a); a=b; b=c;
}

class BoundingBox {
public:
	Vector mn, mx;

	__device__ BoundingBox(): mn(Vector(INF, INF, INF)), mx(Vector(-INF, -INF, -INF)) {};

	__device__ inline void update(const Vector &vec) {
		mn[0] = min(mn[0], vec[0]);
		mn[1] = min(mn[1], vec[1]);
		mn[2] = min(mn[2], vec[2]);
		mx[0] = max(mx[0], vec[0]);
		mx[1] = max(mx[1], vec[1]);
		mx[2] = max(mx[2], vec[2]);
	}

	__device__ inline bool intersect(const Ray &r, double &t) {
		double t0x = (mn[0] - r.O[0]) / r.u[0];
		double t0y = (mn[1] - r.O[1]) / r.u[1];
		double t0z = (mn[2] - r.O[2]) / r.u[2];
		double t1x = (mx[0] - r.O[0]) / r.u[0];
		double t1y = (mx[1] - r.O[1]) / r.u[1];
		double t1z = (mx[2] - r.O[2]) / r.u[2];
		if (t0x > t1x) swap(t0x, t1x);
		if (t0y > t1y) swap(t0y, t1y);
		if (t0z > t1z) swap(t0z, t1z);
		return min(t1x, min(t1y, t1z)) > max(t0x, max(t0y, t0z));
	}
};

class BVH {
public:
	BVH *left, *right;
	BoundingBox bb;
	int triangle_start, triangle_end;
};

class TriangleMesh: public Geometry {
public:
  	// __device__ ~TriangleMesh() {};
	__device__ TriangleMesh() {};

	#define between(A, B, C) ((A) <= (B) && (B) <= (C))

	__device__ BoundingBox compute_bbox(int triangle_start, int triangle_end) {
		BoundingBox bb;
		for (int i = triangle_start; i < triangle_end; i++) {
			bb.update(vertices[indices[i].vtxi]);
			bb.update(vertices[indices[i].vtxj]);
			bb.update(vertices[indices[i].vtxk]);
		}
		return bb;
	}

	__device__ void buildBVH(BVH* cur, int triangle_start, int triangle_end) {
		// std::cout << cur << ' ' << triangle_start << ' ' << triangle_end << '\n';
		// printf("%d %d\n", triangle_start, triangle_end);
		cur->triangle_start = triangle_start;
		cur->triangle_end = triangle_end;
		cur->left = NULL;
		cur->right = NULL;
		cur->bb = compute_bbox(triangle_start, triangle_end);

		Vector diag = cur->bb.mx - cur->bb.mn;
		int max_axis;
		if (diag[0] >= diag[1] && diag[0] >= diag[2])
			max_axis = 0;
		else if (diag[1] >= diag[0] && diag[1] >= diag[2])
			max_axis = 1;
		else
			max_axis = 2;

		int pivot = triangle_start;
		double split = (cur->bb.mn[max_axis] + cur->bb.mx[max_axis]) / 2;
		for (int i = triangle_start; i < triangle_end; i++) {
			double cen = (vertices[indices[i].vtxi][max_axis] + vertices[indices[i].vtxj][max_axis] + vertices[indices[i].vtxk][max_axis]) / 3;
			if (cen < split) {
				swap(indices[i], indices[pivot]);
				pivot++;
			}
		}

		if (pivot <= triangle_start || pivot >= triangle_end - 1 || triangle_end - triangle_start < 5) {
			return;
		}
		cur->left = new BVH;
		cur->right = new BVH;
		buildBVH(cur->left, triangle_start, pivot);
		buildBVH(cur->right, pivot, triangle_end);
	}

	__device__ bool moller_trumbore(const Vector &A, const Vector &B, const Vector &C, Vector& N, const Ray &r, double &t) {
		Vector e1 = B - A;
		Vector e2 = C - A;
		N = cross(e1, e2);
		if (dot(r.u, N) == 0) return 0;
		double beta = dot(e2, cross(A - r.O, r.u)) / dot(r.u, N);
		double gamma = - dot(e1, cross(A - r.O, r.u)) / dot(r.u, N);
		if (!between(0, beta, 1) || !between(0, gamma, 1))	return 0;
		t = dot(A - r.O, N) / dot(r.u, N);
		return beta + gamma <= 1 && t > 0;
	}
	
	__device__ bool intersect(const Ray &r, double &t, Vector &N) override {
		// printf("inter!\n");
		// printf("Intersect mesh\n");
		double t_tmp;
		if (!bvh.bb.intersect(r, t_tmp)) return 0;
		BVH* s[30];
		int s_size = 0;
		s[s_size++] = &bvh;


		double t_min = INF;
		while(s_size) {
			const BVH* cur = s[s_size-1];
			s_size--;
			if (cur->left) {
				double t_left, t_right;
				bool ok_left = cur->left->bb.intersect(r, t_left);
				bool ok_right = cur->right->bb.intersect(r, t_right);
				// printf("%d %d\n", ok_left, ok_right);
				if (ok_left) s[s_size++] = cur->left;
				if (ok_right) s[s_size++] = cur->right;
			} else {
				// Leaf
				for (int i = cur->triangle_start; i < cur->triangle_end; i++) {
					double t_cur;
					Vector A = vertices[indices[i].vtxi], B = vertices[indices[i].vtxj], C = vertices[indices[i].vtxk];
					Vector N_triangle;
					bool inter = moller_trumbore(A, B, C, N_triangle, r, t_cur);
					if (!inter) continue;
					if (t_cur > 0 && t_cur < t_min) {
						t_min = t_cur;
						N = N_triangle;
					}
				} 
			}
		}
		N.normalize();
		t = t_min;
		return t_min != INF;
	}
	TriangleIndices* indices;
	int indices_size;
	Vector* vertices;
	int vertices_size;
	// Vector* normals;
	// int normals_size;
	// Vector* uvs;
	// int uvs_size;
	// Vector* vertexcolors;
	// int vertexcolors_size;
	BoundingBox bb;
	BVH bvh;
};

class TriangleMeshHost {
public:
 	~TriangleMeshHost() {}
	TriangleMeshHost() {};

	void readOBJ(const char* obj) {

		char matfile[255];
		char grp[255];

		FILE* f;
		f = fopen(obj, "r");
		if (f == NULL) {
			printf("Error opening file!\n");
			return;
		}
		int curGroup = -1;
		while (!feof(f)) {
			char line[255];
			if (!fgets(line, 255, f)) break;

			std::string linetrim(line);
			linetrim.erase(linetrim.find_last_not_of(" \r\t") + 1);
			strcpy(line, linetrim.c_str());

			if (line[0] == 'u' && line[1] == 's') {
				sscanf(line, "usemtl %[^\n]\n", grp);
				curGroup++;
			}

			if (line[0] == 'v' && line[1] == ' ') {
				Vector vec;

				Vector col;
				if (sscanf(line, "v %lf %lf %lf %lf %lf %lf\n", &vec[0], &vec[1], &vec[2], &col[0], &col[1], &col[2]) == 6) {
					col[0] = std::min(1., std::max(0., col[0]));
					col[1] = std::min(1., std::max(0., col[1]));
					col[2] = std::min(1., std::max(0., col[2]));

					vertices.push_back(vec);
					vertexcolors.push_back(col);

				} else {
					sscanf(line, "v %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
					vec = vec*0.8+Vector(0, -10, 0);
					vertices.push_back(vec);
				}
			}
			if (line[0] == 'v' && line[1] == 'n') {
				Vector vec;
				sscanf(line, "vn %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
				normals.push_back(vec);
			}
			if (line[0] == 'v' && line[1] == 't') {
				Vector vec;
				sscanf(line, "vt %lf %lf\n", &vec[0], &vec[1]);
				uvs.push_back(vec);
			}
			if (line[0] == 'f') {
				TriangleIndices t;
				int i0, i1, i2, i3;
				int j0, j1, j2, j3;
				int k0, k1, k2, k3;
				int nn;
				t.group = curGroup;

				char* consumedline = line + 1;
				int offset;

				nn = sscanf(consumedline, "%u/%u/%u %u/%u/%u %u/%u/%u%n", &i0, &j0, &k0, &i1, &j1, &k1, &i2, &j2, &k2, &offset);
				if (nn == 9) {
					if (i0 < 0) t.vtxi = vertices.size() + i0; else	t.vtxi = i0 - 1;
					if (i1 < 0) t.vtxj = vertices.size() + i1; else	t.vtxj = i1 - 1;
					if (i2 < 0) t.vtxk = vertices.size() + i2; else	t.vtxk = i2 - 1;
					if (j0 < 0) t.uvi = uvs.size() + j0; else	t.uvi = j0 - 1;
					if (j1 < 0) t.uvj = uvs.size() + j1; else	t.uvj = j1 - 1;
					if (j2 < 0) t.uvk = uvs.size() + j2; else	t.uvk = j2 - 1;
					if (k0 < 0) t.ni = normals.size() + k0; else	t.ni = k0 - 1;
					if (k1 < 0) t.nj = normals.size() + k1; else	t.nj = k1 - 1;
					if (k2 < 0) t.nk = normals.size() + k2; else	t.nk = k2 - 1;
					indices.push_back(t);
				} else {
					nn = sscanf(consumedline, "%u/%u %u/%u %u/%u%n", &i0, &j0, &i1, &j1, &i2, &j2, &offset);
					if (nn == 6) {
						if (i0 < 0) t.vtxi = vertices.size() + i0; else	t.vtxi = i0 - 1;
						if (i1 < 0) t.vtxj = vertices.size() + i1; else	t.vtxj = i1 - 1;
						if (i2 < 0) t.vtxk = vertices.size() + i2; else	t.vtxk = i2 - 1;
						if (j0 < 0) t.uvi = uvs.size() + j0; else	t.uvi = j0 - 1;
						if (j1 < 0) t.uvj = uvs.size() + j1; else	t.uvj = j1 - 1;
						if (j2 < 0) t.uvk = uvs.size() + j2; else	t.uvk = j2 - 1;
						indices.push_back(t);
					} else {
						nn = sscanf(consumedline, "%u %u %u%n", &i0, &i1, &i2, &offset);
						if (nn == 3) {
							if (i0 < 0) t.vtxi = vertices.size() + i0; else	t.vtxi = i0 - 1;
							if (i1 < 0) t.vtxj = vertices.size() + i1; else	t.vtxj = i1 - 1;
							if (i2 < 0) t.vtxk = vertices.size() + i2; else	t.vtxk = i2 - 1;
							indices.push_back(t);
						} else {
							nn = sscanf(consumedline, "%u//%u %u//%u %u//%u%n", &i0, &k0, &i1, &k1, &i2, &k2, &offset);
							if (i0 < 0) t.vtxi = vertices.size() + i0; else	t.vtxi = i0 - 1;
							if (i1 < 0) t.vtxj = vertices.size() + i1; else	t.vtxj = i1 - 1;
							if (i2 < 0) t.vtxk = vertices.size() + i2; else	t.vtxk = i2 - 1;
							if (k0 < 0) t.ni = normals.size() + k0; else	t.ni = k0 - 1;
							if (k1 < 0) t.nj = normals.size() + k1; else	t.nj = k1 - 1;
							if (k2 < 0) t.nk = normals.size() + k2; else	t.nk = k2 - 1;
							indices.push_back(t);
						}
					}
				}

				consumedline = consumedline + offset;

				while (true) {
					if (consumedline[0] == '\n') break;
					if (consumedline[0] == '\0') break;
					nn = sscanf(consumedline, "%u/%u/%u%n", &i3, &j3, &k3, &offset);
					TriangleIndices t2;
					t2.group = curGroup;
					if (nn == 3) {
						if (i0 < 0) t2.vtxi = vertices.size() + i0; else	t2.vtxi = i0 - 1;
						if (i2 < 0) t2.vtxj = vertices.size() + i2; else	t2.vtxj = i2 - 1;
						if (i3 < 0) t2.vtxk = vertices.size() + i3; else	t2.vtxk = i3 - 1;
						if (j0 < 0) t2.uvi = uvs.size() + j0; else	t2.uvi = j0 - 1;
						if (j2 < 0) t2.uvj = uvs.size() + j2; else	t2.uvj = j2 - 1;
						if (j3 < 0) t2.uvk = uvs.size() + j3; else	t2.uvk = j3 - 1;
						if (k0 < 0) t2.ni = normals.size() + k0; else	t2.ni = k0 - 1;
						if (k2 < 0) t2.nj = normals.size() + k2; else	t2.nj = k2 - 1;
						if (k3 < 0) t2.nk = normals.size() + k3; else	t2.nk = k3 - 1;
						indices.push_back(t2);
						consumedline = consumedline + offset;
						i2 = i3;
						j2 = j3;
						k2 = k3;
					} else {
						nn = sscanf(consumedline, "%u/%u%n", &i3, &j3, &offset);
						if (nn == 2) {
							if (i0 < 0) t2.vtxi = vertices.size() + i0; else	t2.vtxi = i0 - 1;
							if (i2 < 0) t2.vtxj = vertices.size() + i2; else	t2.vtxj = i2 - 1;
							if (i3 < 0) t2.vtxk = vertices.size() + i3; else	t2.vtxk = i3 - 1;
							if (j0 < 0) t2.uvi = uvs.size() + j0; else	t2.uvi = j0 - 1;
							if (j2 < 0) t2.uvj = uvs.size() + j2; else	t2.uvj = j2 - 1;
							if (j3 < 0) t2.uvk = uvs.size() + j3; else	t2.uvk = j3 - 1;
							consumedline = consumedline + offset;
							i2 = i3;
							j2 = j3;
							indices.push_back(t2);
						} else {
							nn = sscanf(consumedline, "%u//%u%n", &i3, &k3, &offset);
							if (nn == 2) {
								if (i0 < 0) t2.vtxi = vertices.size() + i0; else	t2.vtxi = i0 - 1;
								if (i2 < 0) t2.vtxj = vertices.size() + i2; else	t2.vtxj = i2 - 1;
								if (i3 < 0) t2.vtxk = vertices.size() + i3; else	t2.vtxk = i3 - 1;
								if (k0 < 0) t2.ni = normals.size() + k0; else	t2.ni = k0 - 1;
								if (k2 < 0) t2.nj = normals.size() + k2; else	t2.nj = k2 - 1;
								if (k3 < 0) t2.nk = normals.size() + k3; else	t2.nk = k3 - 1;								
								consumedline = consumedline + offset;
								i2 = i3;
								k2 = k3;
								indices.push_back(t2);
							} else {
								nn = sscanf(consumedline, "%u%n", &i3, &offset);
								if (nn == 1) {
									if (i0 < 0) t2.vtxi = vertices.size() + i0; else	t2.vtxi = i0 - 1;
									if (i2 < 0) t2.vtxj = vertices.size() + i2; else	t2.vtxj = i2 - 1;
									if (i3 < 0) t2.vtxk = vertices.size() + i3; else	t2.vtxk = i3 - 1;
									consumedline = consumedline + offset;
									i2 = i3;
									indices.push_back(t2);
								} else {
									consumedline = consumedline + 1;
								}
							}
						}
					}
				}

			}

		}
		fclose(f);

	}

	std::vector<TriangleIndices> indices;
	std::vector<Vector> vertices;
	std::vector<Vector> normals;
	std::vector<Vector> uvs;
	std::vector<Vector> vertexcolors;
};

class Scene {
public:
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
				// printf("OKOK\n");
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

	// __device__ Vector getColor(const Ray& ray, int ray_depth) {
	// 	if (ray_depth < 0) return Vector(0., 0., 0.); // terminates recursion at some <- point
	// 	Vector P, N;
	// 	int sphere_id = -1;
	// 	bool inter = intersect_all(ray, P, N, sphere_id);
	// 	Vector color;
	// 	if (inter) {
	// 		if (objects[sphere_id]->mirror) {
	// 			// Reflection
	// 			double epsilon = 1e-6;
	// 			Vector P_adjusted = P + epsilon * N;
	// 			Vector new_direction = ray.u - 2 * dot(ray.u, N) * N;
	// 			Ray reflected_ray(P_adjusted, new_direction, ray.refraction_index);
	// 			return getColor(reflected_ray, ray_depth - 1);
	// 		} else if (objects[sphere_id]->in_refraction_index != objects[sphere_id]->out_refraction_index) {
	// 			// Refraction
	// 			double epsilon = 1e-6;
	// 			double refract_ratio;
	// 			bool out2in = ray.refraction_index == objects[sphere_id]->out_refraction_index;
	// 			if (out2in) { 
	// 				// outside to inside
	// 				refract_ratio = objects[sphere_id]->out_refraction_index / objects[sphere_id]->in_refraction_index;
	// 			} else { 
	// 				// inside to outside
	// 				refract_ratio = objects[sphere_id]->in_refraction_index / objects[sphere_id]->out_refraction_index;
	// 				N = -N;
	// 			}
	// 			if (((out2in && ray.refraction_index > objects[sphere_id]->in_refraction_index) ||
	// 				(!out2in && ray.refraction_index > objects[sphere_id]->out_refraction_index)) &&
	// 				SQR(refract_ratio) * (1 - SQR(dot(ray.u, N))) > 1) { 
	// 				// total internal reflection
	// 				return getColor(Ray(P + epsilon * N, ray.u - 2 * dot(ray.u, N) * N, ray.refraction_index), ray_depth - 1);
	// 			}
	// 			Vector P_adjusted = P - epsilon * N;
	// 			Vector N_component = - sqrt(1 - SQR(refract_ratio) * (1 - SQR(dot(ray.u, N)))) * N;
	// 			Vector T_component = refract_ratio * (ray.u - dot(ray.u, N) * N);
	// 			Vector new_direction = N_component + T_component;
	// 			if (out2in) {
	// 				return getColor(Ray(P_adjusted, new_direction, objects[sphere_id]->in_refraction_index), ray_depth - 1);
	// 			} else {
	// 				return getColor(Ray(P_adjusted, new_direction, objects[sphere_id]->out_refraction_index), ray_depth - 1);
	// 			}
	// 		} else {
	// 			// 	handle diffuse surfaces
	// 			// 	Get shadow
	// 			Vector P_prime;
	// 			int sphere_id_shadow;
	// 			double epsilon = 1e-6;
	// 			Vector P_adjusted = P + epsilon * N;
	// 			Vector direct_color, indirect_color;
	// 			Vector N_prime;
	// 			bool _ = intersect_all(Ray(P_adjusted, NORMED_VEC(L - P_adjusted)), P_prime, N_prime, sphere_id_shadow);
				
	// 			if ((P_prime - P_adjusted).norm2() <= (L - P_adjusted).norm2()) {
	// 				// Is shadow
	// 				direct_color = Vector(0, 0, 0);
	// 			} else {
	// 				// Get direct color
	// 				Geometry* S = objects[sphere_id];
	// 				Vector wlight = L - P;
	// 				wlight.normalize();
	// 				double l = intensity / (4 * PI * (L - P).norm2()) * max(dot(N, wlight), 0.);
	// 				direct_color = l * S->albedo / PI;
	// 			}
	// 			// Get indirect color by launching ray
	// 			unsigned int seed = threadIdx.x;
	// 			double r1 = uniform(rand_states, seed);
	// 			double r2 = uniform(rand_states, seed);
	// 			double x = cos(2 * PI * r1) * sqrt(1 - r2);
	// 			double y = sin(2 * PI * r1) * sqrt(1 - r2);
	// 			double z = sqrt(r2);
	// 			Vector T1;
	// 			if (abs(N[1]) != 0 && abs(N[0]) != 0) {
	// 				T1 = Vector(-N[1], N[0], 0);
	// 			} else {
	// 				T1 = Vector(-N[2], 0, N[0]);
	// 			}
	// 			T1.normalize();
	// 			Vector T2 = cross(N, T1);
	// 			Vector random_direction = x * T1 + y * T2 + z * N;
	// 			indirect_color = ((Geometry *)objects[sphere_id])->albedo * getColor(Ray(P_adjusted, random_direction), ray_depth - 1);
	// 			color = direct_color + indirect_color;
	// 		}
	// 	}
	// 	return color;
	// }

	__device__ Vector getColorIterative(const Ray& input_ray, int max_ray_depth) {
		int types[MAX_RAY_DEPTH];
		Vector direct_colors[MAX_RAY_DEPTH];
		Vector indirect_albedos[MAX_RAY_DEPTH];
		Ray ray = input_ray;
		for (int ray_depth = 0; ray_depth < max_ray_depth; ray_depth++) {
			Vector P, N;
			int sphere_id = -1;
			bool inter = intersect_all(ray, P, N, sphere_id);
			Vector color;
			if (inter) {
				if (objects[sphere_id]->mirror) {
					// Reflection
					types[ray_depth] = 0;
					double epsilon = 1e-6;
					Vector P_adjusted = P + epsilon * N;
					Vector new_direction = ray.u - 2 * dot(ray.u, N) * N;
					Ray reflected_ray(P_adjusted, new_direction, ray.refraction_index);
					ray = reflected_ray;
				} else if (objects[sphere_id]->in_refraction_index != objects[sphere_id]->out_refraction_index) {
					// Refraction
					types[ray_depth] = 0;
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
						ray = Ray(P + epsilon * N, ray.u - 2 * dot(ray.u, N) * N, ray.refraction_index);
						continue;
					}
					Vector P_adjusted = P - epsilon * N;
					Vector N_component = - sqrt(1 - SQR(refract_ratio) * (1 - SQR(dot(ray.u, N)))) * N;
					Vector T_component = refract_ratio * (ray.u - dot(ray.u, N) * N);
					Vector new_direction = N_component + T_component;
					if (out2in) {
						ray = Ray(P_adjusted, new_direction, objects[sphere_id]->in_refraction_index);
					} else {
						ray = Ray(P_adjusted, new_direction, objects[sphere_id]->out_refraction_index);
					}
				} else {
					// 	handle diffuse surfaces
					// 	Get shadow
					Vector P_prime;
					int sphere_id_shadow;
					double epsilon = 1e-6;
					Vector P_adjusted = P + epsilon * N;
					Vector N_prime;
					bool _ = intersect_all(Ray(P_adjusted, NORMED_VEC(L - P_adjusted)), P_prime, N_prime, sphere_id_shadow);
					
					if ((P_prime - P_adjusted).norm2() <= (L - P_adjusted).norm2()) {
						// Is shadow
						direct_colors[ray_depth] = Vector(0, 0, 0);
					} else {
						// Get direct color
						Geometry* S = objects[sphere_id];
						Vector wlight = L - P;
						wlight.normalize();
						double l = intensity / (4 * PI * (L - P).norm2()) * max(dot(N, wlight), 0.);
						direct_colors[ray_depth] = l * S->albedo / PI;
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
					ray = Ray(P_adjusted, random_direction);
					indirect_albedos[ray_depth] = ((Geometry *)objects[sphere_id])->albedo;
					types[ray_depth] = 1;
				}
			}
		}
		Vector ans_color;
		for (int i = max_ray_depth - 1; i >= 0; i--) {
			if (types[i]) {
				// Hits a diffusion object
				ans_color = indirect_albedos[i] * ans_color + direct_colors[i];
			}
		}
		return ans_color;
	}

	Geometry* objects[10];
    int objects_size = 0;
	double intensity = 3e10;
	Vector L;
	curandState* rand_states;
};

__global__ void KernelInit(TriangleMesh *cat, TriangleIndices *indices, int indices_size, Vector *vertices, int vertices_size){
	auto id = threadIdx.x + blockIdx.x * blockDim.x;

	if(!id){
		cat->albedo = Vector(0.25, 0.25, 0.25);
	 	cat->indices_size = indices_size;
		cat->indices = indices;
		cat->vertices_size = vertices_size;
		cat->vertices = vertices;
		// printf("%d %d\n", indices_size, vertices_size);
		// cat->normals_size;
		// cat->normals;
		// cat->uvs_size;
		// cat->uvs;
		// cat->vertexcolors_size;
		// cat->vertexcolors;
		cat->bvh.bb = cat->compute_bbox(0, cat->indices_size);
		cat->buildBVH(&(cat->bvh), 0, cat->indices_size);
	}
}

__global__ void KernelLaunch(double *colors, int W, int H, int num_rays, int num_bounce, TriangleMesh *d_mesh) {
	extern __shared__ double shared_memory[];
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	double *shared_colors = shared_memory;
	Sphere *shared_objects = (Sphere *)&shared_colors[blockDim.x * 3];
	curandState *shared_rand_states = (curandState *)&shared_objects[10];
	Scene *shared_scene = (Scene *)&shared_rand_states[blockDim.x];
	TriangleMesh *shared_mesh = (TriangleMesh *)&shared_scene[1];

	if (!threadIdx.x) {
		shared_scene->L = Vector(-10., 20., 40.);
		shared_scene->objects_size = 0;
		shared_scene->intensity = 3e10;

		Sphere tmp = Sphere(Vector(0, 0, -1000), 940, Vector(0., 1., 0.));
		memcpy(&shared_objects[shared_scene->objects_size], &tmp, sizeof(Sphere));
		// shared_objects[shared_scene->objects_size] =  Sphere(Vector(0, 0, -1000), 940, Vector(0., 1., 0.));
		// tmp->id = shared_scene->objects_size;
		shared_objects[shared_scene->objects_size].id = shared_scene->objects_size;
		shared_scene->objects[shared_scene->objects_size] = (Geometry *)&shared_objects[shared_scene->objects_size];
		++shared_scene->objects_size;

		TriangleMesh mesh = TriangleMesh();
		mesh.albedo = d_mesh->albedo;
		mesh.vertices = d_mesh->vertices;
		mesh.indices = d_mesh->indices;
		mesh.vertices_size = d_mesh->vertices_size;
		mesh.indices_size = d_mesh->indices_size;
		mesh.bvh = d_mesh->bvh;
		mesh.bb = d_mesh->bb;
		memcpy(shared_mesh, &mesh, sizeof(TriangleMesh));

		// printf("%d %d\n", shared_mesh->vertices_size, shared_mesh->indices_size);
		shared_mesh->id = shared_scene->objects_size;
		shared_scene->objects[shared_scene->objects_size] = (Geometry *)shared_mesh;
		++shared_scene->objects_size;

		tmp = Sphere(Vector(0, -1000, 0), 990, Vector(0., 0., 1.));
		memcpy(&shared_objects[shared_scene->objects_size], &tmp, sizeof(Sphere));
		shared_objects[shared_scene->objects_size].id = shared_scene->objects_size;
		shared_scene->objects[shared_scene->objects_size] = (Geometry *)&shared_objects[shared_scene->objects_size];
		++shared_scene->objects_size;


		tmp = Sphere(Vector(0, 1000, 0), 940, Vector(1., 0., 0.));
		memcpy(&shared_objects[shared_scene->objects_size], &tmp, sizeof(Sphere));
		shared_objects[shared_scene->objects_size].id = shared_scene->objects_size;
		shared_scene->objects[shared_scene->objects_size] = (Geometry *)&shared_objects[shared_scene->objects_size];
		++shared_scene->objects_size;

		tmp = Sphere(Vector(-1000, 0, 0), 940, Vector(0., 1., 1.));
		memcpy(&shared_objects[shared_scene->objects_size], &tmp, sizeof(Sphere));
		shared_objects[shared_scene->objects_size].id = shared_scene->objects_size;
		shared_scene->objects[shared_scene->objects_size] = (Geometry *)&shared_objects[shared_scene->objects_size];
		++shared_scene->objects_size;


		tmp = Sphere(Vector(1000, 0, 0), 940, Vector(1., 1., 0.));
		memcpy(&shared_objects[shared_scene->objects_size], &tmp, sizeof(Sphere));
		shared_objects[shared_scene->objects_size].id = shared_scene->objects_size;
		shared_scene->objects[shared_scene->objects_size] = (Geometry *)&shared_objects[shared_scene->objects_size];
		++shared_scene->objects_size;

		tmp = Sphere(Vector(0, 0, 1000), 940, Vector(1., 0., 1.));
		memcpy(&shared_objects[shared_scene->objects_size], &tmp, sizeof(Sphere));
		shared_objects[shared_scene->objects_size].id = shared_scene->objects_size;
		shared_scene->objects[shared_scene->objects_size] = (Geometry *)&shared_objects[shared_scene->objects_size];
		++shared_scene->objects_size;

		// shared_objects[shared_scene->objects_size] = (Geometry) cat;
		// shared_objects[shared_scene->objects_size].id= shared_scene->objects_size;
		// ++shared_scene->objects_size;
		// shared_objects[shared_scene->objects_size] = Geometry(Vector(-20, 0, 0), 10, Vector(0., 0., 0.), 1);
		// shared_objects[shared_scene->objects_size].id = shared_scene->objects_size;
		// shared_scene->objects[shared_scene->objects_size] = &shared_objects[shared_scene->objects_size];
		// ++shared_scene->objects_size;
		// shared_objects[shared_scene->objects_size] = Geometry(Vector(20, 0, 0), 9, Vector(0., 0., 0.), 0, 1, 1.5);
		// shared_objects[shared_scene->objects_size].id = shared_scene->objects_size;
		// shared_scene->objects[shared_scene->objects_size] = &shared_objects[shared_scene->objects_size];
		// ++shared_scene->objects_size;
		// shared_objects[shared_scene->objects_size] = Geometry(Vector(20, 0, 0), 10, Vector(0., 0., 0.), 0, 1.5, 1);
		// shared_objects[shared_scene->objects_size].id = shared_scene->objects_size;
		// shared_scene->objects[shared_scene->objects_size] = &shared_objects[shared_scene->objects_size];
		// ++shared_scene->objects_size;
		shared_scene->rand_states = shared_rand_states;
	}
	__syncthreads();
	// printf("Sync\n");
	
	curand_init(123456, index, 0, shared_scene->rand_states + threadIdx.x);
    int i = (index / num_rays) / W, j = (index / num_rays) % W;
	Vector C(0, 0, 55);
	double alpha = PI/3;
	double z = -W / (2 * tan(alpha/2));
    unsigned int seed = threadIdx.x;
    Vector u_center((double)j - (double)W / 2 + 0.5, (double)H / 2 - i - 0.5, z);
	// Box-muller for anti-aliasing
	double sigma = 0.2;
	double r1 = uniform(shared_scene->rand_states, seed);
	double r2 = uniform(shared_scene->rand_states, seed);
	Vector u = u_center + Vector(sigma * sqrt(-2 * log(r1)) * cos(2 * PI * r2), sigma * sqrt(-2 * log(r1)) * sin(2 * PI * r2), 0);
	u.normalize();
	Ray r(C, u);
	Vector color = shared_scene->getColorIterative(r, num_bounce);
	shared_colors[threadIdx.x * 3 + 0] = color[0];
    shared_colors[threadIdx.x * 3 + 1] = color[1];
    shared_colors[threadIdx.x * 3 + 2] = color[2];
	__syncthreads();
	// color.print();
	colors[blockIdx.x * blockDim.x * 3 + blockDim.x * 0 + threadIdx.x] = shared_colors[blockDim.x * 0 + threadIdx.x];
	colors[blockIdx.x * blockDim.x * 3 + blockDim.x * 1 + threadIdx.x] = shared_colors[blockDim.x * 1 + threadIdx.x];
	colors[blockIdx.x * blockDim.x * 3 + blockDim.x * 2 + threadIdx.x] = shared_colors[blockDim.x * 2 + threadIdx.x];
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
	int colors_size = sizeof(double) * H * W * 3 * num_rays;
	const int BLOCK_DIM = 128;
	int GRID_DIM = W * H * num_rays / BLOCK_DIM;
	
	Scene *d_s;
	double *h_colors, *d_colors;
    char *image;
	h_colors = new double[H * W * 3 * num_rays];
    image = new char[H * W * 3];

	gpuErrchk( cudaDeviceSetLimit(cudaLimitStackSize, 1<<14) );
	
	// Malloc & transfer to GPU
    gpuErrchk( cudaMalloc((void**)&d_s, sizeof(Scene)) );
    gpuErrchk( cudaMalloc((void**)&d_colors, colors_size) );
	TriangleMeshHost* mesh_ptr = new TriangleMeshHost(); // cat

	TriangleMesh *d_mesh = NULL;

	const char *path = "cadnav.com_model/Models_F0202A090/cat.obj";
	mesh_ptr->readOBJ(path);
	TriangleIndices* d_indices;
	Vector* d_vertices;
    gpuErrchk( cudaMalloc((void**)&d_indices, mesh_ptr->indices.size() * sizeof(TriangleIndices)) );
    gpuErrchk( cudaMemcpy(d_indices, &(mesh_ptr->indices[0]), mesh_ptr->indices.size() * sizeof(TriangleIndices), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc((void**)&d_vertices, mesh_ptr->vertices.size() * sizeof(Vector)) );
    gpuErrchk( cudaMemcpy(d_vertices, &(mesh_ptr->vertices[0]), mesh_ptr->vertices.size() * sizeof(Vector), cudaMemcpyHostToDevice) );
	
	gpuErrchk( cudaMalloc((void**)&d_mesh, sizeof(TriangleMesh)) );
	KernelInit<<<1, 1>>>(d_mesh, d_indices, mesh_ptr->indices.size(), d_vertices, mesh_ptr->vertices.size());
	gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    KernelLaunch<<<GRID_DIM, BLOCK_DIM, sizeof(double) * BLOCK_DIM * 3 + sizeof(Geometry) * 10 + sizeof(TriangleMesh) + sizeof(curandState) * BLOCK_DIM + sizeof(Scene)>>>(d_colors, W, H, num_rays, num_bounce,  d_mesh);
	gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
	
	cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, KernelLaunch);

    // std::cout << "Shared memory per block: " << attr.sharedSizeBytes << " bytes" << std::endl;

    gpuErrchk( cudaMemcpy(h_colors, d_colors, colors_size, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(d_s) );
    gpuErrchk( cudaFree(d_colors) );
    gpuErrchk( cudaFree(d_indices) );
    gpuErrchk( cudaFree(d_vertices) );

	for (int i = 0; i < H; ++i) {
		for (int j = 0; j < W; ++j) {
			Vector colors_sum;
			for (int t = 0; t < num_rays; ++t) {
				colors_sum = colors_sum + Vector(
					h_colors[((i * W + j) * num_rays + t) * 3 + 0],
					h_colors[((i * W + j) * num_rays + t) * 3 + 1],
					h_colors[((i * W + j) * num_rays + t) * 3 + 2]
				);
			}
			Vector colors_avg = colors_sum / num_rays;
			image[(i * W + j) * 3 + 0] = min(std::pow(colors_avg[0], 1./2.2), 255.);
			image[(i * W + j) * 3 + 1] = min(std::pow(colors_avg[1], 1./2.2), 255.);
			image[(i * W + j) * 3 + 2] = min(std::pow(colors_avg[2], 1./2.2), 255.);
		}
	}
	delete h_colors;
	stbi_write_png("image.png", W, H, 3, &image[0], 0);
    delete image;

	// int device;
    // cudaGetDevice(&device);

    // cudaDeviceProp props;
    // cudaGetDeviceProperties(&props, device);

    // std::cout << "Device name: " << props.name << std::endl;
    // std::cout << "Shared memory per block: " << props.sharedMemPerBlock << " bytes" << std::endl;
    // std::cout << "Shared memory per multiprocessor: " << props.sharedMemPerMultiprocessor << " bytes" << std::endl;

    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> run_time = end_time-start_time;
    std::cout << "Rendering time: " << run_time.count() << " s\n";
}
