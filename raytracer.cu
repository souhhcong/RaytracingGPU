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
#define PRINT_VEC(v) (printf("%s: (%f %f %f)\n", #v, (v)[0], (v)[1], (v)[2]))
#define INF (1e9+9)
#define MAX_RAY_DEPTH 10
// #define float float

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ inline float uniform(curandState *rand_states, unsigned int tid) {
    curandState local_state = rand_states[tid];
 	float RANDOM = curand_uniform( &local_state );
    rand_states[tid] = local_state;
	return RANDOM;
}

class Vector {
public:
	__device__ __host__ Vector(float x = 0, float y = 0, float z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	__device__ __host__ float norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	__device__ __host__ float norm() const {
		return sqrt(norm2());
	}
	__device__ __host__ void normalize() {
		float n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
	}
	__device__ __host__ float operator[](int i) const { return data[i]; };
	__device__ __host__ float& operator[](int i) { return data[i]; };
	float data[3];
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
__device__ __host__ Vector operator*(const float a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
__device__ __host__ Vector operator*(const Vector& a, const float b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
// Element wise vector multiplication
__device__ __host__ Vector operator*(const Vector& a, const Vector& b) {
	return Vector(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
__device__ __host__ Vector operator/(const Vector& a, const float b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
__device__ __host__ float dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
__device__ __host__ Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

class Ray {
public:
	__device__ Ray(const Vector &O, const Vector &u, float refraction_index = 1.) : O(O), u(u), refraction_index(refraction_index) {};
	// ...
	Vector O, u;
 float refraction_index;
};

class Geometry {
public:
	__device__ Geometry(const Vector &albedo, int id, bool mirror, float in_refraction_index, float out_refraction_index): albedo(albedo), id(id),
	mirror(mirror), in_refraction_index(in_refraction_index), out_refraction_index(out_refraction_index) {}
	__device__ Geometry(): mirror(0), in_refraction_index(1), out_refraction_index(1) {};

	Vector albedo;
	int id;
	bool mirror;
 float in_refraction_index;
 float out_refraction_index;
	__device__ virtual bool intersect(const Ray& r, float &t, Vector &N) { return 0; };
};

class Sphere: public Geometry {
public:
	__host__ __device__ Sphere(){};
	__device__ Sphere(const Vector &C, float R, const Vector& albedo, bool mirror = 0, float in_refraction_index = 1., float out_refraction_index = 1.) : 
	C(C), R(R), Geometry(albedo, id, mirror, in_refraction_index, out_refraction_index) {};
    Vector C;
 float R;
	__device__ bool intersect(const Ray &r, float &t, Vector &N) override {
	 float delta = SQR(dot(r.u, r.O - C)) - ((r.O - C).norm2() - R*R);
		if (delta < 0)
			return 0;
	 float t1 = dot(r.u, C - r.O) - sqrt(delta); // first intersection
	 float t2 = dot(r.u, C - r.O) + sqrt(delta); // second intersection
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
	__device__ __host__ TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1, bool added = false) : vtxi(vtxi), vtxj(vtxj), vtxk(vtxk), uvi(uvi), uvj(uvj), uvk(uvk), ni(ni), nj(nj), nk(nk), group(group){};
    int vtxi, vtxj, vtxk; // indices within the vertex coordinates array
    int uvi, uvj, uvk;    // indices within the uv coordinates array
    int ni, nj, nk;       // indices within the normals array
    int group;            // face group
};

template <typename T> __device__ __host__ void swap ( T& a, T& b ) {
  T c(a); a=b; b=c;
}

class BoundingBox {
public:
	Vector mn, mx;

	__device__ __host__ BoundingBox(): mn(Vector(INF, INF, INF)), mx(Vector(-INF, -INF, -INF)) {};
	__device__ __host__ BoundingBox(
		const Vector &mn_,
		const Vector &mx_
	) : mn(mn_),
		mx(mx_) {}

	__device__ __host__ inline void update(const Vector &vec) {
		mn[0] = min(mn[0], vec[0]);
		mn[1] = min(mn[1], vec[1]);
		mn[2] = min(mn[2], vec[2]);
		mx[0] = max(mx[0], vec[0]);
		mx[1] = max(mx[1], vec[1]);
		mx[2] = max(mx[2], vec[2]);
	}

	__device__ __host__ inline bool intersect(const Ray &r, float &t) {
	 float t0x = (mn[0] - r.O[0]) / r.u[0];
	 float t0y = (mn[1] - r.O[1]) / r.u[1];
	 float t0z = (mn[2] - r.O[2]) / r.u[2];
	 float t1x = (mx[0] - r.O[0]) / r.u[0];
	 float t1y = (mx[1] - r.O[1]) / r.u[1];
	 float t1z = (mx[2] - r.O[2]) / r.u[2];
	if (t0x > t1x) swap(t0x, t1x);
	if (t0y > t1y) swap(t0y, t1y);
	if (t0z > t1z) swap(t0z, t1z);

	// printf("%f %f %f", t0x, t0y, t0z)
	// PRINT_VEC(mn);
	// PRINT_VEC(mx);

	return min(t1x, min(t1y, t1z)) > max(t0x, max(t0y, t0z));
	}
};

class BVH {
public:
	BVH *left, *right;
	BoundingBox bb;
	int triangle_start, triangle_end;
};

class BVHDevice {
public:
	int left, right;
	BoundingBox bb;
	int triangle_start, triangle_end;
};

class TriangleMesh: public Geometry {
public:
  	// __device__ ~TriangleMesh() {};
	__device__ TriangleMesh() {};

	#define between(A, B, C) ((A) <= (B) && (B) <= (C))

	__device__ void get_smooth_normal(Ray r, TriangleIndices tid, Vector &N){
		Vector A, B, C;
		float alpha, t;

		A = vertices[tid.vtxi];
		B = vertices[tid.vtxj];
		C = vertices[tid.vtxk];

		Vector e1 = B - A;
		Vector e2 = C - A;
		N = cross(e1, e2);
		float beta = dot(e2, cross(A - r.O, r.u)) / dot(r.u, N);
		float gamma = - dot(e1, cross(A - r.O, r.u)) / dot(r.u, N);
		t = dot(A - r.O, N) / dot(r.u, N);
	
		alpha = 1 - beta - gamma;
		Vector Na, Nb, Nc;
		// printf("%d %d %d\n", tid.ni, tid.nj, tid.nk);
		Na = normals[tid.ni];
		Nb = normals[tid.nj];
		Nc = normals[tid.nk];
		N = alpha * Na + beta * Nb + gamma * Nc;
		// PRINT_VEC(Na);
		N.normalize();
	}

	__device__ bool moller_trumbore(const Vector &A, const Vector &B, const Vector &C, Vector& N, const Ray &r, float &t) {
		Vector e1 = B - A;
		Vector e2 = C - A;
		N = cross(e1, e2);
		if (dot(r.u, N) == 0) return 0;
		float beta = dot(e2, cross(A - r.O, r.u)) / dot(r.u, N);
		float gamma = - dot(e1, cross(A - r.O, r.u)) / dot(r.u, N);
		if (!between(0, beta, 1) || !between(0, gamma, 1))	return 0;
		t = dot(A - r.O, N) / dot(r.u, N);
		return beta + gamma <= 1 && t > 0;
	}
	
	__device__ bool intersect(const Ray &r, float &t, Vector &N) override {
	 float t_tmp;

		// #define BUILD_BVH(var, idx) var.left = bvh[(idx) * 10 + 0],\
		// 							var.right = bvh[(idx) * 10 + 1],\
		// 							var.bb = BoundingBox(\
		// 								Vector(\
		// 									bvh[(idx) * 10 + 2],\
		// 									bvh[(idx) * 10 + 3],\
		// 									bvh[(idx) * 10 + 4]\
		// 								),\
		// 								Vector(\
		// 									bvh[(idx) * 10 + 5],\
		// 									bvh[(idx) * 10 + 6],\
		// 									bvh[(idx) * 10 + 7]\
		// 								)\
		// 							),\
		// 							var.triangle_start = bvh[(idx) * 10 + 8],\
		// 							var.triangle_end = bvh[(idx) * 10 + 9]
		// PRINT_VEC(tmp);
		BVH root_bvh = bvh;
		// BUILD_BVH(root_bvh, 0);
		if (!root_bvh.bb.intersect(r, t_tmp)) {
			return 0;
		}

		BVH* s[30];
		int s_size = 0;
		s[s_size++] = &root_bvh;


	 float t_min = INF;
	 int idx_min = -1;
		while (s_size) {
			BVH *cur = s[s_size-1];
			s_size--;
			// BVHDevice cur_bvh;
			// BUILD_BVH(cur_bvh, cur);
			if (cur->left != NULL) {
				// BVHDevice left_bvh;
				// BUILD_BVH(left_bvh, cur.left);
				// BVHDevice right_bvh;
				// BUILD_BVH(right_bvh, cur.right);
			 float t_left, t_right;
				bool ok_left = cur->left->bb.intersect(r, t_left);
				bool ok_right = cur->right->bb.intersect(r, t_right);
				if (ok_left) s[s_size++] = cur->left;
				if (ok_right) s[s_size++] = cur->right;
			} else {
				// Leaf
				for (int i = cur->triangle_start; i < cur->triangle_end; i++) {
				 float t_cur;
					Vector A = vertices[indices[i].vtxi], B = vertices[indices[i].vtxj], C = vertices[indices[i].vtxk];
					Vector N_triangle;
					bool inter = moller_trumbore(A, B, C, N_triangle, r, t_cur);
					if (!inter) continue;
					if (t_cur > 1e-4f && t_cur < t_min) {
						t_min = t_cur;
						N = N_triangle;
						idx_min = i;
						//
						// PRINT_VEC(N);
					}
				} 
			}
		}
		N.normalize();
		if(idx_min > -1)
			// PRINT_VEC(N);
			get_smooth_normal(r, indices[idx_min], N);
			// printf("new N ");
			// PRINT_VEC(N);
		t = t_min;
		if(t_min != INF){
			// printf("inter triangle %f\n", t_min);
		}else{
			printf("no hit\n");
		}
		return t_min != INF;
	}

	__device__	BoundingBox compute_bbox(int triangle_start, int triangle_end) {
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
	 	float split = (cur->bb.mn[max_axis] + cur->bb.mx[max_axis]) / 2;
		for (int i = triangle_start; i < triangle_end; i++) {
		 float cen = (vertices[indices[i].vtxi][max_axis] + vertices[indices[i].vtxj][max_axis] + vertices[indices[i].vtxk][max_axis]) / 3;
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

	__device__ void bvhTreeToArray(BVH *cur, float* bvh_arr, size_t &arr_size, size_t arr_idx = 0) {
		// std::cout << arr_idx << ' ' << cur->triangle_start << ' ' << cur->triangle_end << '\n';
		// std::cout << "rfgsg\n";
		// PRINT_VEC(cur->bb.mn);
		// PRINT_VEC(cur->bb.mx);
		
		bvh_arr[arr_idx * 10 + 2] = cur->bb.mn[0];
		bvh_arr[arr_idx * 10 + 3] = cur->bb.mn[1];
		bvh_arr[arr_idx * 10 + 4] = cur->bb.mn[2];
		bvh_arr[arr_idx * 10 + 5] = cur->bb.mx[0];
		bvh_arr[arr_idx * 10 + 6] = cur->bb.mx[1];
		bvh_arr[arr_idx * 10 + 7] = cur->bb.mx[2];
		bvh_arr[arr_idx * 10 + 8] = cur->triangle_start;
		bvh_arr[arr_idx * 10 + 9] = cur->triangle_end;

		if (cur->left) {
			bvh_arr[arr_idx * 10 + 0] = arr_size++;
			bvhTreeToArray(cur->left, bvh_arr, arr_size, bvh_arr[arr_idx * 10 + 0]);
		} else {
			bvh_arr[arr_idx * 10 + 0] = -1;
		}
		if (cur->right) {
			bvh_arr[arr_idx * 10 + 1] = arr_size++;
			bvhTreeToArray(cur->right, bvh_arr, arr_size, bvh_arr[arr_idx * 10 + 1]);
		} else {
			bvh_arr[arr_idx * 10 + 1] = -1;
		}
	}

	TriangleIndices* indices;
	int indices_size;
	Vector* vertices, *normals;
	int vertices_size, normals_size;
	// float* bvh;
	BVH bvh;
};

__device__ Vector rotate(const Vector &v, const float *R) {
    return Vector(
        R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
        R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
        R[6] * v[0] + R[7] * v[1] + R[8] * v[2]
    );
}

__global__ void transform(Vector *vertices, int vertices_size, Vector *normals, int normals_size, Vector translation, const float *rotation_matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vertices_size) {
        // Transform the vertex
        vertices[idx] = rotate(vertices[idx], rotation_matrix);
        vertices[idx][0] += translation[0];
        vertices[idx][1] += translation[1];
        vertices[idx][2] += translation[2];
    }

    if (idx < normals_size) {
        // Transform the normal
        normals[idx] = rotate(normals[idx], rotation_matrix);
		normals[idx][0] += translation[0];
        normals[idx][1] += translation[1];
        normals[idx][2] += translation[2];
    }
}

class TriangleMeshHost {
public:
 	~TriangleMeshHost() {}
	TriangleMeshHost() {};
	void rescale(float scale, Vector offset){
		for(int i = 0; i < vertices.size(); i++){
			vertices[i] = vertices[i] * scale + offset;
		}
	}

	
void readOBJ(const char *obj)
{

    char matfile[255];
    char grp[255];

    FILE *f;
    f = fopen(obj, "r");
    int curGroup = -1;
    while (!feof(f))
    {
        char line[255];
        if (!fgets(line, 255, f))
            break;

        std::string linetrim(line);
        linetrim.erase(linetrim.find_last_not_of(" \r\t") + 1);
        strcpy(line, linetrim.c_str());

        if (line[0] == 'u' && line[1] == 's')
        {
            sscanf(line, "usemtl %[^\n]\n", grp);
            curGroup++;
        }

        if (line[0] == 'v' && line[1] == ' ')
        {
            Vector vec;

            Vector col;
            if (sscanf(line, "v %f %f %f %f %f %f\n", &vec[0], &vec[1], &vec[2], &col[0], &col[1], &col[2]) == 6)
            {
                col[0] = std::min(1.f, std::max(0.f, col[0]));
                col[1] = std::min(1.f, std::max(0.f, col[1]));
                col[2] = std::min(1.f, std::max(0.f, col[2]));

                vertices.push_back(vec);
                vertexcolors.push_back(col);
            }
            else
            {
                sscanf(line, "v %f %f %f\n", &vec[0], &vec[1], &vec[2]);
                vertices.push_back(vec);
            }
        }
        if (line[0] == 'v' && line[1] == 'n')
        {
            Vector vec;
            sscanf(line, "vn %f %f %f\n", &vec[0], &vec[1], &vec[2]);
            normals.push_back(vec);
        }
        if (line[0] == 'v' && line[1] == 't')
        {
            Vector vec;
            sscanf(line, "vt %f %f\n", &vec[0], &vec[1]);
            uvs.push_back(vec);
        }
        if (line[0] == 'f')
        {
            TriangleIndices t;
            int i0, i1, i2, i3;
            int j0, j1, j2, j3;
            int k0, k1, k2, k3;
            int nn;
            t.group = curGroup;

            char *consumedline = line + 1;
            int offset;

            nn = sscanf(consumedline, "%u/%u/%u %u/%u/%u %u/%u/%u%n", &i0, &j0, &k0, &i1, &j1, &k1, &i2, &j2, &k2, &offset);
            if (nn == 9)
            {
                if (i0 < 0)
                    t.vtxi = vertices.size() + i0;
                else
                    t.vtxi = i0 - 1;
                if (i1 < 0)
                    t.vtxj = vertices.size() + i1;
                else
                    t.vtxj = i1 - 1;
                if (i2 < 0)
                    t.vtxk = vertices.size() + i2;
                else
                    t.vtxk = i2 - 1;
                if (j0 < 0)
                    t.uvi = uvs.size() + j0;
                else
                    t.uvi = j0 - 1;
                if (j1 < 0)
                    t.uvj = uvs.size() + j1;
                else
                    t.uvj = j1 - 1;
                if (j2 < 0)
                    t.uvk = uvs.size() + j2;
                else
                    t.uvk = j2 - 1;
                if (k0 < 0)
                    t.ni = normals.size() + k0;
                else
                    t.ni = k0 - 1;
                if (k1 < 0)
                    t.nj = normals.size() + k1;
                else
                    t.nj = k1 - 1;
                if (k2 < 0)
                    t.nk = normals.size() + k2;
                else
                    t.nk = k2 - 1;
                indices.push_back(t);
            }
            else
            {
                nn = sscanf(consumedline, "%u/%u %u/%u %u/%u%n", &i0, &j0, &i1, &j1, &i2, &j2, &offset);
                if (nn == 6)
                {
                    if (i0 < 0)
                        t.vtxi = vertices.size() + i0;
                    else
                        t.vtxi = i0 - 1;
                    if (i1 < 0)
                        t.vtxj = vertices.size() + i1;
                    else
                        t.vtxj = i1 - 1;
                    if (i2 < 0)
                        t.vtxk = vertices.size() + i2;
                    else
                        t.vtxk = i2 - 1;
                    if (j0 < 0)
                        t.uvi = uvs.size() + j0;
                    else
                        t.uvi = j0 - 1;
                    if (j1 < 0)
                        t.uvj = uvs.size() + j1;
                    else
                        t.uvj = j1 - 1;
                    if (j2 < 0)
                        t.uvk = uvs.size() + j2;
                    else
                        t.uvk = j2 - 1;
                    indices.push_back(t);
                }
                else
                {
                    nn = sscanf(consumedline, "%u %u %u%n", &i0, &i1, &i2, &offset);
                    if (nn == 3)
                    {
                        if (i0 < 0)
                            t.vtxi = vertices.size() + i0;
                        else
                            t.vtxi = i0 - 1;
                        if (i1 < 0)
                            t.vtxj = vertices.size() + i1;
                        else
                            t.vtxj = i1 - 1;
                        if (i2 < 0)
                            t.vtxk = vertices.size() + i2;
                        else
                            t.vtxk = i2 - 1;
                        indices.push_back(t);
                    }
                    else
                    {
                        nn = sscanf(consumedline, "%u//%u %u//%u %u//%u%n", &i0, &k0, &i1, &k1, &i2, &k2, &offset);
                        if (i0 < 0)
                            t.vtxi = vertices.size() + i0;
                        else
                            t.vtxi = i0 - 1;
                        if (i1 < 0)
                            t.vtxj = vertices.size() + i1;
                        else
                            t.vtxj = i1 - 1;
                        if (i2 < 0)
                            t.vtxk = vertices.size() + i2;
                        else
                            t.vtxk = i2 - 1;
                        if (k0 < 0)
                            t.ni = normals.size() + k0;
                        else
                            t.ni = k0 - 1;
                        if (k1 < 0)
                            t.nj = normals.size() + k1;
                        else
                            t.nj = k1 - 1;
                        if (k2 < 0)
                            t.nk = normals.size() + k2;
                        else
                            t.nk = k2 - 1;
                        indices.push_back(t);
                    }
                }
            }

            consumedline = consumedline + offset;

            while (true)
            {
                if (consumedline[0] == '\n')
                    break;
                if (consumedline[0] == '\0')
                    break;
                nn = sscanf(consumedline, "%u/%u/%u%n", &i3, &j3, &k3, &offset);
                TriangleIndices t2;
                t2.group = curGroup;
                if (nn == 3)
                {
                    if (i0 < 0)
                        t2.vtxi = vertices.size() + i0;
                    else
                        t2.vtxi = i0 - 1;
                    if (i2 < 0)
                        t2.vtxj = vertices.size() + i2;
                    else
                        t2.vtxj = i2 - 1;
                    if (i3 < 0)
                        t2.vtxk = vertices.size() + i3;
                    else
                        t2.vtxk = i3 - 1;
                    if (j0 < 0)
                        t2.uvi = uvs.size() + j0;
                    else
                        t2.uvi = j0 - 1;
                    if (j2 < 0)
                        t2.uvj = uvs.size() + j2;
                    else
                        t2.uvj = j2 - 1;
                    if (j3 < 0)
                        t2.uvk = uvs.size() + j3;
                    else
                        t2.uvk = j3 - 1;
                    if (k0 < 0)
                        t2.ni = normals.size() + k0;
                    else
                        t2.ni = k0 - 1;
                    if (k2 < 0)
                        t2.nj = normals.size() + k2;
                    else
                        t2.nj = k2 - 1;
                    if (k3 < 0)
                        t2.nk = normals.size() + k3;
                    else
                        t2.nk = k3 - 1;
                    indices.push_back(t2);
                    consumedline = consumedline + offset;
                    i2 = i3;
                    j2 = j3;
                    k2 = k3;
                }
                else
                {
                    nn = sscanf(consumedline, "%u/%u%n", &i3, &j3, &offset);
                    if (nn == 2)
                    {
                        if (i0 < 0)
                            t2.vtxi = vertices.size() + i0;
                        else
                            t2.vtxi = i0 - 1;
                        if (i2 < 0)
                            t2.vtxj = vertices.size() + i2;
                        else
                            t2.vtxj = i2 - 1;
                        if (i3 < 0)
                            t2.vtxk = vertices.size() + i3;
                        else
                            t2.vtxk = i3 - 1;
                        if (j0 < 0)
                            t2.uvi = uvs.size() + j0;
                        else
                            t2.uvi = j0 - 1;
                        if (j2 < 0)
                            t2.uvj = uvs.size() + j2;
                        else
                            t2.uvj = j2 - 1;
                        if (j3 < 0)
                            t2.uvk = uvs.size() + j3;
                        else
                            t2.uvk = j3 - 1;
                        consumedline = consumedline + offset;
                        i2 = i3;
                        j2 = j3;
                        indices.push_back(t2);
                    }
                    else
                    {
                        nn = sscanf(consumedline, "%u//%u%n", &i3, &k3, &offset);
                        if (nn == 2)
                        {
                            if (i0 < 0)
                                t2.vtxi = vertices.size() + i0;
                            else
                                t2.vtxi = i0 - 1;
                            if (i2 < 0)
                                t2.vtxj = vertices.size() + i2;
                            else
                                t2.vtxj = i2 - 1;
                            if (i3 < 0)
                                t2.vtxk = vertices.size() + i3;
                            else
                                t2.vtxk = i3 - 1;
                            if (k0 < 0)
                                t2.ni = normals.size() + k0;
                            else
                                t2.ni = k0 - 1;
                            if (k2 < 0)
                                t2.nj = normals.size() + k2;
                            else
                                t2.nj = k2 - 1;
                            if (k3 < 0)
                                t2.nk = normals.size() + k3;
                            else
                                t2.nk = k3 - 1;
                            consumedline = consumedline + offset;
                            i2 = i3;
                            k2 = k3;
                            indices.push_back(t2);
                        }
                        else
                        {
                            nn = sscanf(consumedline, "%u%n", &i3, &offset);
                            if (nn == 1)
                            {
                                if (i0 < 0)
                                    t2.vtxi = vertices.size() + i0;
                                else
                                    t2.vtxi = i0 - 1;
                                if (i2 < 0)
                                    t2.vtxj = vertices.size() + i2;
                                else
                                    t2.vtxj = i2 - 1;
                                if (i3 < 0)
                                    t2.vtxk = vertices.size() + i3;
                                else
                                    t2.vtxk = i3 - 1;
                                consumedline = consumedline + offset;
                                i2 = i3;
                                indices.push_back(t2);
                            }
                            else
                            {
                                consumedline = consumedline + 1;
                            }
                        }
                    }
                }
            }
        }
    }
    fclose(f);
};

	std::vector<TriangleIndices> indices;
	std::vector<Vector> vertices;
	std::vector<Vector> normals;
	std::vector<Vector> uvs;
	std::vector<Vector> vertexcolors;
	BVH bvh;
	size_t n_bvhs = 0;
	#define between(A, B, C) ((A) <= (B) && (B) <= (C))


};

class Scene {
public:
	__device__ void addObject(Geometry* s) {
		s->id = objects_size;
		objects[objects_size++] = s;
	}

	__device__ bool intersect_all(const Ray& r, Vector &P, Vector &N, int &objectId) {
	 float t_min = INF;
		int id_min = -1;
		Vector N_min;
        for (int i = 0; i < objects_size; i++) {
            Geometry* object_ptr = objects[i];
		 float t;
		 float id = object_ptr->id;
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

	__device__ Vector getColorIterative(curandState *rand_state, const Ray& input_ray, int max_ray_depth) {
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
				 float epsilon = 1e-4;
					Vector P_adjusted = P + epsilon * N;
					Vector new_direction = ray.u - 2 * dot(ray.u, N) * N;
					Ray reflected_ray(P_adjusted, new_direction, ray.refraction_index);
					ray = reflected_ray;
				} else if (objects[sphere_id]->in_refraction_index != objects[sphere_id]->out_refraction_index) {
					// Refraction
					types[ray_depth] = 0;
				 float epsilon = 1e-4;
				 float refract_ratio;
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
				 	float epsilon = 1e-4;
					Vector P_adjusted = P + epsilon * N;
					Vector N_prime;
					bool _ = intersect_all(Ray(P_adjusted, NORMED_VEC(L - P_adjusted)), P_prime, N_prime, sphere_id_shadow);
					
					if ((P_prime - P_adjusted).norm2() <= (L - P_adjusted).norm2()) {
						// Is shadow
						direct_colors[ray_depth] = Vector(0.f, 0.f, 0.f);
					} else {
						// Get direct color
						Geometry* S = objects[sphere_id];
						Vector wlight = L - P;
						wlight.normalize();
					 float l = intensity / (4 * PI * (L - P).norm2()) * max(dot(N, wlight), 0.f);
						direct_colors[ray_depth] = l * S->albedo / PI;
					}
					// printf("%f %f\n", (P_prime - P_adjusted).norm2(), (L - P_adjusted).norm2());
					// Get indirect color by launching ray
					unsigned int seed = threadIdx.x;
					float r1 = curand_uniform(rand_state);
					float r2 = curand_uniform(rand_state);
					float x = cos(2 * PI * r1) * sqrt(1 - r2);
					float y = sin(2 * PI * r1) * sqrt(1 - r2);
					float z = sqrt(r2);
					Vector T1;
					if (abs(N[1]) != 0.f && abs(N[0]) != 0.f) {
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

					// printf("DIFF INTER\n");

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
 	float intensity = 3e10;
	Vector L;
	curandState* rand_states;
};

__global__ void KernelInit(Scene *s, TriangleIndices *indices, int indices_size, Vector *vertices, int vertices_size,Vector *normals, int normals_size) {
 	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	if (!threadId) {
		s->L = Vector(-10., 20., 40.);
		s->objects_size = 0;
		s->intensity = 3e10;
		// s->addObject(new Sphere(Vector(0, 0, 0), 10, Vector(1., 1., 1.))); // white sphere
		s->addObject(new Sphere(Vector(0, 0, -1000), 940.0f, Vector(0.0f, 1.0f, 0.0f))); // green fore wall
		s->addObject(new Sphere(Vector(0, -1000, 0), 990.0f, Vector(0.0f, 0.0f, 1.0f))); // blue floor
		s->addObject(new Sphere(Vector(0, 1000, 0), 940.0f, Vector(1.0f, 0.0f, 0.0f))); // red ceiling
		s->addObject(new Sphere(Vector(-1000, 0, 0), 940.0f, Vector(0.0f, 1.0f, 1.0f))); // cyan left wall
		s->addObject(new Sphere(Vector(1000, 0, 0), 940.0f, Vector(1.0f, 1.0f, 0.0f))); // yellow right wall
		s->addObject(new Sphere(Vector(0, 0, 1000), 940.0f, Vector(1.0f, 0.0f, 1.0f))); // magenta back wall
		// s->addObject(new Sphere(Vector(-20, 0, 0), 10, Vector(0., 0., 0.), 1)); // mirror sphere
		// s->addObject(new Sphere(Vector(20, 0, 0), 9, Vector(0., 0., 0.), 0, 1, 1.5)); // inner nested ssphere
		// s->addObject(new Sphere(Vector(20, 0, 0), 10, Vector(0., 0., 0.), 0, 1.5, 1)); // outer nested sphere

		TriangleMesh* cat = new TriangleMesh();
		cat->albedo = Vector(0.25f, 0.25f, 0.25f);
	 	cat->indices_size = indices_size;
		cat->indices = indices;
		cat->vertices_size = vertices_size;
		cat->vertices = vertices;
		// printf("%d %d\n", indices_size, vertices_size);
		cat->normals_size = normals_size;
		cat->normals = normals;
		// for(int i = 0; i < 10; i++){
		// 	PRINT_VEC(cat->normals[i]);
		// }
		// cat->uvs_size;
		// cat->uvs;
		// cat->vertexcolors_size;
		// cat->vertexcolors;
		cat->bvh.bb = cat->compute_bbox(0, cat->indices_size);
		cat->buildBVH(&(cat->bvh), 0, cat->indices_size);
		s->addObject(cat);
	}
}

__global__ void KernelLaunch(Scene *s, Vector *colors, int W, int H, int num_rays, int num_bounce) {
    // size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curandState rand_state; // state of the random number generator, to prevent repetition
	curand_init(threadId, 0, 0, &rand_state);

	Vector outcolor;
	int i = y*W + x; // pixel index in buffer
	// float coordx = (float) x / W; // pixel x-coordinate on screen
	// int coordy = (float) y / H;

	outcolor = Vector(0.f, 0.f, 0.f);
	
	// curand_init(123456, index, 0, shared_scene->rand_states + threadIdx.x);
    // int i = (index / num_rays) / W, j = (index / num_rays) % W;
	
	// printf("%d %d\n", x, y);
	Vector C(0, 0, 55);
	float alpha = PI/3;
	float z = -W / (2 * tan(alpha/2));

    Vector u_center(x -  (float)W / 2 + 0.5,  (float)H / 2 - y - 0.5, z);
	// Box-muller for anti-aliasing

	float sigma = 0.2;
	for(int i = 0; i < num_rays; i++){
		float r1 = curand_uniform(&rand_state);
		float r2 = curand_uniform(&rand_state);
		Vector u = u_center + Vector(sigma * sqrt(-2 * log(r1)) * cos(2 * PI * r2), sigma * sqrt(-2 * log(r1)) * sin(2 * PI * r2), 0);
		u.normalize();
		Ray r(C, u);
		Vector color = s->getColorIterative(&rand_state, r, num_bounce);
		outcolor = outcolor + color;
	}
	// PRINT_VEC(color);
	outcolor = outcolor / num_rays;
	colors[i] = outcolor;	
}

void allocateAndCopyDataToDevice(TriangleMeshHost* mesh_ptr, Vector*& d_vertices, Vector*& d_normals, TriangleIndices*& d_indices) {
    gpuErrchk(cudaMalloc((void**)&d_vertices, mesh_ptr->vertices.size() * sizeof(Vector)));
    gpuErrchk(cudaMemcpy(d_vertices, &(mesh_ptr->vertices[0]), mesh_ptr->vertices.size() * sizeof(Vector), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**)&d_normals, mesh_ptr->normals.size() * sizeof(Vector)));
    gpuErrchk(cudaMemcpy(d_normals, &(mesh_ptr->normals[0]), mesh_ptr->normals.size() * sizeof(Vector), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**)&d_indices, mesh_ptr->indices.size() * sizeof(TriangleIndices)));
    gpuErrchk(cudaMemcpy(d_indices, &(mesh_ptr->indices[0]), mesh_ptr->indices.size() * sizeof(TriangleIndices), cudaMemcpyHostToDevice));
}

void transformMesh(Vector* d_vertices, int vertices_size, Vector* d_normals, int normals_size, const Vector& translation, const float* rotation_matrix) {
    float* d_rotation_matrix;
    gpuErrchk(cudaMalloc(&d_rotation_matrix, 9 * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_rotation_matrix, rotation_matrix, 9 * sizeof(float), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int max_size = max(vertices_size, normals_size);
    const int numBlocks = (max_size + threadsPerBlock - 1) / threadsPerBlock;

    transform<<<numBlocks, threadsPerBlock>>>(d_vertices, vertices_size, d_normals, normals_size, translation, d_rotation_matrix);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFree(d_rotation_matrix);
}

void renderScene(Scene* d_s, Vector* d_colors, int W, int H, int num_rays, int num_bounce) {
    dim3 block(16, 16, 1);
    dim3 grid(W / block.x, H / block.y, 1);

    KernelLaunch<<<grid, block>>>(d_s, d_colors, W, H, num_rays, num_bounce);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void saveImage(Vector* h_colors, int W, int H) {
    char* image = new char[W * H * 3];
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            image[(i * W + j) * 3 + 0] = min(pow(h_colors[(i * W + j)][0], 1.0 / 2.2), 255.0);
            image[(i * W + j) * 3 + 1] = min(pow(h_colors[(i * W + j)][1], 1.0 / 2.2), 255.0);
            image[(i * W + j) * 3 + 2] = min(pow(h_colors[(i * W + j)][2], 1.0 / 2.2), 255.0);
        }
    }
    stbi_write_png("image.png", W, H, 3, image, 0);
    delete[] image;
}

int main(int argc, char **argv) {
    if (argc != 3) {
		std::cout << "Invalid number of arguments!\nThe first argument is number of rays and the second argument is number of bounces.\n";
		return 0;
	}

	/*
		Measure runtime
	*/
	auto start_time = std::chrono::system_clock::now();

	const int num_rays = atoi(argv[1]), num_bounce = atoi(argv[2]);
	int W = 512;
	int H = 512;
	int colors_size = sizeof(float) * H * W * 3 * num_rays;
	const int BLOCK_DIM = 128;
	int GRID_DIM = W * H * num_rays / BLOCK_DIM;
	float angle = -M_PI/3;

    Vector translation = {0.f, 0.f, 0.f};
    // float rotation_matrix[9] = {
    //     cos(angle), -sin(angle), 0.,
    //     sin(angle), cos(angle), 0.,
    //     0., 0., 1.
    // };
	float rotation_matrix[9] = {
        cos(angle), 0, sin(angle),
       	0, 1, 0,
        -sin(angle), 0., cos(angle),
    };
	
	Scene *d_s;
	Vector *h_colors, *d_colors;
    char *image;
	h_colors = new Vector[H * W];
    image = new char[H * W * 3];

	gpuErrchk( cudaDeviceSetLimit(cudaLimitStackSize, 1<<14) );
	
	// Malloc & transfer to GPU
    gpuErrchk( cudaMalloc((void**)&d_s, sizeof(Scene)) );
    gpuErrchk( cudaMalloc((void**)&d_colors, H * W * sizeof(Vector)) );

	/*
		Instantiate cat object
	*/
	TriangleMeshHost* mesh_ptr = new TriangleMeshHost(); // cat
	const char *path = "cadnav.com_model/Models_F0202A090/cat.obj";
	mesh_ptr->readOBJ(path);
	mesh_ptr->rescale(0.6f, Vector(0.f, -10.f, 0.f));

	/*
		Transfer remaining neccessary mesh information to GPU
	*/
	Vector* d_vertices;
    Vector* d_normals;
    TriangleIndices* d_indices;

    allocateAndCopyDataToDevice(mesh_ptr, d_vertices, d_normals, d_indices);

	float *d_rotation_matrix;
	cudaMalloc(&d_rotation_matrix, 9 * sizeof(float));
	cudaMemcpy(d_rotation_matrix, rotation_matrix, 9 * sizeof(float), cudaMemcpyHostToDevice);

	transformMesh(d_vertices, mesh_ptr->vertices.size(), d_normals, mesh_ptr->normals.size(), translation, rotation_matrix);

	KernelInit<<<1, 1>>>(d_s, d_indices, mesh_ptr->indices.size(), d_vertices, mesh_ptr->vertices.size(), d_normals, mesh_ptr->normals.size());

	dim3 block(16, 16, 1);
	dim3 grid(W / block.x, H / block.y, 1);

	renderScene(d_s, d_colors, W, H, num_rays, num_bounce);

	/*
		Transfer result back from GPU
		Clean memory
		Deduce final result
	*/
    gpuErrchk( cudaMemcpy(h_colors, d_colors, H * W * sizeof(Vector), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(d_s) );
    gpuErrchk( cudaFree(d_colors) );
    gpuErrchk( cudaFree(d_indices) );
    gpuErrchk( cudaFree(d_vertices) );
	// delete[] arr_bvh;
	saveImage(h_colors, W, H);
    delete[] h_colors;

	/*
		Measure runtime
	*/
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> run_time = end_time - start_time;
    std::cout << "Rendering time: " << run_time.count() << " s\n";

	return 0;
}
