#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <math.h>
#define _USE_MATH_DEFINES
#include <omp.h>

// Thanks to https://stackoverflow.com/questions/21237905/how-do-i-generate-thread-safe-uniform-random-numbers
#if defined (_MSC_VER)  // Visual studio
    #define thread_local __declspec( thread )
#elif defined (__GCC__) // GCC
    #define thread_local __thread
#endif

#include <random>
#include <time.h>
#include <thread>
#include <chrono>
#include <ctime>
#include <stack>

#define SQR(X) ((X)*(X))
#define NORMED_VEC(X) ((X) / (X).norm())
#ifndef PI
    #define PI 3.14159265358979323846
#endif
#define INF (1e9+9)

#include <string>
#include <stdio.h>
#include <algorithm>
#include <vector>

// #define NAIVE
// #define BB
#define ENABLE_BVH

class Vector {
public:
	explicit Vector(float x = 0, float y = 0, float z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	float norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	float norm() const {
		return sqrt(norm2());
	}
	void normalize() {
		float n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
	}
	float operator[](int i) const { return data[i]; };
	float& operator[](int i) { return data[i]; };
	float data[3];
};

Vector operator+(const Vector& a, const Vector& b) {
	return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
	return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator-(const Vector& a) {
	return Vector(-a[0], -a[1], -a[2]);
}
Vector operator*(const float a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
Vector operator*(const Vector& a, const float b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
// Element wise vector multiplication
Vector operator*(const Vector& a, const Vector& b) {
	return Vector(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
Vector operator/(const Vector& a, const float b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
float dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

class Ray {
public:
	Ray(const Vector &O, const Vector &u, float refraction_index = 1.) : O(O), u(u), refraction_index(refraction_index) {};
	// ...
	Vector O, u;
	float refraction_index;
};

class Geometry {
public:
	Geometry(const Vector &albedo, int id, bool mirror, float in_refraction_index, float out_refraction_index): albedo(albedo), id(id),
	mirror(mirror), in_refraction_index(in_refraction_index), out_refraction_index(out_refraction_index) {}
	Geometry(): mirror(0), in_refraction_index(1), out_refraction_index(1) {};

	Vector albedo;
	int id;
	bool mirror;
	float in_refraction_index;
	float out_refraction_index;
	virtual bool intersect(const Ray& r, float &t, Vector &N) { return 0; };
};

/* Start of code derived from Prof Bonnel's code */
class TriangleIndices {
public:
	TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1, bool added = false) : vtxi(vtxi), vtxj(vtxj), vtxk(vtxk), uvi(uvi), uvj(uvj), uvk(uvk), ni(ni), nj(nj), nk(nk), group(group) {
	};
	int vtxi, vtxj, vtxk; // indices within the vertex coordinates array
	int uvi, uvj, uvk;  // indices within the uv coordinates array
	int ni, nj, nk;  // indices within the normals array
	int group;       // face group
};

class BoundingBox {
public:
	Vector mn, mx;

	BoundingBox(): mn(Vector(INF, INF, INF)), mx(Vector(-INF, -INF, -INF)) {};

	inline void update(const Vector &vec) {
		mn[0] = std::min(mn[0], vec[0]);
		mn[1] = std::min(mn[1], vec[1]);
		mn[2] = std::min(mn[2], vec[2]);
		mx[0] = std::max(mx[0], vec[0]);
		mx[1] = std::max(mx[1], vec[1]);
		mx[2] = std::max(mx[2], vec[2]);
	}

	inline bool intersect(const Ray &r, float &t) {
		float t0x = (mn[0] - r.O[0]) / r.u[0];
		float t0y = (mn[1] - r.O[1]) / r.u[1];
		float t0z = (mn[2] - r.O[2]) / r.u[2];
		float t1x = (mx[0] - r.O[0]) / r.u[0];
		float t1y = (mx[1] - r.O[1]) / r.u[1];
		float t1z = (mx[2] - r.O[2]) / r.u[2];
		if (t0x > t1x) std::swap(t0x, t1x);
		if (t0y > t1y) std::swap(t0y, t1y);
		if (t0z > t1z) std::swap(t0z, t1z);
		return std::min({t1x, t1y, t1z}) > std::max({t0x, t0y, t0z});
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
  ~TriangleMesh() {}
	TriangleMesh() {};

	#define between(A, B, C) ((A) <= (B) && (B) <= (C))

	// inline bool ray_plane_intersect(const Vector &N, const Vector &A, const Ray &r, float &t) {
	// 	if (dot(r.u, N) == 0) return 0;
	// 	t = dot(A - r.O, N) / dot(r.u, N);
	// 	return 1;
	// }

	BoundingBox compute_bbox(int triangle_start, int triangle_end) {
		BoundingBox bb;
		for (int i = triangle_start; i < triangle_end; i++) {
			bb.update(vertices[indices[i].vtxi]);
			bb.update(vertices[indices[i].vtxj]);
			bb.update(vertices[indices[i].vtxk]);
		}
		return bb;
	}

	void buildBVH(BVH* cur, int triangle_start, int triangle_end) {
		// std::cout << cur << ' ' << triangle_start << ' ' << triangle_end << '\n';
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
				std::swap(indices[i], indices[pivot]);
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

	bool moller_trumbore(const Vector &A, const Vector &B, const Vector &C, Vector& N, const Ray &r, float &t) {
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
	
	bool intersect(const Ray &r, float &t, Vector &N) override {
		#ifdef NAIVE
		float t_min = INF;
		for (auto index: indices) {
			float t_cur;
			Vector A = vertices[index.vtxi], B = vertices[index.vtxj], C = vertices[index.vtxk];
			Vector N_triangle;
			bool inter = moller_trumbore(A, B, C, N_triangle, r, t_cur);
			if (!inter) continue;
			if (t_cur > 0 && t_cur < t_min) {
				t_min = t_cur;
				N = N_triangle;
			}
		} 
		N.normalize();
		t = t_min;
		return t_min != INF;
		#endif

		#ifdef BB
		float t_tmp;
		if (!bvh.bb.intersect(r, t_tmp)) return 0;
		float t_min = INF;
		for (auto index: indices) {
			float t_cur;
			Vector A = vertices[index.vtxi], B = vertices[index.vtxj], C = vertices[index.vtxk];
			Vector N_triangle;
			bool inter = moller_trumbore(A, B, C, N_triangle, r, t_cur);
			if (!inter) continue;
			if (t_cur > 0 && t_cur < t_min) {
				t_min = t_cur;
				N = N_triangle;
			}
		} 
		N.normalize();
		t = t_min;
		return t_min != INF;
		#endif

		#ifdef ENABLE_BVH
		float t_tmp;
		if (!bvh.bb.intersect(r, t_tmp)) return 0;
		std::stack<BVH*> s;
		s.push(&bvh);

		float t_min = INF;
		while(!s.empty()) {
			const BVH* cur = s.top();
			s.pop();
			if (cur->left) {
				float t_left, t_right;
				bool ok_left = cur->left->bb.intersect(r, t_left);
				bool ok_right = cur->right->bb.intersect(r, t_right);
				if (ok_left && t_left < t_min) s.push(cur->left);
				if (ok_right && t_right <  t_min) s.push(cur->right);
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
					}
				} 
			}
		}
		N.normalize();
		t = t_min;
		return t_min != INF;
		#endif
		return 0;
	}

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
				if (sscanf(line, "v %f %f %f %f %f %f\n", &vec[0], &vec[1], &vec[2], &col[0], &col[1], &col[2]) == 6) {
					col[0] = std::min(1.f, std::max(0.f, col[0]));
					col[1] = std::min(1.f, std::max(0.f, col[1]));
					col[2] = std::min(1.f, std::max(0.f, col[2]));

					vertices.push_back(vec);
					vertexcolors.push_back(col);

				} else {
					sscanf(line, "v %f %f %f\n", &vec[0], &vec[1], &vec[2]);
					vec = vec*0.8+Vector(0, -10, 0);
					vertices.push_back(vec);
				}
				bb.update(vec);
			}
			if (line[0] == 'v' && line[1] == 'n') {
				Vector vec;
				sscanf(line, "vn %f %f %f\n", &vec[0], &vec[1], &vec[2]);
				normals.push_back(vec);
			}
			if (line[0] == 'v' && line[1] == 't') {
				Vector vec;
				sscanf(line, "vt %f %f\n", &vec[0], &vec[1]);
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
	BoundingBox bb;
	BVH bvh;
};
/* End of code derived from Prof Bonnel's code */

class Sphere: public Geometry {
public:
	Sphere(const Vector &C, float R, const Vector& albedo, bool mirror = 0, float in_refraction_index = 1., float out_refraction_index = 1.) : 
	C(C), R(R), Geometry(albedo, id, mirror, in_refraction_index, out_refraction_index) {};
	// ...
    Vector C;
    float R;
	bool intersect(const Ray &r, float &t, Vector &N) override {
		float delta = SQR(dot(r.u, r.O - C)) - ((r.O - C).norm2() - R*R);
		if (delta < 0)
			return 0;
		float t1 = dot(r.u, C - r.O) - sqrt(delta); // first intersection
		float t2 = dot(r.u, C - r.O) + sqrt(delta); // second intersection
		if (t2 < 0)
			return 0;
		t = t1 < 0 ? t2 : t1;
		// if (r.u.norm() != 1) {
		// 	std::cout << r.u.norm() << '\n';
		// }
		N = r.O + t * r.u - C;
		N.normalize();
		return 1;
	}
};

// Thanks to https://stackoverflow.com/questions/21237905/how-do-i-generate-thread-safe-uniform-random-numbers
float uniform(const int &seed) {
    static thread_local std::mt19937* generator = nullptr;
    if (!generator) generator = new std::mt19937(clock() + seed);
	    static std::uniform_real_distribution<float> distribution(0, 1);
    return distribution(*generator);
}

class Scene {
public:
	void addObject(Geometry* s) {
		s->id = objects.size();
		objects.push_back(s);
	}
	
	bool intersect_all(const Ray& r, Vector &P, Vector &N, int &objectId) {
		float t_min = INF;
		int id_min = -1;
		Vector N_min;
		for (auto object_ptr: objects) {
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

	Vector getColor(const Ray& ray, int ray_depth) {
		if (ray_depth < 0) return Vector(0., 0., 0.); // terminates recursion at some <- point
		Vector P, N;
		int sphere_id = -1;
		bool inter = intersect_all(ray, P, N, sphere_id);
		Vector color;
		if (inter) {
			if (objects[sphere_id]->mirror) {
				// Reflection
				float epsilon = 1e-3;
				Vector P_adjusted = P + epsilon * N;
				Vector new_direction = ray.u - 2 * dot(ray.u, N) * N;
				Ray reflected_ray(P_adjusted, new_direction, ray.refraction_index);
				return getColor(reflected_ray, ray_depth - 1);
			} else if (objects[sphere_id]->in_refraction_index != objects[sphere_id]->out_refraction_index) {
				// Refraction
				float epsilon = 1e-3;
				float refract_ratio;
				bool out2in = ray.refraction_index == objects[sphere_id]->out_refraction_index;
				if (out2in) { // outside to inside
					refract_ratio = objects[sphere_id]->out_refraction_index / objects[sphere_id]->in_refraction_index;
				} else { // inside to outside
					refract_ratio = objects[sphere_id]->in_refraction_index / objects[sphere_id]->out_refraction_index;
					N = -N;
				}
				if (((out2in && ray.refraction_index > objects[sphere_id]->in_refraction_index) ||
					(!out2in && ray.refraction_index > objects[sphere_id]->out_refraction_index)) &&
					SQR(refract_ratio) * (1 - SQR(dot(ray.u, N))) > 1) { // total internal reflection
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
				// handle diffuse surfaces
				// Get shadow
				Vector P_prime;
				int sphere_id_shadow;
				float epsilon = 1e-3;
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
					float l  = intensity / (4 * PI * (L - P).norm2()) * std::max(dot(N, wlight), 0.f);
					direct_color = l * S->albedo / PI;
				}
				// Get indirect color by launching rays
				unsigned int seed = omp_get_thread_num();
				float r1 = uniform(seed);
				float r2 = uniform(seed);
				float x = cos(2 * PI * r1) * sqrt(1 - r2);
				float y = sin(2 * PI * r1) * sqrt(1 - r2);
				float z = sqrt(r2);
				Vector T1;
				if (abs(N[1]) != 0 && abs(N[0]) != 0) {
					T1 = Vector(-N[1], N[0], 0);
				} else {
					T1 = Vector(-N[2], 0, N[0]);
				}
				T1.normalize();
				Vector T2 = cross(N, T1);
				Vector random_direction = x * T1 + y * T2 + z * N;
				indirect_color = objects[sphere_id]->albedo * getColor(Ray(P_adjusted, random_direction), ray_depth - 1);
				// indirect_color = Vector(0, 0, 0);
				color = direct_color + indirect_color;
			}
		}
		return color;
}
	std::vector<Geometry*> objects;
	float intensity = 3e10;
	Vector L = Vector(-10., 20., 40.);
};

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cout << "Invalid number of arguments!\nThe first argument is number of rays and the second argument is number of bounces.";
		return 0;
	}
	const int num_rays = atoi(argv[1]), num_bounce = atoi(argv[2]);
	auto start_time = std::chrono::system_clock::now();
	int W = 512;
	int H = 512;
	
	std::default_random_engine generator;

	float alpha = PI/3;
	Scene s;
	// s.addObject(new Sphere(Vector(0, 0, 0), 10, Vector(1., 1., 1.))); // white sphere
	// s.addObject(new Sphere(Vector(0, 0, 0), 10, Vector(0., 0., 0.), 0, 1.5, 1)); // refract sphere
	// s.addObject(new Sphere(Vector(-20, 0, 0), 10, Vector(0., 0., 0.), 1)); // mirror sphere
	// s.addObject(new Sphere(Vector(20, 0, 0), 9, Vector(0., 0., 0.), 0, 1, 1.5)); // inner nested ssphere
	// s.addObject(new Sphere(Vector(20, 0, 0), 10, Vector(0., 0., 0.), 0, 1.5, 1)); // outer nested sphere
	s.addObject(new Sphere(Vector(0, 0, -1000), 940, Vector(0., 1., 0.))); // green fore wall
	s.addObject(new Sphere(Vector(0, -1000, 0), 990, Vector(0., 0., 1.))); // blue floor
	s.addObject(new Sphere(Vector(0, 1000, 0), 940, Vector(1., 0., 0.))); // red ceiling
	s.addObject(new Sphere(Vector(-1000, 0, 0), 940, Vector(0., 1., 1.))); // cyan left wall
	s.addObject(new Sphere(Vector(1000, 0, 0), 940, Vector(1., 1., 0.))); // yellow right wall
	s.addObject(new Sphere(Vector(0, 0, 1000), 940, Vector(1., 0., 1.))); // magenta back wall

	TriangleMesh* mesh_ptr = new TriangleMesh(); // cat
	const char *path = "cadnav.com_model/Models_F0202A090/cat.obj";
	mesh_ptr->readOBJ(path);
	mesh_ptr->albedo = Vector(0.25, 0.25, 0.25);
	mesh_ptr->buildBVH(&(mesh_ptr->bvh), 0, mesh_ptr->indices.size());
	s.addObject(mesh_ptr);
	
	std::vector<Vector> vertices;
	std::vector<Vector> normals;
	std::vector<Vector> uvs;
	std::vector<Vector> vertexcolors; 
	Vector C(0, 0, 55);

	std::vector<unsigned char> image(W * H * 3, 0);
	float z = -W / (2 * tan(alpha/2));
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			unsigned int seed = omp_get_thread_num();
			Vector u_center((float)j - (float)W / 2 + 0.5, (float)H / 2 - i - 0.5, z);
			Vector color_total(0, 0, 0);
			for (int t = 0; t < num_rays; t++) {
				// Box-muller for anti-aliasing
				// float sigma = 2 * pow(10, -1);
				float sigma = 0;
				float r1 = uniform(seed);
				float r2 = uniform(seed);
				Vector u = u_center + Vector(sigma * sqrt(-2 * log(r1)) * cos(2 * PI * r2), sigma * sqrt(-2 * log(r1)) * sin(2 * PI * r2), 0);
				u.normalize();
				Ray r(C, u);
				Vector color = s.getColor(r, num_bounce);
				color_total = color_total + color;
			}
			Vector color_avg = color_total / num_rays;
			image[(i * W + j) * 3 + 0] = std::min(std::pow(color_avg[0], 1./2.2), 255.);
			image[(i * W + j) * 3 + 1] = std::min(std::pow(color_avg[1], 1./2.2), 255.);
			image[(i * W + j) * 3 + 2] = std::min(std::pow(color_avg[2], 1./2.2), 255.);
		}
	}
	stbi_write_png("image.png", W, H, 3, &image[0], 0);

    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<float> run_time = end_time-start_time;
    std::cout << "Rendering time: " << run_time.count() << " s\n";
	return 0;
}