#ifndef HITABLEH
#define HITABLEH 

#include "ray.h"
#include <cublas_v2.h>

class material;

struct hit_record
{
    float t;  
    vec3 p;
    vec3 normal; 
    material *mat_ptr;
};

class hitable  {
    public:
        virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, cublasHandle_t handle, float* d_v1, float* d_v2) const = 0;
};

#endif




