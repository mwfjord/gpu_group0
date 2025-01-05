#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

struct RenderInitFunctor {
    curandState *rand_state;

    __device__ void operator()(int i) {
        curand_init(1984, i, 0, &rand_state[i]);
    }
};

struct RenderFunctor {
    hitable **world;
    camera **cam;
    curandState *rand_state;
    int nx, ny, ns;

    __device__ vec3 operator()(int i) {
        int x = i % nx;
        int y = i / nx;
        curandState local_rand_state = rand_state[i];
        vec3 col(0,0,0);
        for(int s=0; s < ns; s++) {
            float u = float(x + curand_uniform(&local_rand_state)) / float(nx);
            float v = float(y + curand_uniform(&local_rand_state)) / float(ny);
            ray r = (*cam)->get_ray(u, v, &local_rand_state);
            col += color(r, world, &local_rand_state);
        }
        col /= float(ns);
        col[0] = sqrt(col[0]);
        col[1] = sqrt(col[1]);
        col[2] = sqrt(col[2]);
        return col;
    }
};

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main(int argc, char **argv) {
    int nx = 1200;
    int ny = 800;
    
    int verbose = 0;
    int ns = 10;
    if(argc >= 3){
        verbose = atoi(argv[2]);
    }

    if(argc >= 2){
        ns = atoi(argv[1]);
    } else {
        if(verbose){
            std::cerr << "Default ns used \n";
        }
    }

    if(verbose){
        std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel \n";
    }
    
    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate random state
    thrust::device_ptr<curandState> d_rand_state = thrust::device_malloc<curandState>(fb_size);
    thrust::device_ptr<curandState> d_rand_state2 = thrust::device_malloc<curandState>(1);

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(thrust::raw_pointer_cast(d_rand_state2));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    thrust::device_ptr<hitable *> d_world = thrust::device_malloc<hitable *>(1);
    thrust::device_ptr<hitable *> d_list = thrust::device_malloc<hitable *>(22*22+1+3);
    thrust::device_ptr<camera *> d_camera = thrust::device_malloc<camera *>(1);
    create_world<<<1,1>>>(
        thrust::raw_pointer_cast(d_list), 
        thrust::raw_pointer_cast(d_world), 
        thrust::raw_pointer_cast(d_camera), 
        nx, ny, 
        thrust::raw_pointer_cast(d_rand_state2));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    clock_t start, stop;
    start = clock();
    
    // allocate FB
    thrust::universal_vector<vec3> fb(num_pixels);
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(num_pixels);

    // Render our buffer
    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nx * ny),
        RenderInitFunctor{thrust::raw_pointer_cast(d_rand_state)}
    );
    thrust::transform(begin, end, fb.begin(), RenderFunctor{
        thrust::raw_pointer_cast(d_world),
        thrust::raw_pointer_cast(d_camera),
        thrust::raw_pointer_cast(d_rand_state),
        nx, ny, ns
    });

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(
        thrust::raw_pointer_cast(d_list),
        thrust::raw_pointer_cast(d_world),
        thrust::raw_pointer_cast(d_camera)
    );
    checkCudaErrors(cudaGetLastError());
    thrust::device_free(d_camera);
    thrust::device_free(d_world);
    thrust::device_free(d_list);
    thrust::device_free(d_rand_state);
    thrust::device_free(d_rand_state2);
    fb.clear();
    fb.shrink_to_fit();

    cudaDeviceReset();
}
