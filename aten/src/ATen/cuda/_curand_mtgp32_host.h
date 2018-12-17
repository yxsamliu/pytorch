#include <hiprand.h>

// Forward declarations of functions that are defined in libcurand_static.a
// This is to avoid multiple-definitions of these when statically linking
// cudarand in both Caffe2 and ATen
#if CAFFE2_STATIC_LINK_CUDA()
hiprandStatus_t hiprandMakeMTGP32Constants(
    const mtgp32_params_fast_t params[],
    mtgp32_kernel_params_t * p);
void mtgp32_init_state(
    unsigned int state[],
    const mtgp32_params_fast_t *para,
    unsigned int seed);
hiprandStatus_t CURANDAPI hiprandMakeMTGP32KernelState(
    hiprandStateMtgp32_t *s,
    mtgp32_params_fast_t params[],
    mtgp32_kernel_params_t *k,
    int n,
    unsigned long long seed);
extern mtgp32_params_fast_t mtgp32dc_params_fast_11213[];
int mtgp32_init_by_array(
    unsigned int state[],
    const mtgp32_params_fast_t *para,
    unsigned int *array, int length);
int mtgp32_init_by_str(
    unsigned int state[],
    const mtgp32_params_fast_t *para,
    unsigned char *array);
extern const int mtgpdc_params_11213_num;

#else // CAFFE2_STATIC_LINK_CUDA
#include <hiprand_mtgp32_host.h>
#include <rocrand_mtgp32_11213.h>
#endif // CAFFE2_STATIC_LINK_CUDA
