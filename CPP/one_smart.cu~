#include <iostream>
#include <math.h>
#include <algorithm>
#include <cassert>

using std::cout;
using std::endl;

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

template<typename T>
class CudaArray {
public:
  CudaArray(size_t size): _size(size), _vect(nullptr), _code(cudaMallocManaged(&_vect, size*sizeof(T))) {
  	if(!isValid()) { cout << getError() << endl; }
  }
  CudaArray(size_t size, const T& init): CudaArray(size) {
  	if(isValid()) CPU_fill_with(init);
  }
  bool isValid() { return _code == cudaSuccess; }
  std::string getError() { return cudaGetErrorString(_code); }
  void CPU_fill_with(const T& init){ std::fill_n(_vect,_size,init); }
  operator T*() { assert(isValid()); return _vect; }
  T& operator [](size_t i) { assert(isValid()); return _vect[i]; }
  ~CudaArray() { if(isValid()) cudaFree(_vect); }
private:
  size_t _size;
  T* _vect;
  cudaError_t _code;
};

int main(void)
{
  size_t N = 1<<20; // 1M elements

  // Allocate Unified Memory -- accessible from CPU or GPU
  CudaArray<float> x(N, 1.0f);
  CudaArray<float> y(N, 2.0f);

	if(!x.isValid() || !y.isValid()) { return 0; }
	 
  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (size_t i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
  
  return 0;
}
