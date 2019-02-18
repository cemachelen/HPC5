# CUDA modules
# autoinit isn't necessary but it helps
# Initialization, context creation, and cleanup
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
# Python modules
import numpy
# Step 1 transfer data
a = numpy.random.randn(4,4)
# Python automatically makes 64bit
# nVida supports 32 bit 
a = a.astype(numpy.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
# Executing a Kernel
# 
mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)
# now the code is compiled and loaded on to the
# device
func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1)
#  
a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print (a_doubled)
print (a)
