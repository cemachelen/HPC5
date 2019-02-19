# HPC 5

My Personal Notes from the Leeds University [HPC5](https://arc.leeds.ac.uk/training/introduction-to-gpu-programming-with-cuda/) training course

Course Link: [goo.gl/NMGpWk](https://docs.google.com/presentation/d/1eIty3x3C57gxsrRLrzbY6iYoVQ397_h2Jn92ZttkKpQ/edit#slide=id.g2fcb30acec_0_0)

The code follows some of the tasks in the course work sheet. Commits every question and bugfix.

:construction: *These are just notes from a course, this is not developed work*  :construction:

### [General GPU](https://docs.google.com/presentation/d/1ZOhz8HfvGn_va7sUvtn3x6W2f9lflZXgyoW_BTXx7-w/edit#slide=id.p11)
* As clockspeeds stall best to move to accelerators
* ARC3 has the discontinued Xeon Phi nodes that go unused
* nVida produce Tesla GPUS like GeForce but for HPCS
* [ARC3](https://arc.leeds.ac.uk/systems/arc3/) has 6 GPU Nodes (4 P100s) and 2 ( 2 k80 ) nodes. ARC4 will have less.
   * `#$ -l coproc_p100=1` Requests 1/4 =  =4 gives whole
   * `#$ -l coproc_k80=1` Requests 1/2 =2 gives whole
* ARC3 GPUs are under utilised
* Always use CPU and GPUs together the GPUs do the heavy lifting, CPU requirement depends on code

### Coding 
* Use Libraries e.g. MATLAB, Ansys
* Directives like OpenACC compile and auto gen
* CUDA extension to C for nVida whereas OpenCL works cross platform

### [CUDA](https://docs.google.com/presentation/d/1cBg-FuWYZhDrkk9tpBZRS5npjo3GTlfxx6Jv_hFXrhw/edit#slide=id.p6)
* [Created by nVida](http://supercomputingblog.com/cuda/what-is-cuda-an-introduction/)
* Not super user friendly even PyCUDA
* Works by sending code to CPU and separate code to GPU Kernel
* Each Streaming Multiproc (SM) has multiple cores
* ARC3 p100s has 56 SMs with 64 single precision cores  or 32 Double Precision cores
* CUDA vectors can help control location in GPU architecture

### CUDA extra
* **NB** `nvcc` doesn't work with every version of every compiler and it's hard to find out which
* [PyCUDA](https://docs.google.com/document/d/1Mprn4iicYpLifyL_id6dEABEs_50GhxCYZYRlxUXzq4/edit) (includes GPU interactive sessions) 
   * `import pycuda.autoinit` important!!
   * `a_gpu = cuda.mem_alloc(a.nbytes)` it's important to cast arrays to allocate memory
   * Still requires C for kernel but pro is that skip the compiling code and allows for using numpy functions
* [CUDApython (numba)](https://devblogs.nvidia.com/numba-python-cuda-acceleration/)
* [Alexnet](https://github.com/ykpengba/AlexNet-A-Practical-Implementation) example on github
   * [tutorial](https://medium.com/coinmonks/understand-alexnet-in-just-3-minutes-with-hands-on-code-using-tensorflow-925d1e2e2f82)

### Useful links
* [NVIDA ecosystem](https://developer.nvidia.com/tools-ecosystem) has a whole suite of accelerated code and libraries etc.

## Trouble shooting

* Compiler requires .cu extension
* tabs are not accepted

<hr>

# Acknowledgements

The course was from the [ARC Training Course:](https://arc.leeds.ac.uk/training/) [HPC5](https://arc.leeds.ac.uk/training/introduction-to-gpu-programming-with-cuda/), delivered by Martin Callaghan.
