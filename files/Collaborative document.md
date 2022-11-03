![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document

2022-11-02-ds-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.

## 🎓 Certificate of attendance

If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website

* [Workshop]([<url>](https://esciencecenter-digital-skills.github.io/2022-11-02-ds-gpu/))
* [Google Colab](https://colab.research.google.com)

## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Hanno Spreeuw

## 🧑‍🙋 Helpers

 Suvayu Ali, Olga Lyashevska

## 🗓️ Agenda

| Time  | Activity |
| ----- | -------- |
| 09:30 | Welcome |
| 09:45 | Introduction |
| 10:00 | Convolve an image with a kernel on a GPU using CuPy |
| 10:30 | **Coffee break** |
| 10:45 | Running CPU/GPU agnostic code using CuPy | 
| 11:15 | **Coffee break** |
| 11:30 | Run your Python code on a GPU using Numba |
| 12:00 | **Lunch break** |
| 13:00 | Introduction to CUDA |
| 14:00 | **Coffee break** |
| 14:15 | CUDA memories and their use |
| 15:00 | **Coffee break** |
| 15:15 | Data sharing and synchronization |
| 16:15 | Wrap-up |
| 16:30 | End |

All times in the agenda are in the **CET** timezone.

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🔧 Exercises


### Challenge: fairer runtime comparison CPU vs. GPU

Compute again the speedup achieved using the GPU, but try to take also into account the time spent transferring the data to the GPU and back.

Hint: to copy a CuPy array back to the host (CPU), use the `cp.asnumpy()` function..


### Challenge: $\kappa, \sigma$ clipping on the GPU

Now that you understand how the $\kappa, \sigma$ clipping algorithm works, perform it on the GPU using CuPy and compute the speedup.

### Challenge: segmentation on the GPU

It is now time to use CuPy to perform the segmentation on the GPU and compute the speedup.


### Challenge: labelling on the GPU

Use CuPy to perform the connected component labelling on the GPU and compute the speedup.


### Challenge: Compute prime numbers with CUDA

Given the following Python code, similar to what we have seen in the previous episode about Numba, write the missing CUDA kernel that computes all the prime numbers up to a certain upper bound.

```python
import numpy as np
import cupy
import math
from cupyx.profiler import benchmark

# CPU version
def all_primes_to(upper : int, prime_list : list):
    for num in range(0, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            prime_list[num] = 1

upper_bound = 100_000
all_primes_cpu = np.zeros(upper_bound, dtype=np.int32)

# GPU version
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
   for ( int number = 0; number < size; number++ )
   {
       int result = 1;
       for ( int factor = 2; factor <= number / 2; factor++ )
       {
           if ( number % factor == 0 )
           {
               result = 0;
               break;
           }
       }

       all_prime_numbers[number] = result;
   }
}
'''
# Allocate memory
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)

# Setup the grid
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)

# Benchmark and test
%timeit -n 1 -r 1 all_primes_to(upper_bound, all_primes_cpu)
execution_gpu = benchmark(all_primes_to_gpu, (grid_size, block_size, (upper_bound, all_primes_gpu)), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")

if np.allclose(all_primes_cpu, all_primes_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

There is no need to modify anything in the code, except writing the body of the CUDA `all_primes_to` inside the `check_prime_gpu_code` string, as we did in the examples so far.

Be aware that the provided CUDA code is a direct port of the Python code, and therefore **very** slow. If you want to test it, user a lower value for `upper_bound`.




## 🧠 Collaborative Notes

GPUs have specialised hardware useful for rendering graphics.  This turns out to be useful for certain kinds of computation (e.g. matrix multiplication).  So often scientific computation can be accelerated by GPUs.

### Image convolution

We will start with generated image, and convolve with a grid of delta functions
```python=
deltas = np.zeros((2048, 2048))
deltas[8::16, 8::16] = 1
```
We can plot this with `matplotlib`
```python=3
import matplotlib
%matplotlib inline  # for rendering matplotlib plots in Jupyter notebooks
import pylab as pyl
pyl.imshow(deltas[0:64, 0:64])
pyl.colorbar()
```
![](https://i.imgur.com/gqfLh22.png)


Continuous:
$f * g(t) = \int_{-\infty}^{\infty} g(t)f(t-x)dx$

Discretised:
$f * g(n) = \displaystyle\sum_{m=-\infty}^{\infty} g(n)f(n-m)$

You can understand the steps to do the convolution with the following animation
![](https://carpentries-incubator.github.io/lesson-gpu-programming/fig/2D_Convolution_Animation.gif)

```python=8
x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
distsq = x**2 + y**2
gauss = np.exp(-distsq/2)
pyl.imshow(gauss)
pyl.show()
```
![](https://i.imgur.com/S9lNmRP.png)

We can now do the convolution with the grid of deltas:
```python=13
from scipy.signal import convolve2d as convolve2d_cpu

convolved_image_using_CPU = convolve2d_cpu(deltas, gauss)
pyl.imshow(convolved_image_using_CPU[0:64, 0:64])
pyl.show()
```
![](https://i.imgur.com/bHqBTXM.png)

### Benchmarking

This convolution is happening on the CPU.  We can time it so that we can compare later on.
```python
%timeit -n 1 -r 1 convolve2d_cpu(deltas, gauss)
# 2.68 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
```

```python=
from cupyx.scipy.signal import convolve2d as convolve2d_gpu
convolved_image_using_GPU = convolve2d_gpu(deltas, gauss)  # this will raise a TypeError
# TypeError: Cannot construct a dtype from an array
```
This is because the data is not on the GPU, and we need to copy our data to the GPU for us run our computation there.  This has to be done by us manually.
```python=4
deltas_gpu = cp.asarray(deltas)
gauss_gpu = cp.asarray(gauss)
convolved_image_using_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
```
Let's time it again, and compare with the CPU time
```python
%timeit -n 7 -r 1 convolve2d_gpu(deltas_gpu, gauss_gpu)
# 2.57 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 7 loops each)
```
However this timing method isn't appropriate for benchmarking GPU code.  This is a artifact of how Python interacts with the GPU.  The kernel is sent to the GPU for execution, and we get back our python prompt.  That doesn't mean the kernel finished running.  So we should use facilities prov

```python=7
from cupyx.profiler import benchmark

convolve_gpu_timing = benchmark(convolve2d_gpu, (deltas_gpu, gauss_gpu), n_repeat=10)
fastest_convolution_gpu = np.min(convolve_gpu_timing.gpu_times)
print(f"fastest convolution took {fastest_convolution_gpu}")
# fastest convolution took 8.198e-03
```

We can compare with the CPU timing:
```python=
convolution_cpu_timing = %timeit -n 1 -r 1 -o convolve2d_cpu(deltas, gauss) # save result in python variable
print(f"speedup factor: {convolution_cpu_timing.best/fastest_convolution_gpu}")
# speedup factor: 328.5230238274502
```

We can get more information about the underlying hardware with some vendor specific commands.  For `NVidia`, you can do this:
```
!nvidia-smi
Wed Nov  2 11:15:10 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN RTX    On   | 00000000:B1:00.0 Off |                  N/A |
| 41%   35C    P8    28W / 280W |    358MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     10247      C   ...core-10.3.0/bin/python3.9      355MiB |
+-----------------------------------------------------------------------------+
```

If we do a fairer comparison, where we also account for time taken to copy the data to and from the GPU, we will notice the speedup factor
```python=32=16
def transfer_compute_transferback():
    deltas_gpu = cp.asarray(deltas)
    gauss_gpu = cp.asarray(gauss)
    convolved_image_using_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
    convolved_image_using_GPU_copied_to_host = cp.asnumpy(convolved_image_using_GPU)
   
execution_gpu = benchmark(transfer_compute_transferback, (), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")
print(f"speedup factor: {convolution_cpu_timing.best/gpu_avg_time}")
# 0.030851 s
# speedup factor: 87.30176926834557
```
*Note:* Since we are waiting for the GPU to finish, we can use the usual `timeit` magic.
```python
%%timeit

deltas_gpu = cp.asarray(deltas)
gauss_gpu = cp.asarray(gauss)
convolved_image_using_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
convolved_image_using_GPU_copied_to_host = cp.asnumpy(convolved_image_using_GPU)
# 30.3 ms ± 264 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
print(f"speedup factor: {convolution_cpu_timing.best/.0303}")
# speedup factor: 88.88848270397179
```
Numpy functions expects a well-defined API, that `cupy` implements.  This means often we can just pass a cupy array to numpy without worrying about transfering the data back to the host.

```python
np.allclose(convolved_image_using_GPU, convolved_image_using_CPU)
# array(True)
```

### A real world example: image processing in radio astronomy
Load the image and ensure the image is in little endian byte order:
```python=
from astropy.io import fits
image_file = "path/to/file"
with fits.open(image_file) as hdul:
    data = hdul[0].data.byteswap().newbyteorder()
```

We can plot the image with:
```python=4
from matplotlib.colors import LogNorm

maxim = data.max()

pyl.matshow(data[500:1000, 500:1000], cmap=pyl.cm.gray_r, norm=LogNorm(vmin = maxim/100, vmax=maxim))
pyl.colorbar()
```
 We will use a technique called $\kappa, \sigma$ (kappa-sigma) clipping to remove background.  We flatten our 2D data to make it easier.
```python=11
# Flattening our 2D data first makes subsequent steps easier.
data_flat = data.ravel()
```
CPU implementation:
```python=12
def kappa_sigma_clipper(data_flat):
    while True:
         med = np.median(data_flat)
         std = np.std(data_flat)
         clipped_lower = data_flat.compress(data_flat > med - 3 * std)
         clipped_both = clipped_lower.compress(clipped_lower < med + 3 * std)
         if len(clipped_both) == len(data_flat):
             break
         data_flat = clipped_both  
    return data_flat
```

```python=22
data_clipped = kappa_sigma_clipper(data_flat)
timing_ks_clipping_cpu = %timeit -o kappa_sigma_clipper(data_flat)
fastest_ks_clipping_cpu = timing_ks_clipping_cpu.best
print(f"Fastest CPU ks clipping time = {1000 * fastest_ks_clipping_cpu:.3e} ms.")
# 555 ms ± 1.27 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# Fastest CPU ks clipping time = 5.537e+02 ms.
```

GPU implementation:
```python=28
data_flat_gpu = cp.asarray(data_flat)
data_gpu_clipped = kappa_sigma_clipper(data_flat_gpu)
timing_ks_clipping_gpu = benchmark(kappa_sigma_clipper, (data_flat_gpu.ravel(), ), n_repeat=10)
fastest_ks_clipping_gpu = np.min(timing_ks_clipping_gpu.gpu_times)
print(f"{1000 * fastest_ks_clipping_gpu} ms")
# 35.92 ms

speedup_factor = fastest_ks_clipping_cpu/fastest_ks_clipping_gpu
print(f"The speedup factor for ks clipping is: {speedup_factor}")
# The speedup factor for ks clipping is: 15.417334751352342
```

#### Segmentation image
```python=
mean_ = data_clipped.mean()
median_ = np.median(data_clipped)
stddev_ = np.std(data_clipped)
max_ = np.amax(data_clipped)
print(f"mean = {mean_:.3e}, median = {median_:.3e}, sttdev = {stddev_:.3e}, maximum = {max_:.3e}")
# mean = -1.945e-06, median = -9.796e-06, sttdev = 1.334e-02,maximum = 4.000e-02

stddev_gpu_clipped = np.std(data_gpu_clipped)
print(f"standard deviation of background_noise = {stddev_gpu_clipped:.4f} Jy/beam")
# standard deviation of background_noise = 0.0133 Jy/beam

data_gpu = cp.asarray(data)
threshold_ = 5 * stddev_gpu_
segmented_image_gpu = np.where(data_gpu > threshold, 1,  0)
timing_segmentation_CPU = %timeit -o np.where(data_gpu > threshold, 1,  0)
fastest_segmentation_CPU = timing_segmentation_CPU.best 
print(f"Fastest CPU segmentation time = {1000 * fastest_segmentation_CPU:.3e} ms.")
# 148 µs ± 11.5 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
# Fastest CPU segmentation time = 1.479e-01 ms.
```

#### Labelling of the segmented data
```python=
from scipy.ndimage import label
labelled_image = np.empty(data.shape)
number_of_sources_in_image = label(segmented_image_gpu.get(), output = labelled_image)
sigma_unicode = "\u03C3"
print(f"The number of sources in the image at the 5{sigma_unicode} level is \
{number_of_sources_in_image}.")

timing_CCL_CPU = %timeit -o label(segmented_image_gpu.get(), output = labelled_image)
fastest_CCL_CPU = timing_CCL_CPU.best
print(f"Fastest CPU CCL time = {1000 * fastest_CCL_CPU:.3e} ms.")
# The number of sources in the image at the 5σ level is 185.
# 35.2 ms ± 34.7 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# Fastest CPU CCL time = 3.513e+01 ms.
print(f"These are all the pixel values we can find in the labelled image: \
{np.unique(labelled_image)}")

from cupyx.scipy.ndimage import label as label_gpu
labelled_image_gpu = cp.empty(data_gpu.shape)
number_of_sources_in_image = label_gpu(segmented_image_gpu, output = labelled_image_gpu)

print(f"The number of sources in the image at the 5{sigma_unicode} level is {number_of_sources_in_image}.")

timing_CCL_GPU = benchmark(label_gpu, (segmented_image_gpu, None, labelled_image_gpu), n_repeat=10)
fastest_CCL_GPU = np.amin(timing_CCL_GPU.gpu_times)
print(f"Fastest CCL on the GPU is {1000 * fastest_CCL_GPU} ms")
print()
speedup_factor = fastest_CCL_CPU/fastest_CCL_GPU
print(f"The speedup factor for CCL is: {speedup_factor}")
# The number of sources in the image at the 5σ level is 185.
# Fastest CCL on the GPU is 0.7349439859390259 ms
# The speedup factor for CCL is: 47.32004594530196
```

### GPU kernels

Consider the following Python code:
```python=
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
```

If we implement it for the GPU, we can do the array operations in parallel, and gain significant speedup.

*Note:* since the GPU is a separate device, any GPU kernel doesn't have return values.  Result of computations have to be "returned" by one of the arguments of the 

```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
```
We can run custom code on the GPU by reading the above code as a string, and compiling with CuPy.
```python=
import cupy

# size of the vectors
size = 1024

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

# CUDA vector_add
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")
vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```
Compare with the CPU implementation:
```python=
import numpy as np

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

# test
np.allclose(c_cpu, c_gpu)
# array(True)
```

Let's say we change `size` to 2048, you will get an exception from cupy.  This is because in CUDA there is a hard upper limit of 1024 threads.  So we could manually change the number of blocks in the call to `vector_add_gpu`:
```python
size = 2048
vector_add_gpu((2, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```
However this would lead to incorrect results, because in your kernel, we didn't handle multiple blocks, and we would be running the kernel twice on one half of the data instead of running once on the whole data.
```python
np.allclose(c_cpu, c_gpu)
# array(False)
```

To extend to more than one block, we can extend as:
```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = blockIdx.x*blockDim.x + threadIdx.x;
    if (item < size) {
        C[item] = A[item] + B[item];    
    }
}
```
We also add a bounds check for size, so that we don't corrupt memory that we do not own.

We can generalise the block calculation further:
```python=
import math

threads_per_block = 1024
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
```
CUDA also offers a block of *shared memory*, which is shared between all threads in a block.  It is denoted by prefixing the declaration with the `__shared__` attribute.  Let's look at filling an integer histogram.
```python=
def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array[item] + 1
        
input_array = np.random.randint(256, size=2048, dtype=np.int32)
output_array = np.zeros(256, dtype=np.int32)
histogram(input_array, output_array)
```
If we naively port it to the GPU as:
```cpp=
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    output[input[item]] = output[input[item]] + 1;
}
```
This would be incorrect, as the same bins are being overwritten by different threads.  We can fix the issue as:
```cpp=
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    atomicAdd(&(output[input[item]]), 1);
}
```
But this approach depends a lot on the input data. A sequence such as *0 0 0 0 1* would have to wait 4 time as long as there are 4 

We can adapt further using a temporary shared histogram
```cpp=
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];

    temp_histogram[threadIdx.x] = 0; // initialise
    __syncthreads();

    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    atomicAdd(&(output[input[item]]), temp_histogram[threadIdx.x]);
}
```


## To load the Galactic Center FITS image.

### Only wWhen using Google Colab

```python=
from google.colab import files
uploaded = files.upload()
```

#### To retrieve the Galactic Center FITS image (not needed for anyone using SURF Jupyter notebooks)

The image can be downloaded at this [link](https://github.com/carpentries-incubator/lesson-gpu-programming/blob/gh-pages/data/GMRT_image_of_Galactic_Center.fits).

## Questions about computation on a GPU
- Is there a library like `cupy` that supports the new Macbook Metal GPUs?
    - Not that I am aware of, but CuPy does already support multiple backends so it is not impossible that they'll support it at some point
- Best practice for writing the `cupy` RawKernel in seperate file not just raw string (and integration with focus on pytorch)
    - Just write your CUDA code in a separate file, read it into a Python string, and pass it to `RawKernel`. Developing the CUDA code in a separate file is more maintainable.
- What is the effect of number of threads on memory footprint (for example many local variable per thread)
    - Having many variables per thread can reduce the number of threads that the GPU can run in parallel, thus reducing overall performance of your code.
- More info on feeding 2D images and 2D blockIds in `cupy`, maybe gaussian as example
    - You can find examples of 2D grids and blocks in the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- Could you add some practical materials on divergence?
    - I found this brief explanation on [thread divergence](https://cvw.cac.cornell.edu/gpu/thread_div)


## 📚 Resources

* [Lesson notes and material](https://carpentries-incubator.github.io/lesson-gpu-programming/)
* [HackMD Markdown Guide](https://www.markdownguide.org/tools/hackmd/)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [Numba User Guide](https://numba.pydata.org/numba-doc/latest/user/index.html)
* [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [Prace course on new GPU architectures (December 2022)](https://events.prace-ri.eu/event/1466/)
* [`ufunc` API used by NumPy](https://numpy.org/doc/stable/reference/ufuncs.html)
