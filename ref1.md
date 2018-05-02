
### Background (Why):

Anyone who knows Openstack is aware of its component Ceilometer (Telemetry). Ceilometer was huge, its main purpose was to help with the Monitoring and Metering of whole of th OpenStack. But because of the scale at which OpenStack operates, Ceilometer would often fall behind and become a major bottleneck for the same function it was designed to do. It had one more problem:- of aggregating these metrics and providing them in a more consumable format.
 
So it was broken into different parts and this gave rise to a time series database - [Gnocchi.](https://github.com/gnocchixyz/gnocchi/). Gnocchi stores data directly on the disk in its custom format. And then using a group of processes on same node. If the storage disk is shared across different nodes then Gnocchi can form a cluster of processes from these nodes to aggregate the data. The received data is referred as metrics, as it the measured usage of the various resources(like cpu, ram, disk, network) in OpenStack.
 
One of the ways Gnocchi is different is the way it handles aggregation. Instead of processing the metrics to generate aggregates (sum,mean,std-deviation etc) whenever and however they are asked for, it makes the user choose on what kind of aggregations need to be done before defining the metrics and calculates and stores the metrics and aggregates immediately.
 
So the user has to tell Gnocchi that the metrics *X* will be received at interval *A* but should be also aggregated to interval *B* to make *Y* aggregates and then data should be archived after *N* time. 
For example: 
Receive *cpu-usage-vm* every *5* sec and generate *sum,mean,std-deviation* for *1m, 10m, 30m, 1h, 12h, 24h, 7d, 15d, 30d*, delete after *30days*.
 
As the received data are immediately aggregated and stored, retrieval is as good as a single disk read. In a way the aggregation problem (as faced by ceilometer, as mentioned above) is never faced and therefore never needed to be solved.  But this pre-computing was still a problem as the amount of data could get really overwhelming and Gnocchi tries to solve this problem by letting a large number of worker aggregate the data forming a cluster over distributed nodes.
 
However this peculiar way of restricting the problem space and pre-committing the range of values for the various of aggregations allowed some space for **Experiments**.
 
 
### Speeding up (How)
 
The workers working across different nodes have their work partitioned at run time.
 - Input size for metrics to be aggregated is already known
 - The aggregation intervals are already defined when the metric is defined
 

    If one knows there are X points at A intervals to be aggregated to generate Y points at B intervals then this forms a classic case of **"Parallelization"**, where you can employ Y threads to aggregate (X/Y) points from X, as shown in the following figure.

| ![Reduction example from devblogs.nvidia.com](https://devblogs.nvidia.com/wp-content/uploads/2014/01/reduce.png) | ![Quadro K620 (Image from Nvidia.com)](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/quadro-desktop-k620-297-udtm@2x.jpg)
|:---:|:---:|
| Reduction example from devblogs.nvidia.com | Quadro K620 (Image from Nvidia.com) |

Now there are various ways one could implement this *Parallelization* in the form of threads by offloading it to the cores of the CPU. But Still there are only so many cores a CPU has and the poor CPU is already busy juggling
processes and there is always some process fighting for the CPU's attentions. Also the CPU has only so many cores, so the idea was to use a GPGPU to get the massive parallelism required.
  
So for this experiment I decided to leverage the GPU my machine was already equipped with, an NVDIA Quadro K620.
 
The specs of the gpu can be seen [here](https://images.nvidia.com/content/pdf/quadro/data-sheets/75509_DS_NV_Quadro_K620_US_NV_HR.pdf). It is a medium grade GPU, with good compute capability (5.0) and 2GB of ram.
 

#### Implementation (is it even possible ?)
 
The first approach was to get atleast the benchmark passing and with a speed up.
 
The benchmark covered a wide range of data formats and multiple areas of the [carbonara library](https://github.com/gnocchixyz/gnocchi/blob/master/gnocchi/carbonara.py).  It has (De-)Serialization  evaluations, resampling and aggregations (sum,min,max,mean,average), and in addition ithe input data for all these benchmark evaluation was not just arrays of random values, but arrays of zeros, random floats between in the range [-1,1]. So 
the requirements was to beat all these benchmarks.
 
First step was to insert a class in [carbonara library (my copy)](https://github.com/sjamgade/carbs/blob/04b9db71e0e4c6d14073c4bd827a23e4dd0437ab/carbonara.py#L225) which could call the cuda code. And simulate addition with a fixed [resample rate(30s)](https://github.com/sjamgade/carbs/blob/04b9db71e0e4c6d14073c4bd827a23e4dd0437ab/carbonara.py#L194) and [fixed input size](https://github.com/sjamgade/carbs/blob/04b9db71e0e4c6d14073c4bd827a23e4dd0437ab/carbonara.py#L191) for all zeros and continuous numbers as floats.
 
This itself gave a performance boost, from here on the game was to increase performance and throughput while keeping the semantics intact.
 
Getting the initial kernel going was easy. There was already a speedup of 5x.
 
The first bottleneck was the amount of data processed in one go. The benchmark assumed an input size of 3600 points for calculation.  This number was very low. Since the GPU does computation in parallel reducing 3600 points to 600 points would in the naive implementation about 600 threads (6 aggregates each). As recent GPGPUs have generally a compute capability > 5, which allows a block to contain a maximum of 2048 threads so 600 was very small, and thus had to be increased to `1024 * 1024 * 6`
 
There is one more reason for the number to be so high, since the GPU has its own memory this data needs to be copied over from HtoD (host to device, cpu-ram to gpu-ram). But even this device memory is not fast enough and every access to load data (from device memory to L2 cache) for computation is expensive, so to hide the latency lots of computation needs to be done. So that while data is loading, any already loaded data can be used for computation. This is just one the ways to hide that latency.
 
#### v1 
This lead to the [first kernel(v1)](https://github.com/sjamgade/carbs/blob/04b9db71e0e4c6d14073c4bd827a23e4dd0437ab/carbonara.py#L251) being created.
Code:
```cpp
__global__ void v1(float *a, int *i) {  
  int perthread=i[0];    
  int counter = i[0]-1;    
  int col = blockIdx.x * blockDim.x + threadIdx.x;    
  int row = blockIdx.y * blockDim.y + threadIdx.y;    
  int index = col * perthread + row;    
  for(;counter;counter--)    
    atomicAdd(a+index, a[index+counter]);    
}
```

Results :
```
Simple continuous range
  time to aggregate 17.5399780 msec
  resample(gpu_sum) speed: 2.34 Hz; 427.72102356 msec
  time to aggregate 56.2980175 msec
  resample(cpu_sum) speed: 2.84 Hz; 352.295875549 msec
Just zeros
  time to aggregate 16.6831017 msec
  resample(gpu_sum) speed: 3.20 Hz; 312.555074692 msec
  time to aggregate 56.0901165 msec
  resample(cpu_sum) speed: 2.87 Hz; 348.960161209 msec
```



Just firing random request for data from different threads leads to a problem of non-coalesced addressing. Since there is no guarantee over how the scheduler will schedule the threads the data access patterns from these threads needs to be aligned to achieve maximum throughput of the L2 cache bus; the bandwidth of this bus is device dependent but mostly in multiples of 32 bytes for older GPGPU and 128 bytes in newer devices.
 
Since the pattern of data access in the first kernel was not correct and getting the pattern correct is a difficult problem an alternate solution was implemented. This required reducing the number of data requests fired but consuming the responses as soon as possible. So each thread did more work, but also created work in a controlled manner.
 
 
 
#### v2
[This lead to the first real kernel(v2) being created.](https://github.com/sjamgade/carbs/blob/04b9db71e0e4c6d14073c4bd827a23e4dd0437ab/carbonara.py#L261)
Code:
```cpp
    __global__ void v2(float *a, int *i) {    
    int perthread=i[0];    
    int counter = i[0]-1;    
    int blklimit=1024;    
    int blk=0;    
    for (blk=0;blk<blklimit;blk++) {    
	    int counter = i[0]-1;    
	    int col = blk * blockDim.x + threadIdx.x;    
	    int row = blockIdx.y * blockDim.y + threadIdx.y;    
	    int index = col * perthread + row;    
	    for(;counter;counter--)    
	        atomicAdd(a+index, a[index+counter]);
        }
    }
```
Results:
 ```
Simple continuous range
  time to aggregate 21.0869312 msec
  resample(gpu_sum) speed: 2.04 Hz; 490.113973618 msec
  time to aggregate 71.5970993 msec
  resample(cpu_sum) speed: 2.36 Hz; 423.68888855 msec
Just zeros
  time to aggregate 20.2741623 msec
  resample(gpu_sum) speed: 2.67 Hz; 374.500989914 msec
  time to aggregate 72.5300312 msec
  resample(cpu_sum) speed: 2.36 Hz; 424.036026001 ms
```
v2 divided the entire set of points into `blks` and threads would once chomp upon its own `blk`, so now we would have reduced [occupancy](https://github.com/sjamgade/carbs/blob/v12/v2.pdf) with less threads, but still have enough work generated to keep the threads busy and enough read requests to keep the "Global Memory" read time hidden by overlapping it with work of  threads from other ["Warps"](https://en.wikipedia.org/wiki/Thread_block#Warps).


 <img align="right" src="https://www.spiedigitallibrary.org/ContentImages/Journals/JBOPFO/21/1/017001/FigureImages/JBO_21_1_017001_f008.png" width="50%" border="1" style="border-color:black;">

 
Even the kernel was doing fine here but nvprof was still reporting efficiency issues.  One of those was the global-memory access pattern. Since each thread was responsible for aggregating the 6 floats, it would load 6 non- contigous memory locations.  This is a problem in the world of GPUs, because the GPU loads not just one memory location but an entire segment of 128 bytes from global memory into the L2/L1 cache.

So if 32 threads of a warp request 4 bytes(float) from non contiguous locations, the GPU would get request for locations that are 24(6\*4) bytes apart form each thread in a warp, resulting into 768 (24\*32) bytes, falling into 6 (768/128) segments, and since there is just one bus all these request will go serially and the warp is stalled.

This is a huge waste of bandwidth as only 24 bytes out of 768 bytes are actually used.  There are many ways to circumvent this problem and thus V3 came into existence.




#### v3
Code:
```cpp
    __global__ void v3(float *a, int *i) {    
    int perthread=i[0];    
    int gridsize = 4*4;   
    int blklimit=1024/gridsize;    
    int blk=1;    
    int row, col, index;    
    __shared__ float inter[6*1024/16];    
    int counter = 0;    
    for (blk=1;blk<blklimit;blk++) {    
	    counter = perthread-1;    
	    col = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x*blk*perthread;    
	    row = blockIdx.y * blockDim.y + threadIdx.y;    
	    int tid = threadIdx.x;    
	    for(counter=0;counter<perthread;counter++){    
		    index = col + blockDim.x*counter + row;    
		    inter[blockDim.x * counter + tid] = a[index];   
	    }    
	    __syncthreads();    
	    counter = perthread-1;	        
	    int x = 0;    
	    for(;counter;counter--){    
		    x += inter[blockDim.x * counter + tid];    
	    }    
	    a[index] = x;
        __syncthreads();    
	}
 }
```
Results:
```
Simple continuous range
  time to aggregate 14.5051479
  resample(gpu_sum) speed: 1.96 Hz
  time to aggregate 75.1349926
  resample(cpu_sum) speed: 2.26 Hz
Just Zeros
  time to aggregate 14.0569210
  resample(gpu_sum) speed: 2.62 Hz
  time to aggregate 74.8288631
  resample(cpu_sum) speed: 2.25 Hz
 ```
 

[V3](https://github.com/sjamgade/carbs/blob/9a3211808315df75a16ad12bc64da8369146d158/carbonara.py#L181) is more "static" because it was written more to test out things. It uses __shared__ memory. This shared memory is called so as it is allocated per block, so any writes from one threads are visible to other threads. And the other advantage is this shared memory is much faster that global memory, and it does not suffers from the addressing coalescing issue as faced by global memory.
 
So the approach here is to load all the data required into the shared memory using adjoining threads to load adjoining location from global memory. Once the shared memory is completely populated (as forced by `__syncthreads` call) each thread can go ahead do its aggregations(sum).  Here I have still used the similar concept of `blk` from `v2` as shared memory is sized in few KB and not in GB like the global memory and thus needs to used conservatively.
 

 
This was a good improvement over v2 and [`nvprof` did not report any major efficiency issues with the kernel](https://raw.githubusercontent.com/sjamgade/carbs/6db68dd1261f05aa76362f7dd267c85f2d96b870/v3.pdf). However this did not really fit the whole solution, because the ratio of time(work-done):time(waiting-to-begin-work) was very bad.
 
![Profiler timeline for v3 kernel, blue shade is the real kernel execution](https://raw.githubusercontent.com/sjamgade/carbs/6db68dd1261f05aa76362f7dd267c85f2d96b870/2018-04-25-162940_1905x340_scrot.png) 

The picture above shows a classic problem in case of GPUs and in general there is not much that can be done about it. As you can see most of the time the GPU is siting idle doing only memory copy between CPU and GPU. 


<img src="https://devblogs.nvidia.com/wp-content/uploads/2012/11/C2050Timeline-1024x670.png" align="right" width="50%" border="1" style="border-color:black;">


In most cases it is suggested to use [Streams](https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/). Streams are "threads" to GPU what "threads" are to CPU. Streams are also more like a pipeline where the CPU can push work for the GPU and it will get picked up whenever resources are available. Streams have one more advantage of running in parallel, So the GPU can process work from different streams independently. This property is useful to do compute and memcpy (between host and GPU memory) synchronously. Since the GPU has different memory and compute engines, these two operations can be overlapped in different streams.




 
#### v4

So to use streams to speed up the whole calculation [v4 was created](https://github.com/sjamgade/carbs/blob/4d626a02704fcd5eb481785b5b3dbe743452f503/carbonara.py#L272). It divides up the work of copying data across streams and then fires compute kernels with appropriate arguments. There was some speed up and aggregation 
performed better and results were better.

Code:
```cpp
            __global__ void v4(float *data, int *i, float *output) {
              int perthread=7;
              int blklimit=128;
              int blk=1;
              int row, col, index;
              extern __shared__ float inter[];
              int counter = 0;
              for (blk=0;blk<blklimit;blk++) {
                  counter = perthread-1;
                  col = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x*blk*perthread;
                  row = blockIdx.y * blockDim.y + threadIdx.y;
                  int tid = threadIdx.x;
                  for(counter=0;counter<perthread;counter++){
                      index = col + blockDim.x*counter + row;
                      inter[blockDim.x * counter + tid] = data[index];
                  }
                  __syncthreads();

                  int x = 0;
                  for(counter=0;counter<perthread;counter++){
                      x += inter[blockDim.x * counter + tid];
                  }
                  index = col + row;
                  output[tid] = x;
                  __syncthreads();
                  
              }
            }
```
Result:
```
Simple continuous range
  time to aggregate 10.4160309 msec
  resample(gpu_sum) speed: 1.84 Hz
  time to aggregate 79.3750286 msec
  resample(cpu_sum) speed: 1.91 Hz
```     

 ![Asynchronous copy and execution over Streams](https://raw.githubusercontent.com/sjamgade/carbs/v4/2018-04-25-113103_1887x525_scrot.png)
 So instead of using `blk`s for dividing the data, streams were used and each block performed aggregations over only a part of the data. However from the timeline it was clear that there were still more optimizations possible. But if one looked at the whole picture all I was doing is increasing complexity for no real return. As there were still other parts of the benchmark that needed attention and there were still bottlenecks for a real world use case. I tried to use `blk`s with stream but to no avail because the execution on GPU was still faster than memcpy
 
There were more optimizations that could be done, for example using a big `memalloc` and share parts of it across streams. This was already an improvement. However there was no way to speed up memcpy which had turned out to be the main bottleneck and execution would still be stalled because of memcpy.


One way I looked at reducing the time required to `memcpy` was utilizing the bus between CPU and GPU. There is a slight possibility that the PCIe but was underutilized and by launching memcpy from different threads this could be improved. However that was not the case, my CPU has a PCIe3 x16 bus so according to [this link](https://www.deskdecode.com/difference-between-pci-express-gen-1-gen-2-gen-3/) it was capable of doing atleast 16GB/s however the GPU (Quadro K620) has PCIe2 x16 bus thus limiting the mempcy speed.
 
 
So Leaving the code as it is I turned to other parts of the library which could be "Gpu"ed to gain benefits and for this I choose the v3 kernel to begin as that was simple and yet not too simple.
 
There were certain other optimizations performed like that of compiling the kernel code at load time, because a real world scenario would not require rewriting kernel (Hopefully !!). Later I removed the use of `jinja` as compiling those templates was as slow as 130, even the kernel execution was faster than that.
 
#### round_timestamps

[round_timestamps](https://github.com/sjamgade/carbs/commit/efd993d8ffb6ded2aa7d76171f428d6d4b648538)  was the easiest of all beacause each thread could operate on individual data and there was no complicated math to be done. So each `numpy.datetime64[ns]` could be represented by an `unsigned long long` on the GPU and a floor operation was nothing but an integer division. The data was written back so no extra 
allocation was necessary. [Perf analysis for round_timestamps](https://raw.githubusercontent.com/sjamgade/carbs/V3/round_timestamp.pdf)

```cpp
     __global__ void round_timestamp(unsigned long long *a, unsigned long long freq) {
	    int col;
        unsigned long long UNIX_UNIVERSAL_START64={UNIX_UNIVERSAL_START64};
        col = blockIdx.x * blockDim.x + threadIdx.x;
        a[col] = (UNIX_UNIVERSAL_START64 + ((a[col] - UNIX_UNIVERSAL_START64) / freq) * freq);
    }
 ```


![Profiler timeline for round_timestamps](https://raw.githubusercontent.com/sjamgade/carbs/V3/2018-04-25-123214_1912x374_scrot.png)
 
#### count_uniqs([diff](https://github.com/sjamgade/carbs/commit/4c3b16b1fb9da500365a8a77420deffe25d07372))
So far everything is fine and still have speed up 4x with such a naively implemented kernel. But the kernel so far has been relying on one assumption of `resample-factor`. This `resample-factor` is what the code calls `reduce_by`. When aggregating sample spaced at 5s interval to 35s interval it easy to calculate 7(35/5). This 7 so far was hard-coded into the kernel, the kernel launch parameters, and also in shared memory size calculations.
 
However this will not be the case in real world and an algorithm was needed to calculate this `resample-factor`. Calculating this number is not easy as there can be cases where data is available at 7s interval and needs to be aggregated to 13s interval or 55s interval even 300s interval. And on top of this there could be "timegaps" in data when no data is available.
 
So to count which intervals needed to be aggregated together the following algorithm was implemented:

In the kernel [count_uniqs](https://github.com/sjamgade/carbs/blob/5ec998c914e8e01230515be4193a75ef0639ca4e/carbonara.py#L41)
 
- if timestamps `t` (from round_timestamps) at position `n` is not same as that at `n-1`:
	-  then the thread `T` is the `first` one in series of `t` from `n`; and `should` be marked
- only marked thread `should` count all following `t`'s until something other than `t` is encountered.
 
This way at the end each marked thread will have the count of `uniqs` that of each timestamp that exists.
 
This kernel had many efficiency issues, the first being the non-coalesced memory access, and the second due to divergent nature of the kernel many warps would be less efficient ie divergence would either "turn-off" many threads or in some cases this divergence could lead to some serialization of "if-else-halfs".
 
However even with such a naive implementation the benchmark was fast and could beat the CPU benchmark.
 
Next step was to link this aggregation kernel and get the final results.
 
So after some thought I found that `count_uniqs` kernel could also do the aggregation as it was already picking the unique timestamps. Thus with some modifications the kernel was able to do both and the kernel (v3) was no longer required.
 

There were other small optimizations that were made which turned out as learning:

- there is no need to copy result of `round_timestamps` back to host memory as `count_uniqs` could reuse it
- aggregation was a by-product of `count_uniques` and so can be other methods of aggregation thus helping to hide latency introduced while copying data to/from GPU
 

### Conclusion
Computing on GPU requires for high workloads intense enough to hide the memcpy latency. When this condition is applied to application of metrics aggregation like Gnocchi, or Ceilometer-Api (in past) is it better to do the aggregation when it is asked for consumption rather than when it is received. Because the amount of data received (for pre-computing) at any moment will always be small in quantity, as compared to already stored data.

For example:


        @5s interval for 30 days  < data size analysed code
               1.3 million points < 1024*1024*6 points
 
And in all cases the GPU turned out to better than CPU. So if there existed a hypothetical library of such function which could do exactly what these application required then they all could benefit from it.  Such a library would be capable to doing basic aggregations, and also provide an **easy** interface to write **strong** abstractions converting into GPU code, to have custom aggregate functions.

One might argue that there are already many opensource libraries which are capable pushing computation from the GPU, but one of the important **learnings** from this experiment was:  

> Writing CUDA code is very domain specific and coming up with a generic solution will require a lot of engineering. 
> it would help to forge such a library if many applications belonging to the same OpenSource Domain can benefit. 

But for that to happen One Project will have to walk the path and demonstrably set an example for others to follow. And I would recommend Gnocchi.



[Repo](https://github.com/sjamgade/carbs)
[Wiki](https://github.com/sjamgade/carbs/wiki)
