Here I have organized my thoughts as I was performing an experiment on [Gnocchi](https://github.com/gnocchixyz/gnocchi/) - the database


- [What and Why](intro.md)  
       Its an experiment to explore General-Purpose-GPU compute space where I try to shift computation from CPU to GPU for the OpenSource project Gnocchi 

- [How](profiling.md)  
      This part goes into implementation details. It explains the iterations based on profiling results.

- [Now What](conclusion.md)  
      It talks about Gpu-fication of other parts of the library. It also has my learning and conclusions from the experiment. 



#### Repo Readme:

This repo has experiments and analysis for implementing various kernels for the carbonara library from the Gnocchi project.

The repo is organized into various branches.
- v12 This branch has the version v1 and v2
- V3 (final) this branch is for version v3 and also the final implementation 
- v3.1 This was intermediate improvement over v3 but the speed-up was not great
- v4 This implements the kernel using stream, a more advanced approach
- threaded Another approach where work is launched from different threads, but overhead is too high


### Originally published here:
https://www.suse.com/c/an-experiment-with-gnocchi-the-database-part-1/

https://www.suse.com/c/an-experiment-with-gnocchi-the-database-part-2/

https://www.suse.com/c/an-experiment-with-gnocchi-the-database-part-3/
