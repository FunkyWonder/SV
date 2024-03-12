# SV
SV

- dynamic parallelism (were using it)
- cuda is a pain to get working
- Why was PF a function pointer? Pointless, causes segfault when rewritten to cuda.

What is the range of rnor? 

https://stackoverflow.com/questions/67966258/cuda-architectures-is-empty-for-target-cmtc-28d80

Generate build files:
```bash
cmake -Ssrc -Bbuild -GNinja
```
Compile:
```bash
ninja
```



also works:
nvcc -c ./src/main.cu -o ./test/SV-file.o --gpu-architecture=compute_86 --gpu-code=compute_86 -g -G -rdc=true --diag-suppress=177,550

nvcc ./test/SV-file.o -o ./test/SV-file -lcudadevrt -lcudart  --gpu-architecture=sm_86 