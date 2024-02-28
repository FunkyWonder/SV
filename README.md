# SV
SV

- dynamic parallelism (were using it)
- cuda is a pain to get working
- Why was PF a function pointer? Pointless, causes segfault when rewritten to cuda.

What is the range of rnor? 

https://stackoverflow.com/questions/67966258/cuda-architectures-is-empty-for-target-cmtc-28d80

Generate build files:

cmake -S/home/arnec/Documents/GitHub/SV/src -B/home/arnec/Documents/GitHub/SV/build -GNinja

Compile:
ninja