nvcc -c ./src/main.cu -o ./test/SV-file.o --gpu-architecture=compute_86 --gpu-code=compute_86 -rdc=true --diag-suppress=177,550 # -g -G
nvcc ./test/SV-file.o -o ./test/SV-file -lcudadevrt -lcudart  --gpu-architecture=sm_86

cd test
cuda-gdb SV-file