CUDADIR ?= /usr/local/cuda/
EXENAME = cuda_demo

all:
	cp main.cpp main.cu
	$(CUDADIR)/bin/nvcc --compiler-bindir=compilers/ --library glut -o $(EXENAME) main.cu 

run:
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(CUDADIR)/lib64/ ./$(EXENAME)

clean:
	-rm *.o *.cu $(EXENAME)
