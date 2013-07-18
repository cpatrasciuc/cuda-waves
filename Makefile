# Copyright (c) 2011 Cristian Patrasciuc. 
# Use of this source code is governed by an MIT license that can be
# found in the LICENSE file.

CUDADIR ?= /usr/local/cuda/
EXENAME = cuda_demo

all: build run clean

build:
	cp main.cpp main.cu
	$(CUDADIR)/bin/nvcc --compiler-bindir=compilers/ --library glut -o $(EXENAME) main.cu 

run:
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(CUDADIR)/lib64/ ./$(EXENAME)

clean:
	-rm *.o *.cu $(EXENAME)
