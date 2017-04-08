# Source files

BIN = hw5_conv
CC_SRC := $(wildcard *.cpp)
CU_SRC := hw5_conv.cu

# Compilor and linker options

#CXX=g++ -m64 -Wl,--trace
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g -std=c++11

INC_DIR= #-I/cs/local/lib/pkg/freeglut-3.0.0/include

L_DIR = -L/cs/local/lib/pkg/cudatoolkit/lib64
LIBS = cudart
LDLIBS  := $(addprefix -l, $(LIBS))
#LDLIBS  := -lGL -lglut

NVCC      = nvcc
NVCCFLAGS=-O3 -m64 -arch compute_20

# Build folder

BUILD_DIR=build
OBJS=$(CC_SRC:%.cpp=$(BUILD_DIR)/%.o)
CU_OBJS=$(CU_SRC:%.cu=$(BUILD_DIR)/%.o)

default: $(BIN)
dirs:
	mkdir -p $(BUILD_DIR)/
clean:
	rm -rf $(BUILD_DIR) *~ $(BIN)

$(BIN): dirs $(CU_OBJS) $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(CU_OBJS) $(L_DIR) $(LDLIBS)

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $< $(INC_DIR) $(CXXFLAGS) -c -o $@

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@


hw5_blas: hw5_blas.cu
	nvcc -lcublas hw5_blas.cu -o hw5_blas


.PHONY: sync
sync:
	rsync -rav .. q7w9a@thetis.ugrad.cs.ubc.ca:~/cs418

.PHONY: syncout
syncout:
	rsync -rav q7w9a@thetis.ugrad.cs.ubc.ca:~/cs418/hw5/out /tmp/

CMD?=zsh -i

.PHONY: shell
shell: sync
	ssh q7w9a@thetis.ugrad.cs.ubc.ca -C "ssh lin13.ugrad.cs.ubc.ca -C \"cd cs418/hw5 && zsh -ic \\\"$(CMD)\\\"\""
