# Define the compiler and flags
CXX = g++
CXXFLAGS = -g -I ./Eigen/

# Define the source file and output binary
SRC = kdv_Crank_nicolson.cpp
OUT = kdv.out

# Target to compile the code
compile:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT)
