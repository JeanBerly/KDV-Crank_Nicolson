# Define the compiler and flags
CXX = g++
CXXFLAGS = -I ./Eigen/

# Define the source file and output binary
SRC = kdv_Crank_nicholson.cpp
OUT = kdv.out

# Target to compile the code
compile:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT)
