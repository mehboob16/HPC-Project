# Use nvcc as the CUDA compiler
NVCC = nvcc

# NVCC Flags: Optimize at level 2 and pass -Wall to the host compiler.
NVCCFLAGS = -O2 -Xcompiler="-Wall"

# Executable and source names
EXE = v3.exe
SRC = v3.cu

# Default target: compile and then run
all: $(EXE) run

# Build the executable from the .cu source file
$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(EXE) $(SRC) -lm

# Run the executable and record execution time
run: $(EXE)
	/usr/bin/time -v ./$(EXE) 2>&1 | tee execution_time.txt

# Run nvprof to generate GPU trace profiling information
profiling: $(EXE)
	nvprof ./$(EXE) 2>&1 | tee profiling_output.txt
# Clean generated executable and report files
clean:
	rm -f $(EXE) nvprof_report.txt execution_time.txt
