# Use nvcc as the compiler.
NVCC = nvcc

# NVCC flags (enable all warnings, optimization level 2).
NVCCFLAGS = -O2

# Executable and source definitions.
EXE = v2.exe
SRC = v2.cu

# Default target: compile the executable, then run it.
all: $(EXE) run

# Build the CUDA executable.
$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(EXE) $(SRC) -lm

# Run the executable and record execution time.
run: $(EXE)
	/usr/bin/time -v ./$(EXE) 2>&1 | tee execution_time.txt

# Run CUDA profiler (nvprof) on the executable and save the output.
profiling: $(EXE)
	nvprof ./$(EXE) 2>&1 | tee profiling_output.txt

# Clean up generated files.
clean:
	rm -f $(EXE) nvprof_report.txt execution_time.txt
