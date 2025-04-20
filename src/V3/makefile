NVCC = nvcc
NVCCFLAGS = -O2

EXE = v3.exe
SRC = v3.cu

all: $(EXE) run profile

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	nvprof ./$(EXE)

clean:
	rm -f $(EXE)
