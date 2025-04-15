CC = gcc
CFLAGS = -Wall -O2 -pg
LDFLAGS = -pg

EXE = nn.exe
SRC = nn.c

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm $(LDFLAGS)

run: $(EXE)
	/usr/bin/time -v ./$(EXE) 2>&1 | tee execution_time.txt

profiling: run
	gprof $(EXE) gmon.out > report.txt
	gprof -b -q $(EXE) | gprof2dot | dot -Tsvg -o gprof_callgraph.svg

clean:
	rm -f $(EXE) gmon.out report.txt gprof_callgraph.svg execution_time.txt
