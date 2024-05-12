CC = mpicc
CFLAGS = -Wall -Wextra -std=c99
LIBS = -lm
SRC = 112525005_hw2.c
OBJ = $(SRC:.c=.o)
EXECUTABLE = 112525005_hw2

all: $(EXECUTABLE)

debug: CFLAGS += -D DEBUG
debug: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(EXECUTABLE)