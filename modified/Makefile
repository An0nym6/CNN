NVCC = nvcc
SRC = $(wildcard *.cu)
CFLAGS = -I. -std=c++11 -w

fashion: $(SRC)
	$(NVCC) -o $@ $^ $(CFLAGS)

clean:
	rm fashion
