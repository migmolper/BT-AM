CC = g++
CFLAGS = -std=c++11 -Ofast

all: nist2solera

nist2solera: nist2solera.cpp
	$(CC) $(CFLAGS) -o nist2solera nist2solera.cpp

install: nist2solera
	cp nist2solera /usr/local/bin/

clean:
	rm -f nist2solera