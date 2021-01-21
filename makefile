CC = gcc-mp-9
CFLAGS = -Wall -fPIC -std=c99 -O3
DEPS = munkres.h
OBJ = munkres.o dpc.o
LIBS = -lm

all: libdpc

dpc.o: dpc.c
	$(CC) $(CFLAGS) -c dpc.c

munkres.o: munkres.c munkres.h
	 $(CC) $(CFLAGS) -c munkres.c

munkres: munkres.c
	$(CC) $(CFLAGS) munkres.c -o munkres

libmunkres: munkres.o munkres.h
	$(CC) -shared -o libmunkres.so munkres.o -lm

libdpc: munkres.o munkres.h dpc.o
	$(CC) -shared -o libdpc.so munkres.o dpc.o $(LIBS)

clean_munkres: munkres.o libmunkres.so
	rm -f ./munkres.o ./libmunkres.so

clean_dpc: dpc.o libdpc.so
	rm -f ./dpc.o ./libdpc.so

clean_libs:
	rm -f libmunkres.so libdpc.so munkres.o dpc.o

clean:
	rm -f ./*.o ./a.out ./munkres
