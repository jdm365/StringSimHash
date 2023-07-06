#CXX = clang++
#CXXFLAGS = -std=c++11 -stdlib=libc++ -Wall
CXX = g++
CXXFLAGS = -std=c++11 -Wall
CXXFLAGS += -mtune=native -Ofast -fopenmp -mavx2
PYFLAGS = `python3 -m pybind11 --includes`
PYFILES = -o StringDedup.so -shared -fPIC
LDLIBS =  -lgomp -lpython3.10
SRCDIR = src
INCLUDES = -Iinclude
BINDIR = bin/release
LIBDIR = lib/StringDedup
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
TARGET = main


all:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYFLAGS) $(SOURCES) $(LDLIBS) -o $(TARGET)
	mv $(TARGET) $(BINDIR)

clean:
	rm -f $(OBJECTS) $(TARGET)

run:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYFLAGS) $(SOURCES) $(LDLIBS) -o $(TARGET)
	mv $(TARGET) $(BINDIR)
	./$(BINDIR)/$(TARGET)


python:
	$(CXX) $(CXXFLAGS) $(SOURCES) $(INCLUDES) $(PYFLAGS) $(PYFILES) $(LDLIBS)
	mv StringDedup.so $(LIBDIR)
	cd lib && pip install .
