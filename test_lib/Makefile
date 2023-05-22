CXX = clang++
CXXFLAGS = -std=c++11 -stdlib=libc++ -Wall
CXXFLAGS += -mtune=native -march=native -Ofast -fopenmp
LDLIBS = -lstdc++ -lomp
SRCDIR = src
INCLUDES = -Iinclude
BINDIR = bin/release
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
TARGET = main


all:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(LDLIBS) -o $(TARGET)
	mv $(TARGET) $(BINDIR)

clean:
	rm -f $(OBJECTS) $(TARGET)

run:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(LDLIBS) -o $(TARGET)
	mv $(TARGET) $(BINDIR)
	./$(BINDIR)/$(TARGET)
