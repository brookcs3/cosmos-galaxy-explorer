CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wno-deprecated -Wno-unused-function
INCLUDES = -I/opt/homebrew/include -Ilib
LDFLAGS = -L/opt/homebrew/lib
LIBS = -lglfw -lGLEW -framework OpenGL -framework CoreAudio -framework AudioToolbox -framework CoreFoundation

TARGET = cosmos

# ImGui sources
IMGUI_SRC = lib/imgui.cpp lib/imgui_draw.cpp lib/imgui_tables.cpp lib/imgui_widgets.cpp \
            lib/imgui_impl_glfw.cpp lib/imgui_impl_opengl3.cpp

IMGUI_OBJ = $(IMGUI_SRC:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

# Compile ImGui object files
lib/%.o: lib/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): sample.cpp $(IMGUI_OBJ) lib/json.hpp lib/miniaudio.h shaders/*.vert shaders/*.frag
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) sample.cpp $(IMGUI_OBJ) $(LIBS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) lib/*.o
