# ------------------------------------------------------------
# Build + run with:
#   make
#   make run
# ------------------------------------------------------------

CXX     := g++
OUTPUT  := cnn_compute
OS      := $(shell uname)
SRC_DIR := ./src
BIN_DIR := ./bin

# ------------------------------------------------------------
# Common flags
# ------------------------------------------------------------
CXX_FLAGS := -O3 -std=c++23 -Wall -Wextra -Wno-unused-result
INCLUDES  := -I$(SRC_DIR)

# ------------------------------------------------------------
# Linux
# ------------------------------------------------------------
ifeq ($(OS), Linux)
    LDFLAGS := -L/usr/local/lib -lsfml-window -lsfml-system -lGL
endif

# ------------------------------------------------------------
# macOS
# ------------------------------------------------------------
ifeq ($(OS), Darwin)
    SFML_DIR := /opt/homebrew/Cellar/sfml/3.0.1
    INCLUDES += -I$(SFML_DIR)/include
    LDFLAGS  := -L$(SFML_DIR)/lib \
                -Wl,-rpath,$(SFML_DIR)/lib \
                -lsfml-window -lsfml-system \
                -framework OpenGL
endif

# ------------------------------------------------------------
# Sources
# ------------------------------------------------------------
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(SRC_FILES:.cpp=.o)

DEP_FILES := $(OBJ_FILES:.o=.d)
-include $(DEP_FILES)

# ------------------------------------------------------------
# Shaders
# ------------------------------------------------------------
SHADER_SRC := ./shaders
SHADER_DST := ./bin/shaders

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------
all: $(BIN_DIR)/$(OUTPUT)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(SHADER_DST):
	mkdir -p $(SHADER_DST)

copy_shaders: | $(SHADER_DST)
	cp -f $(SHADER_SRC)/*.comp $(SHADER_DST) 2>/dev/null || true

$(BIN_DIR)/$(OUTPUT): $(OBJ_FILES) copy_shaders | $(BIN_DIR)
	$(CXX) $(OBJ_FILES) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) -MMD -MP -c $(CXX_FLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(OBJ_FILES) $(DEP_FILES) $(BIN_DIR)/$(OUTPUT)
	rm -rf $(SHADER_DST)

run: $(BIN_DIR)/$(OUTPUT)
	cd $(BIN_DIR) && ./$(OUTPUT)

.PHONY: all clean run copy_shaders