# ------------------------------------------------------------
# Build + run with:
#   make
#   make run
#   make inspect
# ------------------------------------------------------------

CXX      := g++
OUTPUT   := cnn_compute
INSPECT  := inspect_dataset

OS       := $(shell uname)
SRC_DIR  := src
TOOLS_DIR:= tools
BIN_DIR  := bin

# ------------------------------------------------------------
# Common flags
# ------------------------------------------------------------
CXX_FLAGS := -O3 -std=c++23 -Wall -Wextra -Wno-unused-result
INCLUDES  := -I$(SRC_DIR)
INCLUDES += -Iexternal/glad/include



# ------------------------------------------------------------
# Linux
# ------------------------------------------------------------
ifeq ($(OS), Linux)
    LDFLAGS := -L/usr/local/lib -lsfml-window -lsfml-system -lGL
endif

# ------------------------------------------------------------
# macOS (placeholder for later)
# ------------------------------------------------------------
ifeq ($(OS), Darwin)
    SFML_DIR := /opt/homebrew/Cellar/sfml/3.0.1
    INCLUDES += -I$(SFML_DIR)/include
    LDFLAGS  := -L$(SFML_DIR)/lib \
                -lsfml-window -lsfml-system \
                -framework OpenGL
endif

# ------------------------------------------------------------
# Sources
# ------------------------------------------------------------
MAIN_SRC     := $(SRC_DIR)/main.cpp
INSPECT_SRC  := $(TOOLS_DIR)/inspect_dataset.cpp
GLAD_SRC := external/glad/src/gl.c

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------
all: $(BIN_DIR)/$(OUTPUT) $(BIN_DIR)/$(INSPECT)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Main program
$(BIN_DIR)/$(OUTPUT): $(MAIN_SRC) $(GLAD_SRC) | $(BIN_DIR) copy_shaders
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

# Dataset inspection tool
$(BIN_DIR)/$(INSPECT): $(INSPECT_SRC) | $(BIN_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $< -o $@

# ------------------------------------------------------------
# Shaders
# ------------------------------------------------------------
SHADER_SRC := shaders
SHADER_DST := bin/shaders

$(SHADER_DST):
	mkdir -p $(SHADER_DST)

copy_shaders: | $(SHADER_DST)
	cp $(SHADER_SRC)/*.comp $(SHADER_DST) 2>/dev/null || true

# ------------------------------------------------------------
# Convenience targets
# ------------------------------------------------------------
run: $(BIN_DIR)/$(OUTPUT)
	cd $(BIN_DIR) && ./$(OUTPUT)

inspect: $(BIN_DIR)/$(INSPECT)
	cd $(BIN_DIR) && ./$(INSPECT) ../data/train.bin

clean:
	rm -f $(BIN_DIR)/$(OUTPUT) $(BIN_DIR)/$(INSPECT)

.PHONY: all clean run inspect copy_shaders