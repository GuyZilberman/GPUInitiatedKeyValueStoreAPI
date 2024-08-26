# Build directories
BUILD_DIR = build
COMMON_BUILD_DIR = $(BUILD_DIR)/common
KV_STORE_BUILD_DIR = $(BUILD_DIR)/key_value_store
KV_API_BUILD_DIR = $(BUILD_DIR)/kv_api_app
SRC_DIR = src
COMMON_DIR = $(SRC_DIR)/common
KV_STORE_DIR = $(SRC_DIR)/key_value_store
KV_API_DIR = $(SRC_DIR)/kv_api_app

# Source and object files
KV_LIB_SRCS = $(wildcard $(KV_STORE_DIR)/*.cu) $(wildcard $(COMMON_DIR)/*.cpp)
KV_LIB_OBJS = $(KV_LIB_SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
KV_LIB_OBJS := $(KV_LIB_OBJS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
BENCHMARK_SRCS = $(wildcard $(KV_API_DIR)/*.cu)
BENCHMARK_OBJS = $(BENCHMARK_SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Generate dependency files
DEPS = $(KV_LIB_OBJS:.o=.d) $(BENCHMARK_OBJS:.o=.d)

# Set the parent directory
PARENT_DIR := "$(BUILD_DIR)/.."

# Set the include and library paths for yaml-cpp
YAML_CPP_INC := -I$(PARENT_DIR)/include
YAML_CPP_LIB := $(PARENT_DIR)/libs/libyaml-cpp.a