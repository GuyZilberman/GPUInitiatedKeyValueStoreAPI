# Build directories
BUILD_DIR = build
COMMON_BUILD_DIR = $(BUILD_DIR)/common
KV_STORE_BUILD_DIR = $(BUILD_DIR)/key_value_store
KV_API_APP_BUILD_DIR = $(BUILD_DIR)/kv_api_app

# Source directories
SRC_DIR = src
KVSTORE_SRC_DIR = $(SRC_DIR)/lib
COMMON_DIR = $(KVSTORE_SRC_DIR)/common
KV_STORE_DIR = $(KVSTORE_SRC_DIR)/key_value_store
BENCHMARK_SRC_DIR = $(SRC_DIR)/benchmark
KV_API_APP_DIR = $(BENCHMARK_SRC_DIR)/kv_api_app

# Source and object files
KV_LIB_SRCS = $(wildcard $(KV_STORE_DIR)/*.cu) $(wildcard $(COMMON_DIR)/*.cpp)
KV_LIB_OBJS = $(patsubst $(KVSTORE_SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(patsubst $(KVSTORE_SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(KV_LIB_SRCS)))
BENCHMARK_SRCS = $(wildcard $(KV_API_APP_DIR)/*.cu)
BENCHMARK_OBJS = $(BENCHMARK_SRCS:$(KV_API_APP_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Set the parent directory
PARENT_DIR := "$(BUILD_DIR)/.."

# Set the include and library paths for yaml-cpp
YAML_CPP_INC := -I$(PARENT_DIR)/include
YAML_CPP_LIB := $(PARENT_DIR)/libs/libyaml-cpp.a