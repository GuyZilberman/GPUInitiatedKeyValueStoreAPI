# Compiler and linker configuration
CC = nvcc
CXX = nvcc
CFLAGS = -arch=sm_80 -I. ${PLIOPS_CFLAGS} -I${COMMON_DIR} -I${KV_STORE_DIR}
LFLAGS = -L. ${PLIOPS_LFLAGS} -lrt -lpthread -lcuda -lgdrapi -ltbb 
KV_APP_LIBS = -lkey_value_store
DEBUG_FLAGS = -G -g
CONFIG_YAML_PATH := $(CURDIR)/cfg/config.yaml
CFLAGS += -DCONFIG_YAML_PATH=\"$(CONFIG_YAML_PATH)\"

# Conditional flags
ifdef DEBUG
    CFLAGS += $(DEBUG_FLAGS)
else
    CFLAGS += -O3  # Apply optimization when not in debug mode
endif

ifdef CHECK_WRONG_ANSWERS
    CFLAGS += -DCHECK_WRONG_ANSWERS
endif

ifdef STORELIB_LOOPBACK
    $(info Using storelib loopback)
    CFLAGS += -DSTORELIB_LOOPBACK
endif

ifdef IN_MEMORY_STORE
    $(info Using in-memory store)
    MAX_VALUE_SIZE := $(shell yq '.COMPILE_TIME.KV_STORE.IN_MEMORY_STORE.MAX_VALUE_SIZE' cfg/config.yaml)
    MAX_KEY_SIZE := $(shell yq '.COMPILE_TIME.KV_STORE.IN_MEMORY_STORE.MAX_KEY_SIZE' cfg/config.yaml)
    CFLAGS += -DIN_MEMORY_STORE -DMAX_VALUE_SIZE=$(MAX_VALUE_SIZE) -DMAX_KEY_SIZE=$(MAX_KEY_SIZE)
endif

ifdef XDP_ON_HOST
    $(info Using XDP on host)
    CFLAGS += -DXDP_ON_HOST
endif

ifdef VALUE_SIZE
    CFLAGS += -DVALUE_SIZE=$(VALUE_SIZE)
endif

ifdef NUM_KEYS
    CFLAGS += -DNUM_KEYS=$(NUM_KEYS)
endif
