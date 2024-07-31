# Compiler and linker configuration
CC = nvcc
CXX = nvcc
CFLAGS = -arch=sm_80 -I. ${PLIOPS_CFLAGS} -I${COMMON_DIR} -I${KV_STORE_DIR}
LFLAGS = -L. ${PLIOPS_LFLAGS} -lrt -lpthread -lcuda -lgdrapi -ltbb 
LIBS = -lkey_value_store
DEBUG_FLAGS = -G -g

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
    CFLAGS += -DIN_MEMORY_STORE
endif

ifdef XDP_ON_HOST
    $(info Using XDP on host)
    CFLAGS += -DXDP_ON_HOST
endif

ifdef VALUE_SIZE
    CFLAGS += -DVALUE_SIZE=$(VALUE_SIZE)
endif

ifdef QUEUE_SIZE
    CFLAGS += -DQUEUE_SIZE=$(QUEUE_SIZE)
endif

ifdef NUM_KEYS
    CFLAGS += -DNUM_KEYS=$(NUM_KEYS)
endif
