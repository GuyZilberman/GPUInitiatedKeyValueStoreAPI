# Paths
XDP_ON_HOST_HEADER_PATH = /etc/opt/pliops/xdp-onhost/store_lib_expo.h
XDP_HEADER_PATH = /etc/pliops/store_lib_expo.h

# Determine which header is available
ifneq ($(wildcard $(XDP_ON_HOST_HEADER_PATH)),)
    XDP_ON_HOST = 1
else ifneq ($(wildcard $(XDP_HEADER_PATH)),)
    XDP = 1
endif

# PLIOPS paths and flags
ifdef XDP_ON_HOST
    PLIOPS_PATH = /opt/pliops/xdp-onhost/lib
    PLIOPS_CFLAGS = -I${PLIOPS_PATH}
    PLIOPS_LFLAGS = -L${PLIOPS_PATH} -lstorelib
else ifdef XDP
    PLIOPS_PATH = /etc/pliops
    PLIOPS_CFLAGS = -I${PLIOPS_PATH}
    PLIOPS_LFLAGS = -L${PLIOPS_PATH} -lstorelib
endif

# Compiler and linker configuration
CC = nvcc
CXX = nvcc
CFLAGS = -arch=sm_80 -I. ${PLIOPS_CFLAGS} -I./src/common -I./src/key_value_store
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

# Source and object files
KV_LIB_SRCS = src/key_value_store/key_value_store.cu src/common/gdrcopy_common.cpp
KV_LIB_OBJS = $(KV_LIB_SRCS:.cu=.o)
KV_LIB_OBJS := $(KV_LIB_OBJS:.cpp=.o)
BENCHMARK_SRCS = src/kv_api_app/KV_API_App.cu
BENCHMARK_OBJS = $(BENCHMARK_SRCS:.cu=.o)

# Target rules
.PHONY: all clean

all: kvstore kvapp

debug: CFLAGS += $(DEBUG_FLAGS)
debug: all

# Specific rule for key_value_store.cu
src/key_value_store/key_value_store.o: src/key_value_store/key_value_store.cu
	$(CC) $(CFLAGS) -c $< -o $@

libkey_value_store.a: $(KV_LIB_OBJS)
	ar rcs $@ $(KV_LIB_OBJS)

kvstore: libkey_value_store.a
	rm -f $(KV_LIB_OBJS)

kvapp: $(BENCHMARK_OBJS) libkey_value_store.a
	$(CC) $(CFLAGS) $(BENCHMARK_OBJS) -o $@ $(LFLAGS) $(LIBS)
	rm -f $(BENCHMARK_OBJS)

# Generic rules for other .cu and .cpp files
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.a kvapp
