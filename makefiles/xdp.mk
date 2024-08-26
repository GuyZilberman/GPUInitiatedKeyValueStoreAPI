# Paths
XDP_ON_HOST_HEADER_PATH = /etc/opt/pliops/xdp-onhost/store_lib_expo.h
XDP_HEADER_PATH = /etc/pliops/store_lib_expo.h

# Determine which header is available
ifneq ($(wildcard $(XDP_ON_HOST_HEADER_PATH)),)
    XDP_ON_HOST = 1
else ifneq ($(wildcard $(XDP_HEADER_PATH)),)
    XDP = 1
endif

# Set PLIOPS_PATH based on the flags
ifdef XDP_ON_HOST
    PLIOPS_PATH = /opt/pliops/xdp-onhost/lib
else ifdef XDP
    PLIOPS_PATH = /etc/pliops
endif

# Check if either XDP_ON_HOST or XDP is defined
ifeq ($(XDP_ON_HOST)$(XDP),1)
    PLIOPS_CFLAGS = -I${PLIOPS_PATH}
    PLIOPS_LFLAGS = -L${PLIOPS_PATH} -lstorelib
endif