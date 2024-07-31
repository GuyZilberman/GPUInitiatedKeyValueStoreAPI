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