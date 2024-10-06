include makefiles/xdp.mk
include makefiles/paths.mk
include makefiles/config.mk
include makefiles/rules.mk

.PHONY: default all clean

all: directories kvstore kvapp

directories: $(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_APP_BUILD_DIR)

libkey_value_store.a: $(KV_LIB_OBJS)
	$(CC) $(CFLAGS) $(YAML_CPP_INC) -c $(KV_LIB_OBJS) 
	ar rcs $@ $(KV_LIB_OBJS) $(YAML_CPP_LIB)

kvstore: libkey_value_store.a
	rm -f $(KV_LIB_OBJS)

kvapp: $(BENCHMARK_OBJS) libkey_value_store.a
	$(CC) $(CFLAGS) $(BENCHMARK_CFLAGS) $(BENCHMARK_OBJS) -o $@ $(LFLAGS) $(BENCHMARK_LFLAGS) $(KV_APP_LIBS)
	rm -f $(BENCHMARK_OBJS)

clean:
	rm -f $(BUILD_DIR)/*.o *.a kvapp
	rm -rf $(BUILD_DIR)
