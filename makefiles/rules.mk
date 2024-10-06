default: all

$(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_APP_BUILD_DIR):
	mkdir -p $@

# Compile and generate Key Value Store object files
$(BUILD_DIR)/%.o: $(KVSTORE_SRC_DIR)/%.cu | $(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(KVSTORE_SRC_DIR)/%.cpp | $(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR)
	$(CXX) $(CFLAGS) -c $< -o $@

# Compile and generate KV API App object files
$(BUILD_DIR)/%.o: $(KV_API_APP_DIR)/%.cu | $(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_APP_BUILD_DIR)
	$(CC) $(CFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(KV_API_APP_DIR)/%.cpp | $(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_APP_BUILD_DIR)
	$(CXX) $(CFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@