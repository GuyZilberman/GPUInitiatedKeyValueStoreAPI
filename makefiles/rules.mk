default: all

$(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_APP_BUILD_DIR):
	mkdir -p $@

# Compile and generate object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_APP_BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_APP_BUILD_DIR)
	$(CXX) $(CFLAGS) -c $< -o $@
