default: all

$(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_BUILD_DIR):
	mkdir -p $@

# Compile and generate dependency files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(COMMON_BUILD_DIR) $(KV_STORE_BUILD_DIR) $(KV_API_BUILD_DIR)
	$(CXX) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)



