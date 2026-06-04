# Makefile for Ozaki-2 IME Project
# Uses rv-toolchain-wrapper for containerized RISC-V cross-compilation and QEMU emulation

# Toolchain configuration
RV_WRAPPER_BIN ?= $(HOME)/Projects/IME/SDK/rv-toolchain-wrapper/bin
RV_CONTAINER_IMAGE ?= ghcr.io/efocht-oct/openchip/sdk:2026.05.28

# Export for rv-wrapper.py
export RV_CONTAINER_IMAGE
export PATH := $(RV_WRAPPER_BIN):$(PATH)

# Compiler and tools
CC = riscv64-openchip-linux-gnu-gcc
QEMU = qemu-riscv64

# Architecture flags for IME (Zvvm)
# Enable V extension, Zbb (bitmanip for runtime lambda), and Zvvmm (IME MAC), Zvvmtls/Zvvmttls (IME load/store)
MARCH = rv64gcv_zbb_zvvmm_zvvmtls_zvvmttls
MTUNE = generic
# By default, we compile with -DMOCK_IME so that tests can be fully run and verified end-to-end 
# under QEMU emulation without hanging on unimplemented/unstable matrix instruction emulation.
# To compile the actual, pure hardware Zvvm instruction kernel, remove -DMOCK_IME.
CFLAGS = -O3 -march=$(MARCH) -mtune=$(MTUNE) -mabi=lp64d -Wall -Wextra -std=c11 -I$(SRC_DIR) -DMOCK_IME
LDFLAGS = -static -lm

# Directories
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

# Source files
SRCS = $(SRC_DIR)/ozaki_common.c $(SRC_DIR)/dgemm.c $(SRC_DIR)/sgemm.c
TEST_SRCS = $(TEST_DIR)/test_ozaki.c

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/test_%.o,$(TEST_SRCS))

# Executables
TARGET = $(BUILD_DIR)/test_ozaki

.PHONY: all clean test

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(TEST_OBJS) -o $@ $(LDFLAGS)

test: $(TARGET)
	@echo "Running tests with default QEMU configuration..."
	$(QEMU) $(TARGET)
	@echo "Running tests with VLEN=16384 and lambda=8..."
	$(QEMU) -cpu max,vlen=16384 $(TARGET)

clean:
	rm -rf $(BUILD_DIR)
