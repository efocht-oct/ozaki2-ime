# Makefile for Ozaki-2 IME Project
# Uses rv-toolchain-wrapper for containerized RISC-V cross-compilation and QEMU emulation

# Toolchain configuration
RV_WRAPPER_BIN ?= $(HOME)/Projects/IME/SDK/rv-toolchain-wrapper/bin
RV_CONTAINER_IMAGE ?= ghcr.io/efocht-oct/openchip/sdk:2026.05.28

# Test configuration (can be overridden via environment variables)
VLEN ?= 16384
LAMBDA ?= 8

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

# QEMU CPU configuration for IME
# VLEN is set via -cpu max,vlen=$(VLEN)
# IME features are enabled via custom feature flags
QEMU_CPU = -cpu rv64,v=true,vlen=$(VLEN),x-zvvm=true,x-zvfofp8min=true,x-zvfbfa=true,x-zvvfmmbf16=true,zvfh=true,zfbfmin=true

.PHONY: all clean test test-vlen test-lambda

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
	@echo "Running tests with VLEN=$(VLEN) and lambda=$(LAMBDA)..."
	$(QEMU) $(QEMU_CPU) $(TARGET) $(LAMBDA)

test-vlen: $(TARGET)
	@echo "Running tests with VLEN=$(VLEN) and lambda=$(LAMBDA)..."
	$(QEMU) $(QEMU_CPU) $(TARGET) $(LAMBDA)

# Run test with specific VLEN (e.g., make test-vlen VLEN=512 LAMBDA=2)
# Run test with specific lambda (e.g., make test LAMBDA=4)

clean:
	rm -rf $(BUILD_DIR)
