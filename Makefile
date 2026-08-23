# Makefile for Ozaki-2 IME Project
# Uses rv-toolchain-wrapper for containerized RISC-V cross-compilation and QEMU emulation

# Toolchain configuration
RV_WRAPPER_BIN ?= $(HOME)/Projects/IME/SDK/rv-toolchain-wrapper/bin
RV_CONTAINER_IMAGE ?= ghcr.io/efocht-oct/openchip/sdk:2026.05.28

# Test configuration (can be overridden via environment variables)
VLEN ?= 16384
LAMBDA ?= 8
MOCK_IME ?= 1

# Export for rv-wrapper.py
export RV_CONTAINER_IMAGE
export PATH := $(RV_WRAPPER_BIN):$(PATH)

# Compiler and tools
CC = riscv64-openchip-linux-gnu-gcc
QEMU = qemu-riscv64
HOSTCC ?= cc

# Architecture flags for IME (Zvvm)
# Enable V extension, Zbb (bitmanip for runtime lambda), and Zvvmm (IME MAC), Zvvmtls/Zvvmttls (IME load/store)
MARCH = rv64gcv_zbb_zvvmm_zvvmtls_zvvmttls
MTUNE = generic
# By default, compile with MOCK_IME=1 so tests can run end-to-end under QEMU
# without depending on real matrix instruction emulation. Use MOCK_IME=0 to
# compile the actual Zvvm instruction kernel.
CFLAGS = -O3 -march=$(MARCH) -mtune=$(MTUNE) -mabi=lp64d -Wall -Wextra -std=c11 -I$(SRC_DIR)
ifeq ($(MOCK_IME),1)
CFLAGS += -DMOCK_IME
endif
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

.PHONY: all clean test test-moduli test-vlen test-lambda

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
	@echo "Running tests with VLEN=$(VLEN), lambda=$(LAMBDA), MOCK_IME=$(MOCK_IME)..."
	$(QEMU) $(QEMU_CPU) $(TARGET) $(LAMBDA)

test-moduli: tests/test_moduli.c src/ozaki_common.c src/ozaki_common.h | $(BUILD_DIR)
	$(HOSTCC) -std=c11 -Wall -Wextra -I$(SRC_DIR) tests/test_moduli.c src/ozaki_common.c -o $(BUILD_DIR)/test_moduli
	$(BUILD_DIR)/test_moduli

test-vlen: $(TARGET)
	@echo "Running tests with VLEN=$(VLEN), lambda=$(LAMBDA), MOCK_IME=$(MOCK_IME)..."
	$(QEMU) $(QEMU_CPU) $(TARGET) $(LAMBDA)

# Run test with specific VLEN (e.g., make test-vlen VLEN=512 LAMBDA=2 MOCK_IME=0)
# Run test with specific lambda (e.g., make test LAMBDA=4 MOCK_IME=1)

clean:
	rm -rf $(BUILD_DIR)
