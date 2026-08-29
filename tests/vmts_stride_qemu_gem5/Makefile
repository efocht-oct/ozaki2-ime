RV_TOOLCHAIN_WRAPPER ?= $(HOME)/Projects/rv-toolchain-wrapper
RV_CONTAINER_IMAGE ?= $(HOME)/Projects/IME/SDK/SIF/sdk-2026.06.29.sif
export RV_CONTAINER_IMAGE
export PATH := $(RV_TOOLCHAIN_WRAPPER)/bin:$(PATH)

CC = riscv64-openchip-linux-gnu-gcc
CFLAGS = -O3 -g -static -std=gnu11 -march=rv64gcv_zbb_zvvmm_zvvmtls_zvvmttls -mtune=generic -mabi=lp64d -Wall -Wextra
TARGET = test_vmts_stride.elf

.PHONY: all clean qemu
all: $(TARGET)

$(TARGET): test_vmts_stride.c
	$(CC) $(CFLAGS) $< -o $@

qemu: $(TARGET)
	qemu-riscv64 -cpu rv64,v=true,vlen=16384,x-zvvm=true ./$(TARGET)

clean:
	rm -f $(TARGET)
