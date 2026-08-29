RV_TOOLCHAIN_WRAPPER ?= $(HOME)/Projects/rv-toolchain-wrapper
RV_CONTAINER_IMAGE ?= $(HOME)/Projects/IME/SDK/SIF/sdk-2026.06.29.sif
export RV_CONTAINER_IMAGE
export PATH := $(RV_TOOLCHAIN_WRAPPER)/bin:$(PATH)

CXX = riscv64-openchip-linux-gnu-g++
CXXFLAGS = -O3 -g -static -std=gnu++17 -march=rv64gcv_zbb_zvvmm_zvvmtls_zvvmttls
LDFLAGS = -lm
TARGET = test_tile_ls.elf

.PHONY: all clean qemu
all: $(TARGET)

$(TARGET): test_tile_ls.cxx
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

qemu: $(TARGET)
	qemu-riscv64 -cpu rv64,v=true,vlen=16384,x-zvvm=true ./$(TARGET)

clean:
	rm -f $(TARGET)
