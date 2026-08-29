RV_TOOLCHAIN_WRAPPER ?= $(HOME)/Projects/rv-toolchain-wrapper
RV_CONTAINER_IMAGE ?= $(HOME)/Projects/IME/SDK/SIF/sdk-2026.06.29.sif

export RV_CONTAINER_IMAGE
export PATH := $(RV_TOOLCHAIN_WRAPPER)/bin:$(PATH)

CC ?= riscv64-openchip-linux-gnu-gcc
HOSTCC ?= cc
CFLAGS = -O3 -march=rv64gcv_zbb_zvvmm_zvvmtls_zvvmttls -mtune=generic -mabi=lp64d -Wall -Wextra -std=gnu11
LDFLAGS = -static -lm

CASE = u54_case.bin
HOST_GENERATOR = generate_u54_case
RISCV_RUNNER = run_u54_ime.elf

.PHONY: all generate run clean

all: $(HOST_GENERATOR) $(RISCV_RUNNER) $(CASE)

$(HOST_GENERATOR): generate_u54_case.c
	$(HOSTCC) -O2 -Wall -Wextra -std=c11 $< -o $@

$(RISCV_RUNNER): run_u54_ime.c
	riscv64-openchip-linux-gnu-gcc $(CFLAGS) $< -o $@ $(LDFLAGS)

$(CASE): $(HOST_GENERATOR)
	./$(HOST_GENERATOR) $@ 0

generate: $(CASE)

run: all
	GEM5_IME_LAMBDA_MODE=force \
	GEM5_IME_LAMBDA_SEW8=32 \
	GEM5_IME_LAMBDA_SEW16=16 \
	GEM5_IME_LAMBDA_SEW32=8 \
	qemu-riscv64 $(RISCV_RUNNER) $(CURDIR)/$(CASE)

clean:
	rm -f $(HOST_GENERATOR) $(RISCV_RUNNER) $(CASE)
