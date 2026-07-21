/* Minimal sys/hwprobe.h compat shim for older glibc targets.
 *
 * The riscv_hwprobe() interface (RISC-V CPU feature probing) was added to
 * glibc around 2.39. Buildroot 2024.02.2's default riscv64 glibc snapshot
 * (2.38) predates it, so this header - plus riscv_hwprobe_compat.c, which
 * provides the __riscv_hwprobe symbol via a raw syscall - lets cpuinfo (a
 * TFLite/XNNPACK dependency) build against our older glibc. Only the
 * identifiers cpuinfo's riscv/linux backend actually references are
 * declared here; see riscv_hwprobe_compat.c for the syscall number.
 */
#ifndef _SYS_HWPROBE_COMPAT_H
#define _SYS_HWPROBE_COMPAT_H 1

#include <stddef.h>

struct riscv_hwprobe {
    long long key;
    unsigned long long value;
};

#define RISCV_HWPROBE_KEY_MVENDORID 0
#define RISCV_HWPROBE_KEY_MARCHID   1
#define RISCV_HWPROBE_KEY_MIMPID    2

#ifdef __cplusplus
extern "C" {
#endif

int __riscv_hwprobe(struct riscv_hwprobe *pairs, size_t pair_count,
                     size_t cpu_count, unsigned long *cpus,
                     unsigned int flags);

#ifdef __cplusplus
}
#endif

#endif /* _SYS_HWPROBE_COMPAT_H */
