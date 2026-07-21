/* Provides __riscv_hwprobe() for targets whose glibc predates it (see
 * riscv_hwprobe_compat_include/sys/hwprobe.h). Implemented as a direct
 * syscall - the riscv_hwprobe(2) syscall (number 258, added in Linux 6.4)
 * is a stable kernel UAPI regardless of glibc version.
 */
#include <sys/syscall.h>
#include <unistd.h>
#include <stddef.h>

#ifndef __NR_riscv_hwprobe
#define __NR_riscv_hwprobe 258
#endif

struct riscv_hwprobe {
    long long key;
    unsigned long long value;
};

int __riscv_hwprobe(struct riscv_hwprobe *pairs, size_t pair_count,
                     size_t cpu_count, unsigned long *cpus,
                     unsigned int flags)
{
    return (int)syscall(__NR_riscv_hwprobe, pairs, pair_count, cpu_count, cpus, flags);
}
