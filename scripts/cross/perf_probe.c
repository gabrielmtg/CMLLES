/* Standalone diagnostic: reports the real errno from perf_event_open() for
 * each hardware event used by the CMLLES benchmarks, since benchmark_*.c
 * silently drops perf_event_open() failures (negative fd -> writes 0, no
 * error printed). Also probes a PERF_TYPE_SOFTWARE baseline, which should
 * succeed regardless of whether a hardware PMU is registered, to separate
 * "perf subsystem unavailable" from "no hardware PMU backend registered".
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>

static long perf_event_open_syscall(struct perf_event_attr *hw, pid_t pid,
                                     int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw, pid, cpu, group_fd, flags);
}

static void try_event(const char *label, uint32_t type, uint64_t config) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type       = type;
    attr.size       = sizeof(attr);
    attr.config     = config;
    attr.disabled   = 1;

    errno = 0;
    long fd = perf_event_open_syscall(&attr, 0, -1, -1, 0);
    if (fd < 0) {
        printf("%-28s FAIL fd=%ld errno=%d (%s)\n", label, fd, errno, strerror(errno));
    } else {
        uint64_t before = 0, after = 0;
        ioctl((int)fd, PERF_EVENT_IOC_RESET, 0);
        ioctl((int)fd, PERF_EVENT_IOC_ENABLE, 0);
        for (volatile long i = 0; i < 50000000L; i++) {}
        ioctl((int)fd, PERF_EVENT_IOC_DISABLE, 0);
        ssize_t n = read((int)fd, &after, sizeof(after));
        printf("%-28s OK   fd=%ld read_bytes=%zd value=%lu\n",
               label, fd, n, (unsigned long)after);
        (void)before;
        close((int)fd);
    }
}

/* Mirrors benchmark_*.c's perf_open_all()/perf_reset_enable()/perf_disable_read():
 * opens all N events *simultaneously* (competing for the SoC's 4 hardware
 * counter slots at once), then enables/measures/disables/reads them all
 * together, unlike try_event() above which tests one event fully in
 * isolation. use_raw_l1d selects PERF_TYPE_RAW/0x202 (SiFive
 * DCACHE_MISS_MMIO_ACCESSES) for the L1D slot instead of the generic
 * PERF_TYPE_HW_CACHE mapping that returns 0 on this platform. */
static void try_simultaneous(const char *label_prefix, int use_raw_l1d) {
    const char *names[5] = {
        "cycles", "instructions", "branch_misses", "l1d_access", "l1d_miss"
    };
    struct perf_event_attr attrs[5];
    int fds[5];
    memset(attrs, 0, sizeof(attrs));

    attrs[0].type = PERF_TYPE_HARDWARE; attrs[0].config = PERF_COUNT_HW_CPU_CYCLES;
    attrs[1].type = PERF_TYPE_HARDWARE; attrs[1].config = PERF_COUNT_HW_INSTRUCTIONS;
    attrs[2].type = PERF_TYPE_HARDWARE; attrs[2].config = PERF_COUNT_HW_BRANCH_MISSES;

    if (use_raw_l1d) {
        attrs[3].type = PERF_TYPE_RAW; attrs[3].config = 0x202; /* DCACHE_MISS_MMIO_ACCESSES, used here as a stand-in for "access" too */
        attrs[4].type = PERF_TYPE_RAW; attrs[4].config = 0x202; /* DCACHE_MISS_MMIO_ACCESSES */
    } else {
        attrs[3].type = PERF_TYPE_HW_CACHE;
        attrs[3].config = PERF_COUNT_HW_CACHE_L1D
                         | ((uint64_t)PERF_COUNT_HW_CACHE_OP_READ << 8)
                         | ((uint64_t)PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
        attrs[4].type = PERF_TYPE_HW_CACHE;
        attrs[4].config = PERF_COUNT_HW_CACHE_L1D
                         | ((uint64_t)PERF_COUNT_HW_CACHE_OP_READ << 8)
                         | ((uint64_t)PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    }

    printf("--- simultaneous (%s) ---\n", label_prefix);

    for (int i = 0; i < 5; i++) {
        attrs[i].size = sizeof(struct perf_event_attr);
        attrs[i].disabled = 1;
        errno = 0;
        fds[i] = (int)perf_event_open_syscall(&attrs[i], 0, -1, -1, 0);
        if (fds[i] < 0) {
            printf("  %-16s OPEN FAIL fd=%d errno=%d (%s)\n",
                   names[i], fds[i], errno, strerror(errno));
        }
    }

    for (int i = 0; i < 5; i++) {
        if (fds[i] >= 0) {
            ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
            ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    for (volatile long i = 0; i < 50000000L; i++) {}
    for (int i = 0; i < 5; i++) {
        if (fds[i] >= 0) ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
    }

    for (int i = 0; i < 5; i++) {
        if (fds[i] < 0) continue;
        uint64_t val = 0;
        ssize_t n = read(fds[i], &val, sizeof(val));
        printf("  %-16s OK   read_bytes=%zd value=%lu\n", names[i], n, (unsigned long)val);
        close(fds[i]);
    }
}

int main(void) {
    FILE *f = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
    if (f) {
        char buf[32] = {0};
        fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        printf("perf_event_paranoid = %s", buf);
    } else {
        printf("perf_event_paranoid = <could not read: %s>\n", strerror(errno));
    }

    printf("--- baseline (software event, should always work) ---\n");
    try_event("SW_TASK_CLOCK", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK);
    try_event("SW_CPU_CLOCK",  PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK);

    printf("--- hardware events used by the benchmarks ---\n");
    try_event("HW_CPU_CYCLES",    PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    try_event("HW_INSTRUCTIONS",  PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    try_event("HW_BRANCH_MISSES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);

    uint64_t l1_loads  = PERF_COUNT_HW_CACHE_L1D
                       | ((uint64_t)PERF_COUNT_HW_CACHE_OP_READ << 8)
                       | ((uint64_t)PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    uint64_t l1_misses = PERF_COUNT_HW_CACHE_L1D
                       | ((uint64_t)PERF_COUNT_HW_CACHE_OP_READ << 8)
                       | ((uint64_t)PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    try_event("HW_CACHE_L1D_READ_ACCESS", PERF_TYPE_HW_CACHE, l1_loads);
    try_event("HW_CACHE_L1D_READ_MISS",   PERF_TYPE_HW_CACHE, l1_misses);

    printf("--- SiFive U74 raw event (isolated, no competition) ---\n");
    try_event("RAW_DCACHE_MISS_MMIO_0x202", PERF_TYPE_RAW, 0x202);
    try_event("RAW_INTEGER_LOAD_0x200",     PERF_TYPE_RAW, 0x200);
    try_event("RAW_FP_LOAD_0x80000",        PERF_TYPE_RAW, 0x80000);
    try_event("RAW_INT_OR_FP_LOAD_0x80200", PERF_TYPE_RAW, 0x80200);

    try_simultaneous("generic HW_CACHE for L1D, like perf_open_all()", 0);
    try_simultaneous("PERF_TYPE_RAW 0x202 for L1D slot", 1);

    return 0;
}
