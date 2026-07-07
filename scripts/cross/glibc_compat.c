#undef _GNU_SOURCE
#undef _ISOC23_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#pragma GCC visibility push(hidden)

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}

long long __isoc23_strtoll(const char *nptr, char **endptr, int base) {
    return strtoll(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) {
    return strtoul(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base) {
    return strtoull(nptr, endptr, base);
}

double __isoc23_strtod(const char *nptr, char **endptr) {
    return strtod(nptr, endptr);
}

float __isoc23_strtof(const char *nptr, char **endptr) {
    return strtof(nptr, endptr);
}

long double __isoc23_strtold(const char *nptr, char **endptr) {
    return strtold(nptr, endptr);
}

long __isoc23_strtol_l(const char *nptr, char **endptr, int base, void *loc) {
    (void)loc;
    return strtol(nptr, endptr, base);
}

long long __isoc23_strtoll_l(const char *nptr, char **endptr, int base, void *loc) {
    (void)loc;
    return strtoll(nptr, endptr, base);
}

unsigned long __isoc23_strtoul_l(const char *nptr, char **endptr, int base, void *loc) {
    (void)loc;
    return strtoul(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull_l(const char *nptr, char **endptr, int base, void *loc) {
    (void)loc;
    return strtoull(nptr, endptr, base);
}

double __isoc23_strtod_l(const char *nptr, char **endptr, void *loc) {
    (void)loc;
    return strtod(nptr, endptr);
}

float __isoc23_strtof_l(const char *nptr, char **endptr, void *loc) {
    (void)loc;
    return strtof(nptr, endptr);
}

long double __isoc23_strtold_l(const char *nptr, char **endptr, void *loc) {
    (void)loc;
    return strtold(nptr, endptr);
}

int __isoc23_sscanf(const char *str, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vsscanf(str, fmt, ap);
    va_end(ap);
    return r;
}

int __isoc23_fscanf(FILE *stream, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vfscanf(stream, fmt, ap);
    va_end(ap);
    return r;
}

int __isoc23_scanf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vscanf(fmt, ap);
    va_end(ap);
    return r;
}

#pragma GCC visibility pop
