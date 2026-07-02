#include <stdio.h>
#include <stdlib.h>
#include "genann.h"

int main(void) {
    srand(42);

    struct { const char *name; int inputs; int hidden; } configs[] = {
        {"small",  64, 16},
        {"medium", 128, 32},
        {"large",  256, 64},
    };
    int n = sizeof(configs) / sizeof(configs[0]);

    for (int i = 0; i < n; i++) {
        genann *ann = genann_init(configs[i].inputs, 4,
                                  configs[i].hidden, configs[i].inputs);
        char path[256];
        snprintf(path, sizeof(path), "model/ae_%s_f64.genann", configs[i].name);
        FILE *fp = fopen(path, "w");
        if (!fp) { perror(path); return 1; }
        genann_write(ann, fp);
        fclose(fp);
        printf("Generated %s\n", path);
        genann_free(ann);
    }
    return 0;
}
