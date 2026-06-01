#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../../genann/genann.h"

#define NUM_FEATURES    4
#define NUM_CLASSES     3
#define HIDDEN_NEURONS  16
#define HIDDEN_LAYERS   1
#define LEARNING_RATE   0.01
#define EPOCHS          5000
#define MAX_SAMPLES     200
#define DATASET_PATH    "../../../datasets/iris/iris.data"

static const char *class_names[NUM_CLASSES] = {
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
};

static double inputs[MAX_SAMPLES * NUM_FEATURES];
static double targets[MAX_SAMPLES * NUM_CLASSES];
static int num_samples = 0;

static double feat_min[NUM_FEATURES];
static double feat_max[NUM_FEATURES];

static void compute_normalization(void)
{
    int i, j;
    for (j = 0; j < NUM_FEATURES; j++) {
        feat_min[j] = inputs[j];
        feat_max[j] = inputs[j];
    }
    for (i = 1; i < num_samples; i++) {
        for (j = 0; j < NUM_FEATURES; j++) {
            double v = inputs[i * NUM_FEATURES + j];
            if (v < feat_min[j]) feat_min[j] = v;
            if (v > feat_max[j]) feat_max[j] = v;
        }
    }
    for (i = 0; i < num_samples; i++) {
        for (j = 0; j < NUM_FEATURES; j++) {
            double range = feat_max[j] - feat_min[j];
            if (range > 0.0)
                inputs[i * NUM_FEATURES + j] =
                    (inputs[i * NUM_FEATURES + j] - feat_min[j]) / range;
            else
                inputs[i * NUM_FEATURES + j] = 0.0;
        }
    }
}

static int load_dataset(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erro: não foi possível abrir '%s'\n", path);
        return -1;
    }

    char line[1024];
    num_samples = 0;

    while (fgets(line, sizeof(line), f) && num_samples < MAX_SAMPLES) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;

        double *p = inputs + num_samples * NUM_FEATURES;
        double *c = targets + num_samples * NUM_CLASSES;
        c[0] = c[1] = c[2] = 0.0;

        char *tok = strtok(line, ",");
        int j;
        for (j = 0; j < NUM_FEATURES && tok; j++) {
            p[j] = atof(tok);
            tok = strtok(NULL, ",");
        }

        if (!tok) continue;

        tok[strcspn(tok, "\r\n")] = '\0';

        if (strcmp(tok, class_names[0]) == 0)      c[0] = 1.0;
        else if (strcmp(tok, class_names[1]) == 0)  c[1] = 1.0;
        else if (strcmp(tok, class_names[2]) == 0)  c[2] = 1.0;
        else {
            fprintf(stderr, "Classe desconhecida: '%s'\n", tok);
            continue;
        }

        num_samples++;
    }

    fclose(f);
    printf("[Genann] %d amostras carregadas de '%s'\n", num_samples, path);
    return 0;
}

static int argmax(const double *v, int n)
{
    int best = 0;
    int i;
    for (i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

static double time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main(void)
{
    printf("========================================\n");
    printf("  Iris MLP — Genann (C puro)\n");
    printf("========================================\n\n");

    srand((unsigned)time(NULL));

    if (load_dataset(DATASET_PATH) != 0)
        return 1;

    compute_normalization();

    genann *ann = genann_init(NUM_FEATURES, HIDDEN_LAYERS,
                              HIDDEN_NEURONS, NUM_CLASSES);
    if (!ann) {
        fprintf(stderr, "Erro ao criar rede neural\n");
        return 1;
    }

    printf("Topologia: %d -> %d -> %d\n",
           NUM_FEATURES, HIDDEN_NEURONS, NUM_CLASSES);
    printf("Pesos totais: %d\n", ann->total_weights);
    printf("Treinando por %d épocas (lr=%.4f)...\n\n", EPOCHS, LEARNING_RATE);

    double t_start = time_ns();

    int epoch, i;
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        for (i = 0; i < num_samples; i++) {
            genann_train(ann,
                         inputs + i * NUM_FEATURES,
                         targets + i * NUM_CLASSES,
                         LEARNING_RATE);
        }
    }

    double t_train = (time_ns() - t_start) / 1e6;

    int correct = 0;
    double total_infer_ns = 0.0;

    for (i = 0; i < num_samples; i++) {
        double t0 = time_ns();
        const double *out = genann_run(ann, inputs + i * NUM_FEATURES);
        total_infer_ns += time_ns() - t0;

        int predicted = argmax(out, NUM_CLASSES);
        int actual    = argmax(targets + i * NUM_CLASSES, NUM_CLASSES);
        if (predicted == actual) correct++;
    }

    double accuracy = (double)correct / num_samples * 100.0;
    double avg_infer_us = total_infer_ns / num_samples / 1e3;

    printf("─── Resultados ─────────────────────────\n");
    printf("Acurácia:              %d/%d (%.1f%%)\n",
           correct, num_samples, accuracy);
    printf("Tempo de treinamento:  %.2f ms\n", t_train);
    printf("Latência inferência:   %.2f µs (média por amostra)\n",
           avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n",
           total_infer_ns / 1e6, num_samples);
    printf("─────────────────────────────────────────\n");

    genann_free(ann);

    return 0;
}
