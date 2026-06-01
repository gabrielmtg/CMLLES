/*
 * Iris MLP Classifier — FANN (Fast Artificial Neural Network)
 *
 * Rede: 4 inputs → 16 hidden (sigmoid symmetric) → 3 outputs (sigmoid symmetric)
 * Treinamento e inferência em C puro usando libfann.
 *
 * Compilação: make
 * Execução:   ./iris_mlp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fann.h>

/* ─── Configurações ─────────────────────────────────────────── */
#define NUM_FEATURES    4
#define NUM_CLASSES     3
#define HIDDEN_NEURONS  16
#define MAX_EPOCHS      5000
#define REPORT_INTERVAL 1000
#define DESIRED_ERROR   0.001f
#define LEARNING_RATE   0.01f
#define DATASET_PATH    "../../../datasets/iris/iris.data"
#define TRAIN_FILE      "iris_fann.train"
#define MAX_SAMPLES     200

/* ─── Nomes das classes ─────────────────────────────────────── */
static const char *class_names[NUM_CLASSES] = {
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
};

/* ─── Dados para avaliação ──────────────────────────────────── */
static float eval_inputs[MAX_SAMPLES][NUM_FEATURES];
static int   eval_labels[MAX_SAMPLES];
static int   num_samples = 0;

/* ─── Tempo em nanossegundos ────────────────────────────────── */
static double time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/*
 * Gera o arquivo .train no formato FANN a partir do iris.data.
 * Formato:
 *   <num_samples> <num_inputs> <num_outputs>
 *   <input1> <input2> <input3> <input4>
 *   <output1> <output2> <output3>
 *   ...
 *
 * Também aplica normalização min-max [0,1] nos inputs.
 */
static int generate_train_file(const char *csv_path, const char *train_path)
{
    FILE *fin = fopen(csv_path, "r");
    if (!fin) {
        fprintf(stderr, "Erro: não foi possível abrir '%s'\n", csv_path);
        return -1;
    }

    /* Primeira passada: ler tudo em memória */
    float  data[MAX_SAMPLES][NUM_FEATURES];
    int    labels[MAX_SAMPLES];
    int    count = 0;
    char   line[1024];

    while (fgets(line, sizeof(line), fin) && count < MAX_SAMPLES) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;

        char *tok = strtok(line, ",");
        int j;
        for (j = 0; j < NUM_FEATURES && tok; j++) {
            data[count][j] = (float)atof(tok);
            tok = strtok(NULL, ",");
        }
        if (!tok) continue;

        tok[strcspn(tok, "\r\n")] = '\0';

        if (strcmp(tok, class_names[0]) == 0)      labels[count] = 0;
        else if (strcmp(tok, class_names[1]) == 0)  labels[count] = 1;
        else if (strcmp(tok, class_names[2]) == 0)  labels[count] = 2;
        else continue;

        count++;
    }
    fclose(fin);

    /* Normalização min-max */
    float fmin[NUM_FEATURES], fmax[NUM_FEATURES];
    int i, j;
    for (j = 0; j < NUM_FEATURES; j++) {
        fmin[j] = data[0][j];
        fmax[j] = data[0][j];
    }
    for (i = 1; i < count; i++) {
        for (j = 0; j < NUM_FEATURES; j++) {
            if (data[i][j] < fmin[j]) fmin[j] = data[i][j];
            if (data[i][j] > fmax[j]) fmax[j] = data[i][j];
        }
    }
    for (i = 0; i < count; i++) {
        for (j = 0; j < NUM_FEATURES; j++) {
            float range = fmax[j] - fmin[j];
            data[i][j] = (range > 0.0f)
                         ? (data[i][j] - fmin[j]) / range
                         : 0.0f;
        }
    }

    /* Salvar dados de avaliação */
    num_samples = count;
    for (i = 0; i < count; i++) {
        for (j = 0; j < NUM_FEATURES; j++)
            eval_inputs[i][j] = data[i][j];
        eval_labels[i] = labels[i];
    }

    /* Escrever arquivo FANN */
    FILE *fout = fopen(train_path, "w");
    if (!fout) {
        fprintf(stderr, "Erro: não foi possível criar '%s'\n", train_path);
        return -1;
    }

    fprintf(fout, "%d %d %d\n", count, NUM_FEATURES, NUM_CLASSES);
    for (i = 0; i < count; i++) {
        fprintf(fout, "%f %f %f %f\n",
                data[i][0], data[i][1], data[i][2], data[i][3]);

        /* One-hot encoding */
        for (j = 0; j < NUM_CLASSES; j++) {
            fprintf(fout, "%s%d", j > 0 ? " " : "", (labels[i] == j) ? 1 : 0);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);

    printf("[FANN] Arquivo de treino gerado: '%s' (%d amostras)\n",
           train_path, count);
    return 0;
}

/* ─── Utilitário: argmax ────────────────────────────────────── */
static int argmax_f(fann_type *v, int n)
{
    int best = 0;
    int i;
    for (i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

/* ─── Main ──────────────────────────────────────────────────── */
int main(void)
{
    printf("========================================\n");
    printf("  Iris MLP — FANN (C puro)\n");
    printf("========================================\n\n");

    /* 1. Gerar arquivo de treino */
    if (generate_train_file(DATASET_PATH, TRAIN_FILE) != 0)
        return 1;

    /* 2. Criar rede neural: 3 camadas (input, hidden, output) */
    const unsigned int layers[] = {NUM_FEATURES, HIDDEN_NEURONS, NUM_CLASSES};
    struct fann *ann = fann_create_standard_array(3, layers);
    if (!ann) {
        fprintf(stderr, "Erro ao criar rede neural\n");
        return 1;
    }

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_learning_rate(ann, LEARNING_RATE);

    printf("Topologia: %d -> %d -> %d\n",
           NUM_FEATURES, HIDDEN_NEURONS, NUM_CLASSES);
    printf("Conexões totais: %u\n", fann_get_total_connections(ann));
    printf("Treinando por até %d épocas (erro alvo=%.4f)...\n\n",
           MAX_EPOCHS, DESIRED_ERROR);

    /* 3. Treinamento */
    double t_start = time_ns();

    fann_train_on_file(ann, TRAIN_FILE, MAX_EPOCHS, REPORT_INTERVAL,
                       DESIRED_ERROR);

    double t_train = (time_ns() - t_start) / 1e6; /* ms */

    /* 4. Avaliação */
    int correct = 0;
    double total_infer_ns = 0.0;
    int i;

    for (i = 0; i < num_samples; i++) {
        fann_type input[NUM_FEATURES];
        int j;
        for (j = 0; j < NUM_FEATURES; j++)
            input[j] = (fann_type)eval_inputs[i][j];

        double t0 = time_ns();
        fann_type *out = fann_run(ann, input);
        total_infer_ns += time_ns() - t0;

        int predicted = argmax_f(out, NUM_CLASSES);
        if (predicted == eval_labels[i]) correct++;
    }

    double accuracy = (double)correct / num_samples * 100.0;
    double avg_infer_us = total_infer_ns / num_samples / 1e3;

    /* 5. Resultados */
    printf("\n─── Resultados ─────────────────────────\n");
    printf("Acurácia:              %d/%d (%.1f%%)\n",
           correct, num_samples, accuracy);
    printf("Tempo de treinamento:  %.2f ms\n", t_train);
    printf("Latência inferência:   %.2f µs (média por amostra)\n",
           avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n",
           total_infer_ns / 1e6, num_samples);
    printf("─────────────────────────────────────────\n");

    /* 6. Cleanup */
    fann_destroy(ann);

    return 0;
}
