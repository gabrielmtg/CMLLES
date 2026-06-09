#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fann.h>

#define NUM_FEATURES    4
#define NUM_CLASSES     3
#define HIDDEN_NEURONS  16
#define MAX_EPOCHS      5000
#define REPORT_INTERVAL 1000
#define DESIRED_ERROR   0.001f
#define LEARNING_RATE   0.01f

#define DATASET_PATH    "../../../datasets/iris/iris.data"
#define TRAIN_DATA      "model/iris_fann.train"
#define MODEL_OUT       "model/iris_model.net"
#define MAX_SAMPLES     200

static const char *class_names[NUM_CLASSES] = {
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
};

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static int argmax_f(fann_type *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

static int generate_train_file(const char *csv_path, const char *train_path) {
    FILE *fin = fopen(csv_path, "r");
    if (!fin) return -1;

    float data[MAX_SAMPLES][NUM_FEATURES];
    int labels[MAX_SAMPLES];
    int count = 0;
    char line[1024];

    while (fgets(line, sizeof(line), fin) && count < MAX_SAMPLES) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

        char *tok = strtok(line, ",");
        for (int j = 0; j < NUM_FEATURES && tok; j++) {
            data[count][j] = (float)atof(tok);
            tok = strtok(NULL, ",");
        }
        
        if (!tok) continue;
        tok[strcspn(tok, "\r\n")] = '\0';

        if (strcmp(tok, class_names[0]) == 0) labels[count] = 0;
        else if (strcmp(tok, class_names[1]) == 0) labels[count] = 1;
        else if (strcmp(tok, class_names[2]) == 0) labels[count] = 2;
        else continue;


        count++;
    }
    fclose(fin);

    float fmin[NUM_FEATURES], fmax[NUM_FEATURES];
    for (int j = 0; j < NUM_FEATURES; j++) {
        fmin[j] = data[0][j];
        fmax[j] = data[0][j];
    }
    
    for (int i = 1; i < count; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (data[i][j] < fmin[j]) fmin[j] = data[i][j];
            if (data[i][j] > fmax[j]) fmax[j] = data[i][j];
        }
    }
    
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            float range = fmax[j] - fmin[j];
            data[i][j] = (range > 0.0f) ? (data[i][j] - fmin[j]) / range : 0.0f;
        }
    }

    FILE *fout = fopen(train_path, "w");
    if (!fout) return -1;

    fprintf(fout, "%d %d %d\n", count, NUM_FEATURES, NUM_CLASSES);
    for (int i = 0; i < count; i++) {
        fprintf(fout, "%f %f %f %f\n", data[i][0], data[i][1], data[i][2], data[i][3]);
        for (int j = 0; j < NUM_CLASSES; j++) {
            fprintf(fout, "%s%d", j > 0 ? " " : "", (labels[i] == j) ? 1 : 0);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);

    return 0;
}

void evaluate_and_print_metrics(struct fann *ann, struct fann_train_data *data, double train_time_ms) {
    unsigned int total_samples = fann_length_train_data(data);
    int correct = 0;
    double total_infer_ns = 0.0;

    for (unsigned int i = 0; i < total_samples; i++) {
        double t0 = time_ns();
        fann_type *out = fann_run(ann, data->input[i]);
        total_infer_ns += (time_ns() - t0);


        int pred_label = argmax_f(out, NUM_CLASSES);
        int true_label = argmax_f(data->output[i], NUM_CLASSES);

        if (pred_label == true_label) {
            correct++;
        }
    }

    double accuracy = ((double)correct / total_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / total_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n─── Resultados ─────────────────────────\n");
    printf("Acuracia:              %d/%u (%.1f%%)\n", correct, total_samples, accuracy);
    printf("Tempo de treinamento:  %.2f ms\n", train_time_ms);
    printf("Latencia inferencia:   %.2f us (media por amostra)\n", avg_infer_us);
    printf("Inferencia total:      %.2f ms (%u amostras)\n", total_infer_ms, total_samples);
    printf("─────────────────────────────────────────\n");
}

int main(void) {
    printf("========================================\n");
    printf("  Iris MLP — FANN (C puro)\n");
    printf("========================================\n\n");

    if (generate_train_file(DATASET_PATH, TRAIN_DATA) != 0) {
        fprintf(stderr, "Erro ao gerar arquivo de treino.\n");
        return 1;
    }

    struct fann_train_data *data = fann_read_train_from_file(TRAIN_DATA);
    if (!data) return 1;

    unsigned int total_samples = fann_length_train_data(data);
    printf("[FANN] Arquivo de treino gerado/carregado: '%s' (%u amostras)\n", TRAIN_DATA, total_samples);

    const unsigned int layers[] = {NUM_FEATURES, HIDDEN_NEURONS, NUM_CLASSES};
    struct fann *ann = fann_create_standard_array(3, layers);
    if (!ann) return 1;

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
    fann_set_learning_rate(ann, LEARNING_RATE);

    printf("Topologia: %d -> %d -> %d\n", NUM_FEATURES, HIDDEN_NEURONS, NUM_CLASSES);
    printf("Conexoes totais: %u\n", fann_get_total_connections(ann));
    printf("Treinando por ate %d epocas (erro alvo=%.4f)...\n\n", MAX_EPOCHS, DESIRED_ERROR);

    double t_start = time_ns();
    fann_train_on_data(ann, data, MAX_EPOCHS, REPORT_INTERVAL, DESIRED_ERROR);
    double train_time_ms = (time_ns() - t_start) / 1e6;

    evaluate_and_print_metrics(ann, data, train_time_ms);

    fann_save(ann, MODEL_OUT);

    fann_destroy_train(data);
    fann_destroy(ann);

    return 0;
}
