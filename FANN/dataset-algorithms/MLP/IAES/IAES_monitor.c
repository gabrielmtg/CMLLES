#include "fann.h"
#include <fann_data.h>
#include <fann_train.h>
#include <stdio.h>
#include <time.h>

#define NUM_LAYERS 4
#define INPUTS 12
#define HIDDEN_LAYER_1 16
#define HIDDEN_LAYER_2 16
#define OUTPUT 1

#define WINDOW_SIZE 10
#define N_SIGNALS 4

#define TRAIN_DATA "model/IAES_fann.train"
#define MODEL_OUT "model/IAES_model.net"

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

void evaluate_and_print_metrics(struct fann *ann, struct fann_train_data *data, double train_time_ms) {
    unsigned int total_samples = fann_length_train_data(data);
    int correct = 0;
    double total_infer_ns = 0.0;

    for (unsigned int i = 0; i < total_samples; i++) {
        double t0 = time_ns();
        fann_type *out = fann_run(ann, data->input[i]);
        total_infer_ns += (time_ns() - t0); // Acumula o tempo de inferência

        int pred_label = (out[0] >= 0.5f) ? 1 : 0;
        int true_label = (data->output[i][0] >= 0.5f) ? 1 : 0;

        if (pred_label == true_label) {
            correct++;
        }
    }

    double accuracy = ((double)correct / total_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / total_samples) / 1e3; // Converte ns para µs
    double total_infer_ms = total_infer_ns / 1e6;                 // Converte ns para ms

    printf("\n─── Resultados ─────────────────────────\n");
    printf("Acurácia:              %d/%u (%.1f%%)\n", correct, total_samples, accuracy);
    printf("Tempo de treinamento:  %.2f ms\n", train_time_ms);
    printf("Latência inferência:   %.2f µs (média por amostra)\n", avg_infer_us);
    printf("Inferência total:      %.2f ms (%u amostras)\n", total_infer_ms, total_samples);
    printf("─────────────────────────────────────────\n");
}

int main() {
    const unsigned int num_inputs = INPUTS;
    const unsigned int neurons_hidden_layer_1 = HIDDEN_LAYER_1;
    const unsigned int neurons_hidden_layer_2 = HIDDEN_LAYER_2;
    const unsigned int num_outputs = OUTPUT;
    const unsigned int num_layers = NUM_LAYERS;
    const float learning_rate = 0.0001f;

    const unsigned int max_epochs = 200;
    const unsigned int epochs_between_reports = 10;
    const float desired_error = 0.001f;

    struct fann_train_data *data = fann_read_train_from_file(TRAIN_DATA);
    if (!data) {
        fprintf(stderr, "nao foi possivel carregar %s \n", TRAIN_DATA);
        return -1;
    }

    struct fann *ann = fann_create_standard(num_layers, num_inputs, neurons_hidden_layer_1, neurons_hidden_layer_2, num_outputs);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID);
    fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
    fann_set_learning_rate(ann, learning_rate);

    double t_start = time_ns();
    fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
    double train_time_ms = (time_ns() - t_start) / 1e6;

    evaluate_and_print_metrics(ann, data, train_time_ms);

    fann_save(ann, MODEL_OUT);

    fann_destroy_train(data);
    fann_destroy(ann);

    return 0;
}
