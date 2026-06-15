#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <fann.h>

#define MAX_EPOCHS 50
#define REPORT_INTERVAL 10
#define DESIRED_ERROR 0.001f
#define LEARNING_RATE 0.001f
#define ANOMALY_THRESHOLD 0.02f

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main(void) {
    struct fann_train_data *data = fann_read_train_from_file("model/nsl_train.data");
    if (!data) {
        return 1;
    }

    unsigned int input_dim = data->num_input;
    const unsigned int layers[] = {input_dim, 64, 32, 16, 32, 64, input_dim};
    struct fann *ann = fann_create_standard_array(7, layers);
    if (!ann) {
        fann_destroy_train(data);
        return 1;
    }

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
    fann_set_learning_rate(ann, LEARNING_RATE);

    double t_start = time_ns();
    fann_train_on_data(ann, data, MAX_EPOCHS, REPORT_INTERVAL, DESIRED_ERROR);
    double train_time_ms = (time_ns() - t_start) / 1e6;

    fann_save(ann, "model/nsl_model.net");
    fann_destroy_train(data);

    FILE *f_test = fopen("model/nsl_test.txt", "r");
    if (!f_test) {
        fann_destroy(ann);
        return 1;
    }

    int num_test, num_features, num_targets;
    if (fscanf(f_test, "%d %d %d", &num_test, &num_features, &num_targets) != 3) {
        fclose(f_test);
        fann_destroy(ann);
        return 1;
    }

    float *input = (float *)malloc(num_features * sizeof(float));
    int correct = 0;
    double total_infer_ns = 0.0;

    for (int i = 0; i < num_test; i++) {
        for (int j = 0; j < num_features; j++) {
            if (fscanf(f_test, "%f", &input[j]) != 1) break;
        }
        float target;
        if (fscanf(f_test, "%f", &target) != 1) break;

        double t0 = time_ns();
        fann_type *out = fann_run(ann, input);
        total_infer_ns += (time_ns() - t0);

        float mse = 0.0f;
        for (int j = 0; j < num_features; j++) {
            float diff = input[j] - out[j];
            mse += diff * diff;
        }
        mse /= num_features;

        int pred = (mse > ANOMALY_THRESHOLD) ? 1 : 0;
        int actual = (target > 0.5f) ? 1 : 0;

        if (pred == actual) {
            correct++;
        }
    }

    free(input);
    fclose(f_test);
    fann_destroy(ann);

    double accuracy = ((double)correct / num_test) * 100.0;
    double avg_infer_us = (total_infer_ns / num_test) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n");
    printf("%d/%d (%.1f%%)\n", correct, num_test, accuracy);
    printf("%.2f ms\n", train_time_ms);
    printf("%.2f us\n", avg_infer_us);
    printf("%.2f ms\n", total_infer_ms);

    return 0;
}
