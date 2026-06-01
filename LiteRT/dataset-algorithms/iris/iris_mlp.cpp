#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model/iris_model_data.h"

#define NUM_FEATURES  4
#define NUM_CLASSES   3
#define MAX_SAMPLES   200
#define DATASET_PATH  "../../../datasets/iris/iris.data"
static constexpr int kTensorArenaSize = 4 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static const char *class_names[NUM_CLASSES] = {
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
};

static float all_inputs[MAX_SAMPLES][NUM_FEATURES];
static int   all_labels[MAX_SAMPLES];
static int   num_samples = 0;

static double time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
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

        char *tok = strtok(line, ",");
        for (int j = 0; j < NUM_FEATURES && tok; j++) {
            all_inputs[num_samples][j] = (float)atof(tok);
            tok = strtok(NULL, ",");
        }
        if (!tok) continue;

        tok[strcspn(tok, "\r\n")] = '\0';

        if (strcmp(tok, class_names[0]) == 0)      all_labels[num_samples] = 0;
        else if (strcmp(tok, class_names[1]) == 0)  all_labels[num_samples] = 1;
        else if (strcmp(tok, class_names[2]) == 0)  all_labels[num_samples] = 2;
        else continue;

        num_samples++;
    }
    fclose(f);

    float fmin[NUM_FEATURES], fmax[NUM_FEATURES];
    for (int j = 0; j < NUM_FEATURES; j++) {
        fmin[j] = all_inputs[0][j];
        fmax[j] = all_inputs[0][j];
    }
    for (int i = 1; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (all_inputs[i][j] < fmin[j]) fmin[j] = all_inputs[i][j];
            if (all_inputs[i][j] > fmax[j]) fmax[j] = all_inputs[i][j];
        }
    }
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            float range = fmax[j] - fmin[j];
            all_inputs[i][j] = (range > 0.0f)
                               ? (all_inputs[i][j] - fmin[j]) / range
                               : 0.0f;
        }
    }

    printf("[LiteRT] %d amostras carregadas de '%s'\n", num_samples, path);
    return 0;
}

static int argmax(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main()
{
    printf("========================================\n");
    printf("  Iris MLP — LiteRT / TFLite Micro\n");
    printf("========================================\n\n");

    if (load_dataset(DATASET_PATH) != 0)
        return 1;

    const tflite::Model *model = tflite::GetModel(iris_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        fprintf(stderr, "Versão do modelo incompatível!\n");
        return 1;
    }

    tflite::MicroMutableOpResolver<3> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();

    tflite::MicroInterpreter interpreter(model, resolver,
                                          tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "Erro ao alocar tensores!\n");
        return 1;
    }

    TfLiteTensor *input_tensor  = interpreter.input(0);
    TfLiteTensor *output_tensor = interpreter.output(0);

    printf("Topologia: %d -> 16 (ReLU) -> %d (Softmax)\n",
           NUM_FEATURES, NUM_CLASSES);
    printf("Tensor arena: %d bytes\n\n", kTensorArenaSize);

    int correct = 0;
    double total_infer_ns = 0.0;

    for (int i = 0; i < num_samples; i++) {
        float *in_data = input_tensor->data.f;
        for (int j = 0; j < NUM_FEATURES; j++)
            in_data[j] = all_inputs[i][j];

        double t0 = time_ns();
        if (interpreter.Invoke() != kTfLiteOk) {
            fprintf(stderr, "Erro na inferência (amostra %d)!\n", i);
            continue;
        }
        total_infer_ns += time_ns() - t0;

        float *out_data = output_tensor->data.f;
        int predicted = argmax(out_data, NUM_CLASSES);
        if (predicted == all_labels[i]) correct++;
    }

    double accuracy = (double)correct / num_samples * 100.0;
    double avg_infer_us = total_infer_ns / num_samples / 1e3;

    printf("─── Resultados ─────────────────────────\n");
    printf("Acurácia:              %d/%d (%.1f%%)\n",
           correct, num_samples, accuracy);
    printf("Latência inferência:   %.2f µs (média por amostra)\n",
           avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n",
           total_infer_ns / 1e6, num_samples);
    printf("─────────────────────────────────────────\n");

    return 0;
}
