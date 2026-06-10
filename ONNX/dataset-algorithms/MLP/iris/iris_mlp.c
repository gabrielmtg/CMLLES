#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <onnxruntime_c_api.h>

#define NUM_FEATURES  4
#define NUM_CLASSES   3
#define MAX_SAMPLES   200
#define MODEL_PATH    "model/iris_mlp.onnx"
#define DATASET_PATH  "../../../../datasets/iris/iris.data"

static const OrtApi* g_ort = NULL;

#define CHECK_ORT(expr) do {                                         \
    OrtStatus* _status = (expr);                                     \
    if (_status) {                                                   \
        const char* _msg = g_ort->GetErrorMessage(_status);          \
        fprintf(stderr, "ORT error [%s:%d]: %s\n",                  \
                __FILE__, __LINE__, _msg);                           \
        g_ort->ReleaseStatus(_status);                               \
        exit(1);                                                     \
    }                                                                \
} while (0)

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

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main() {
    printf("========================================\n");
    printf("  Iris MLP — ONNX Runtime (C puro)\n");
    printf("========================================\n\n");

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    CHECK_ORT(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "iris", &env));

    OrtSessionOptions* session_options;
    CHECK_ORT(g_ort->CreateSessionOptions(&session_options));
    CHECK_ORT(g_ort->SetIntraOpNumThreads(session_options, 1));
    CHECK_ORT(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));

    OrtSession* session;
    CHECK_ORT(g_ort->CreateSession(env, MODEL_PATH, session_options, &session));

    FILE *fin = fopen(DATASET_PATH, "r");
    if (!fin) {
        printf("Erro ao abrir %s\n", DATASET_PATH);
        return 1;
    }

    float** x_data = malloc(MAX_SAMPLES * sizeof(float*));
    int* y_data = malloc(MAX_SAMPLES * sizeof(int));
    int num_samples = 0;
    char line[1024];

    while (fgets(line, sizeof(line), fin) && num_samples < MAX_SAMPLES) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

        x_data[num_samples] = malloc(NUM_FEATURES * sizeof(float));
        char *tok = strtok(line, ",");
        for (int j = 0; j < NUM_FEATURES && tok; j++) {
            x_data[num_samples][j] = (float)atof(tok);
            tok = strtok(NULL, ",");
        }
        
        if (!tok) {
            free(x_data[num_samples]);
            continue;
        }

        tok[strcspn(tok, "\r\n")] = '\0';
        if (strcmp(tok, class_names[0]) == 0) y_data[num_samples] = 0;
        else if (strcmp(tok, class_names[1]) == 0) y_data[num_samples] = 1;
        else if (strcmp(tok, class_names[2]) == 0) y_data[num_samples] = 2;
        else {
            free(x_data[num_samples]);
            continue;
        }
        num_samples++;
    }
    fclose(fin);

    float fmin[NUM_FEATURES], fmax[NUM_FEATURES];
    for (int j = 0; j < NUM_FEATURES; j++) {
        fmin[j] = x_data[0][j];
        fmax[j] = x_data[0][j];
    }
    for (int i = 1; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (x_data[i][j] < fmin[j]) fmin[j] = x_data[i][j];
            if (x_data[i][j] > fmax[j]) fmax[j] = x_data[i][j];
        }
    }
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            float range = fmax[j] - fmin[j];
            x_data[i][j] = (range > 0.0f) ? (x_data[i][j] - fmin[j]) / range : 0.0f;
        }
    }

    printf("Modelo carregado: %s\n", MODEL_PATH);
    printf("[ONNX Runtime] %d amostras carregadas de '%s'\n\n", num_samples, DATASET_PATH);

    OrtMemoryInfo* memory_info;
    CHECK_ORT(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    int64_t input_shape[] = {1, NUM_FEATURES};

    int correct = 0;
    double total_infer_ns = 0.0;

    for (int i = 0; i < num_samples; i++) {
        OrtValue* input_tensor = NULL;
        OrtValue* output_tensor = NULL;

        CHECK_ORT(g_ort->CreateTensorWithDataAsOrtValue(memory_info, x_data[i], NUM_FEATURES * sizeof(float), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

        double t0 = time_ns();
        
        CHECK_ORT(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));
        
        total_infer_ns += (time_ns() - t0);

        float* out_ptr;
        CHECK_ORT(g_ort->GetTensorMutableData(output_tensor, (void**)&out_ptr));

        int pred = argmax(out_ptr, NUM_CLASSES);
        if (pred == y_data[i]) correct++;

        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(output_tensor);
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("─── Resultados ─────────────────────────\n");
    printf("Acurácia:              %d/%d (%.1f%%)\n", correct, num_samples, accuracy);
    printf("Latência inferência:   %.2f us (média por amostra)\n", avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n", total_infer_ms, num_samples);
    printf("─────────────────────────────────────────\n");

    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    for(int i = 0; i < num_samples; i++) free(x_data[i]);
    free(x_data);
    free(y_data);

    return 0;
}
