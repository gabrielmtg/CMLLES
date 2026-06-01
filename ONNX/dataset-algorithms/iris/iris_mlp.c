#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <onnxruntime_c_api.h>

#define NUM_FEATURES  4
#define NUM_CLASSES   3
#define MAX_SAMPLES   200
#define MODEL_PATH    "model/iris_mlp.onnx"
#define DATASET_PATH  "../../../datasets/iris/iris.data"

static const char *class_names[NUM_CLASSES] = {
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
};

static float inputs[MAX_SAMPLES][NUM_FEATURES];
static int   labels[MAX_SAMPLES];
static int   num_samples = 0;

static double time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

#define ORT_CHECK(api, expr)                                          \
    do {                                                              \
        OrtStatus *_s = (expr);                                       \
        if (_s) {                                                     \
            fprintf(stderr, "ONNX Runtime erro: %s\n",               \
                    (api)->GetErrorMessage(_s));                      \
            (api)->ReleaseStatus(_s);                                 \
            exit(1);                                                  \
        }                                                             \
    } while (0)

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
        int j;
        for (j = 0; j < NUM_FEATURES && tok; j++) {
            inputs[num_samples][j] = (float)atof(tok);
            tok = strtok(NULL, ",");
        }
        if (!tok) continue;

        tok[strcspn(tok, "\r\n")] = '\0';

        if (strcmp(tok, class_names[0]) == 0)      labels[num_samples] = 0;
        else if (strcmp(tok, class_names[1]) == 0)  labels[num_samples] = 1;
        else if (strcmp(tok, class_names[2]) == 0)  labels[num_samples] = 2;
        else continue;

        num_samples++;
    }
    fclose(f);

    float fmin[NUM_FEATURES], fmax[NUM_FEATURES];
    int i;
    for (j = 0; j < NUM_FEATURES; j++) {
        fmin[j] = inputs[0][j];
        fmax[j] = inputs[0][j];
    }
    for (i = 1; i < num_samples; i++) {
        for (j = 0; j < NUM_FEATURES; j++) {
            if (inputs[i][j] < fmin[j]) fmin[j] = inputs[i][j];
            if (inputs[i][j] > fmax[j]) fmax[j] = inputs[i][j];
        }
    }
    for (i = 0; i < num_samples; i++) {
        for (j = 0; j < NUM_FEATURES; j++) {
            float range = fmax[j] - fmin[j];
            inputs[i][j] = (range > 0.0f)
                           ? (inputs[i][j] - fmin[j]) / range
                           : 0.0f;
        }
    }

    printf("[ONNX Runtime] %d amostras carregadas de '%s'\n", num_samples, path);
    return 0;
}

static int argmax(const float *v, int n)
{
    int best = 0;
    int i;
    for (i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main(void)
{
    printf("========================================\n");
    printf("  Iris MLP — ONNX Runtime (C puro)\n");
    printf("========================================\n\n");

    if (load_dataset(DATASET_PATH) != 0)
        return 1;

    const OrtApi *ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv *env = NULL;
    ORT_CHECK(ort, ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "iris", &env));

    OrtSessionOptions *opts = NULL;
    ORT_CHECK(ort, ort->CreateSessionOptions(&opts));
    ORT_CHECK(ort, ort->SetIntraOpNumThreads(opts, 1));

    OrtSession *session = NULL;
    ORT_CHECK(ort, ort->CreateSession(env, MODEL_PATH, opts, &session));

    printf("Modelo carregado: %s\n\n", MODEL_PATH);

    /* 3. Preparar estruturas para inferência */
    OrtMemoryInfo *mem_info = NULL;
    ORT_CHECK(ort, ort->CreateCpuMemoryInfo(OrtArenaAllocator,
                                             OrtMemTypeDefault, &mem_info));

    const char *input_names[]  = {"input"};
    const char *output_names[] = {"output"};
    int64_t input_shape[] = {1, NUM_FEATURES};

    /* 4. Avaliação */
    int correct = 0;
    double total_infer_ns = 0.0;
    int i;

    for (i = 0; i < num_samples; i++) {
        float sample[NUM_FEATURES];
        int j;
        for (j = 0; j < NUM_FEATURES; j++)
            sample[j] = inputs[i][j];

        OrtValue *input_tensor = NULL;
        ORT_CHECK(ort, ort->CreateTensorWithDataAsOrtValue(
            mem_info, sample, sizeof(sample),
            input_shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor));

        OrtValue *output_tensor = NULL;

        double t0 = time_ns();
        ORT_CHECK(ort, ort->Run(session, NULL,
                                input_names, (const OrtValue *const *)&input_tensor, 1,
                                output_names, 1, &output_tensor));
        total_infer_ns += time_ns() - t0;

        float *output_data = NULL;
        ORT_CHECK(ort, ort->GetTensorMutableData(output_tensor,
                                                  (void **)&output_data));

        int predicted = argmax(output_data, NUM_CLASSES);
        if (predicted == labels[i]) correct++;

        ort->ReleaseValue(output_tensor);
        ort->ReleaseValue(input_tensor);
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

    ort->ReleaseMemoryInfo(mem_info);
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(opts);
    ort->ReleaseEnv(env);

    return 0;
}
