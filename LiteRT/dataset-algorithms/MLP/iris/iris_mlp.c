#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tensorflow/lite/c/c_api.h>

#define NUM_FEATURES  4
#define NUM_CLASSES   3
#define MAX_SAMPLES   200
#define MODEL_PATH    "model/iris_mlp.tflite"
#define DATASET_PATH  "../../../../datasets/iris/iris.data"

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
    printf("  Iris MLP — LiteRT / TFLite (C Puro)\n");
    printf("========================================\n\n");

    // 1. Carrega o modelo .tflite
    TfLiteModel* model = TfLiteModelCreateFromFile(MODEL_PATH);
    if (!model) {
        printf("Erro ao carregar o modelo: %s\n", MODEL_PATH);
        return 1;
    }

    // 2. Configurações do Interpretador (ex: número de threads)
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    // 3. Cria o Interpretador e aloca os Tensores de memória
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        printf("Erro ao alocar tensores do LiteRT.\n");
        return 1;
    }

    // Lógica de Carregamento do Dataset (igual aos códigos anteriores)
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
        if (!tok) { free(x_data[num_samples]); continue; }
        
        tok[strcspn(tok, "\r\n")] = '\0';
        if (strcmp(tok, class_names[0]) == 0) y_data[num_samples] = 0;
        else if (strcmp(tok, class_names[1]) == 0) y_data[num_samples] = 1;
        else if (strcmp(tok, class_names[2]) == 0) y_data[num_samples] = 2;
        else { free(x_data[num_samples]); continue; }
        num_samples++;
    }
    fclose(fin);

    // Normalização Min-Max
    float fmin[NUM_FEATURES], fmax[NUM_FEATURES];
    for (int j = 0; j < NUM_FEATURES; j++) { fmin[j] = x_data[0][j]; fmax[j] = x_data[0][j]; }
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

    printf("[LiteRT] %d amostras carregadas para inferência\n\n", num_samples);

    int correct = 0;
    double total_infer_ns = 0.0;

    // Obtém o ponteiro para o Tensor de Entrada (índice 0)
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    // Obtém o ponteiro para o Tensor de Saída (índice 0)
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    // 4. Loop de Inferência
    for (int i = 0; i < num_samples; i++) {
        // Copia os dados do C para o tensor de entrada do LiteRT
        TfLiteTensorCopyFromBuffer(input_tensor, x_data[i], NUM_FEATURES * sizeof(float));

        double t0 = time_ns();
        
        // Executa o Forward Pass
        if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
            printf("Erro na inferência da amostra %d\n", i);
            continue;
        }
        
        total_infer_ns += (time_ns() - t0);

        // Copia os dados do tensor de saída de volta para um array em C
        float out_data[NUM_CLASSES];
        TfLiteTensorCopyToBuffer(output_tensor, out_data, NUM_CLASSES * sizeof(float));

        // Pega a maior probabilidade
        int pred = argmax(out_data, NUM_CLASSES);
        if (pred == y_data[i]) correct++;
    }

    // 5. Exibe os resultados
    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("─── Resultados LiteRT ────────────────────\n");
    printf("Acurácia:              %d/%d (%.1f%%)\n", correct, num_samples, accuracy);
    printf("Latência inferência:   %.2f us (média por amostra)\n", avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n", total_infer_ms, num_samples);
    printf("─────────────────────────────────────────\n");

    // Limpeza de memória do ecossistema LiteRT
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    // Limpeza de memória do dataset
    for(int i = 0; i < num_samples; i++) free(x_data[i]);
    free(x_data);
    free(y_data);

    return 0;
}
