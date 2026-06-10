#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <tensorflow/lite/c/c_api.h>

#define TEST_DATA "model/iaes_test_data.txt"
#define TFLITE_MODEL "model/iaes_model.tflite"
#define INPUTS 12

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    printf("========================================\n");
    printf("  IAES Monitor MLP — LiteRT C API (C)\n");
    printf("========================================\n\n");

    TfLiteModel* model = TfLiteModelCreateFromFile(TFLITE_MODEL);
    if (model == NULL) {
        printf("Erro ao carregar o arquivo modelo: %s\n", TFLITE_MODEL);
        return 1;
    }

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    if (interpreter == NULL) {
        printf("Erro ao criar o interpretador LiteRT.\n");
        TfLiteModelDelete(model);
        TfLiteInterpreterOptionsDelete(options);
        return 1;
    }

    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        printf("Erro ao alocar tensores do interpretador.\n");
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        return 1;
    }

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) {
        printf("Erro ao abrir %s. Rode o script Python primeiro!\n", TEST_DATA);
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    if (fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs) != 3) {
        printf("Erro ao ler cabeçalho do dataset.\n");
        fclose(fin);
        return 1;
    }

    float** x_data = (float**)malloc(num_samples * sizeof(float*));
    float* x_data_flat = (float*)malloc(num_samples * INPUTS * sizeof(float));
    float* y_data = (float*)malloc(num_samples * sizeof(float));

    for (int i = 0; i < num_samples; i++) {
        x_data[i] = &x_data_flat[i * INPUTS];
        for (int j = 0; j < INPUTS; j++) {
            if (fscanf(fin, "%f", &x_data[i][j]) != 1) {
                printf("Aviso: Falha ao ler amostra %d, input %d\n", i, j);
            }
        }
        if (fscanf(fin, "%f", &y_data[i]) != 1) {
            printf("Aviso: Falha ao ler label da amostra %d\n", i);
        }
    }
    fclose(fin);

    printf("Modelo LiteRT em C carregado. Avaliando %d amostras\n", num_samples);

    int correct = 0;
    double total_infer_ns = 0.0;

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    for (int i = 0; i < num_samples; i++) {
        float* in_ptr = (float*)TfLiteTensorData(input_tensor);
        for (int j = 0; j < INPUTS; j++) {
            in_ptr[j] = x_data[i][j];
        }

        double t0 = time_ns();
        TfLiteInterpreterInvoke(interpreter);
        total_infer_ns += (time_ns() - t0);

        float* out_ptr = (float*)TfLiteTensorData(output_tensor);
        float out_val = out_ptr[0];

        int pred = (out_val >= 0.5f) ? 1 : 0;
        int true_val = (y_data[i] >= 0.5f) ? 1 : 0;

        if (pred == true_val) correct++;
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n─── Resultados LiteRT (C API) ───────────────────\n");
    printf("Acurácia:              %d/%d (%.1f%%)\n", correct, num_samples, accuracy);
    printf("Latência inferência:   %.2f us (média por amostra)\n", avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n", total_infer_ms, num_samples);
    printf("─────────────────────────────────────────\n");

    free(x_data_flat);
    free(x_data);
    free(y_data);

    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return 0;
}
