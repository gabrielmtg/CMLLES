#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <tensorflow/lite/c/c_api.h>

#define TEST_DATA "model/nsl_test_data.txt"
#define TFLITE_MODEL "model/nsl_model.tflite"

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    printf("=============================================\n");
    printf("  NSL-KDD Autoencoder Monitor — LiteRT (C)\n");
    printf("=============================================\n\n");

    TfLiteModel* model = TfLiteModelCreateFromFile(TFLITE_MODEL);
    if (model == NULL) {
        printf("Erro ao carregar o arquivo modelo: %s\n", TFLITE_MODEL);
        return 1;
    }

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    
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
        printf("Erro ao abrir o ficheiro %s. Corra o script Python primeiro!\n", TEST_DATA);
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    if (fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs) != 3) {
        printf("Erro ao ler cabecalho.\n");
        return 1;
    }

    float** x_data = malloc(num_samples * sizeof(float*));
    float* x_data_flat = (float*)malloc(num_samples * num_inputs * sizeof(float));
    float* y_data = malloc(num_samples * sizeof(float));

    for (int i = 0; i < num_samples; i++) {
        x_data[i] = &x_data_flat[i * num_inputs];
        for (int j = 0; j < num_inputs; j++) {
            fscanf(fin, "%f", &x_data[i][j]);
        }
        fscanf(fin, "%f", &y_data[i]);
    }
    fclose(fin);

    printf("Modelo LiteRT carregado. A avaliar %d amostras com %d atributos...\n", num_samples, num_inputs);

    int correct = 0;
    double total_infer_ns = 0.0;
    float threshold = 0.008f; 

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    for (int i = 0; i < num_samples; i++) {
        float* in_ptr = (float*)TfLiteTensorData(input_tensor);
        for (int j = 0; j < num_inputs; j++) {
            in_ptr[j] = x_data[i][j];
        }

        double t0 = time_ns();
        TfLiteInterpreterInvoke(interpreter);
        total_infer_ns += (time_ns() - t0);

        float* out_ptr = (float*)TfLiteTensorData(output_tensor);
        
        float mse = 0.0f;
        for (int j = 0; j < num_inputs; j++) {
            float diff = x_data[i][j] - out_ptr[j];
            mse += (diff * diff);
        }
        mse /= num_inputs;

        int true_val = (y_data[i] >= 0.5f) ? 1 : 0;
        int pred = (mse >= threshold) ? 1 : 0;

        if (pred == true_val) correct++;
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n─── Resultados LiteRT (Autoencoder) ────────────\n");
    printf("Exatidão:              %d/%d (%.1f%%)\n", correct, num_samples, accuracy);
    printf("Latência inferência:   %.2f us (média por amostra)\n", avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n", total_infer_ms, num_samples);
    printf("Limiar (Threshold):    %.4f\n", threshold);
    printf("────────────────────────────────────────────────\n");

    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
    
    free(x_data_flat);
    free(x_data);
    free(y_data);

    return 0;
}
