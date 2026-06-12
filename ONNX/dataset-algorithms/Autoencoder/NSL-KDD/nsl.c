#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <onnxruntime_c_api.h>

#define TEST_DATA "model/nsl_test_data.txt"
#define ONNX_MODEL "model/nsl_model.onnx"

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    printf("=============================================\n");
    printf("  NSL-KDD Autoencoder Monitor — ONNX (C)\n");
    printf("=============================================\n\n");

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

#define ORT_ABORT_ON_ERROR(expr) do { \
    OrtStatus* onnx_status = (expr); \
    if (onnx_status != NULL) { \
        const char* msg = g_ort->GetErrorMessage(onnx_status); \
        fprintf(stderr, "Erro do ONNX Runtime: %s\n", msg); \
        g_ort->ReleaseStatus(onnx_status); \
        exit(1); \
    } \
} while(0)

    OrtEnv* env;
    ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "NSL_ONNX", &env));

    OrtSessionOptions* session_options;
    ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
    ORT_ABORT_ON_ERROR(g_ort->SetIntraOpNumThreads(session_options, 1));
    ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));

    OrtSession* session = NULL;
    ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, ONNX_MODEL, session_options, &session));

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) {
        printf("Erro ao abrir o ficheiro %s. Corra o script Python primeiro!\n", TEST_DATA);
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs);

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

    printf("Modelo ONNX carregado. A avaliar %d amostras com %d atributos...\n", num_samples, num_inputs);

    OrtMemoryInfo* memory_info;
    ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    int64_t input_shape[] = {1, num_inputs};

    int correct = 0;
    double total_infer_ns = 0.0;

    float threshold = 0.008f; 

    for (int i = 0; i < num_samples; i++) {
        OrtValue* input_tensor = NULL;
        OrtValue* output_tensor = NULL;

        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, x_data[i], num_inputs * sizeof(float), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

        double t0 = time_ns();
        
        ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));
        
        total_infer_ns += (time_ns() - t0);

        float* out_ptr;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&out_ptr));

        float mse = 0.0f;
        for (int j = 0; j < num_inputs; j++) {
            float diff = x_data[i][j] - out_ptr[j];
            mse += (diff * diff);
        }
        mse /= num_inputs;

        int true_val = (y_data[i] >= 0.5f) ? 1 : 0;
        int pred = (mse >= threshold) ? 1 : 0;

        if (pred == true_val) correct++;

        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(output_tensor);
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n─── Resultados ONNX (Autoencoder) ──────────────\n");
    printf("Exatidão:              %d/%d (%.1f%%)\n", correct, num_samples, accuracy);
    printf("Latência inferência:   %.2f us (média por amostra)\n", avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n", total_infer_ms, num_samples);
    printf("Limiar (Threshold):    %.4f\n", threshold);
    printf("────────────────────────────────────────────────\n");

    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    
    free(x_data_flat);
    free(x_data);
    free(y_data);

    return 0;
}
