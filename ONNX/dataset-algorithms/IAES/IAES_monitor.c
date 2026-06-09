#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <onnxruntime_c_api.h>

#define TEST_DATA "model/iaes_test_data.txt"
#define ONNX_MODEL "model/iaes_model.onnx"
#define INPUTS 12

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    printf("========================================\n");
    printf("  IAES Monitor MLP — ONNX Runtime (C)\n");
    printf("========================================\n\n");

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "IAES_ONNX", &env);

    OrtSessionOptions* session_options;
    g_ort->CreateSessionOptions(&session_options);
    g_ort->SetIntraOpNumThreads(session_options, 1);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);

    OrtSession* session;
    g_ort->CreateSession(env, ONNX_MODEL, session_options, &session);

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) {
        printf("Erro ao abrir %s. Rode o script Python primeiro!\n", TEST_DATA);
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs);

    float** x_data = malloc(num_samples * sizeof(float*));
    float* y_data = malloc(num_samples * sizeof(float));

    for (int i = 0; i < num_samples; i++) {
        x_data[i] = malloc(INPUTS * sizeof(float));
        for (int j = 0; j < INPUTS; j++) {
            fscanf(fin, "%f", &x_data[i][j]);
        }
        fscanf(fin, "%f", &y_data[i]);
    }
    fclose(fin);

    printf("modelo ONNX carregado. Avaliando %d amostras\n", num_samples);

    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    int64_t input_shape[] = {1, INPUTS};

    int correct = 0;
    double total_infer_ns = 0.0;

    for (int i = 0; i < num_samples; i++) {
        OrtValue* input_tensor = NULL;
        OrtValue* output_tensor = NULL;

        g_ort->CreateTensorWithDataAsOrtValue(memory_info, x_data[i], INPUTS * sizeof(float), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

        double t0 = time_ns();
        
        g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
        
        total_infer_ns += (time_ns() - t0);

        float* out_ptr;
        g_ort->GetTensorMutableData(output_tensor, (void**)&out_ptr);

        int pred = (out_ptr[0] >= 0.5f) ? 1 : 0;
        int true_val = (y_data[i] >= 0.5f) ? 1 : 0;

        if (pred == true_val) correct++;

        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(output_tensor);
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n─── Resultados ONNX ─────────────────────────\n");
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
