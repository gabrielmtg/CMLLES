#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

const OrtApi* g_ort = NULL;

#define CHECK_STATUS(expr) do { \
    OrtStatus* onnx_status = (expr); \
    if (onnx_status != NULL) { \
        const char* msg = g_ort->GetErrorMessage(onnx_status); \
        fprintf(stderr, "ERRO DO ONNX RUNTIME: %s\n", msg); \
        g_ort->ReleaseStatus(onnx_status); \
        exit(-1); \
    } \
} while (0)

int main() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Falha critica ao inicializar a API do ONNX.\n");
        return -1;
    }

    OrtEnv* env;
    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CmapssInference", &env));

    OrtSessionOptions* session_options;
    CHECK_STATUS(g_ort->CreateSessionOptions(&session_options));

    const char* model_path = "model/cmapss_lstm.onnx"; 
    OrtSession* session = NULL;
    
    printf("Tentando carregar o modelo em: %s\n", model_path);
    CHECK_STATUS(g_ort->CreateSession(env, model_path, session_options, &session));
    printf("Modelo carregado com sucesso!\n");

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    int64_t input_shape[] = {1, 30, 14};
    size_t input_tensor_size = 1 * 30 * 14;
    
    float* input_tensor_values = (float*)malloc(input_tensor_size * sizeof(float));
    for (size_t i = 0; i < input_tensor_size; i++) {
        input_tensor_values[i] = 0.5f; 
    }

    OrtMemoryInfo* memory_info;
    CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtValue* input_tensor = NULL;
    CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, input_tensor_values, input_tensor_size * sizeof(float), 
        input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor
    ));

    OrtValue* output_tensor = NULL;
    printf("Executando inferencia...\n");
    CHECK_STATUS(g_ort->Run(
        session, NULL, input_names, (const OrtValue* const*)&input_tensor, 
        1, output_names, 1, &output_tensor
    ));

    float* output_arr;
    CHECK_STATUS(g_ort->GetTensorMutableData(output_tensor, (void**)&output_arr));
    float rul_em_ciclos = output_arr[0] * 125.0f;

    printf("========================================\n");
    printf("Vida Util Restante Prevista (RUL): %.2f ciclos\n", rul_em_ciclos);
    printf("========================================\n");

    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    free(input_tensor_values);

    return 0;
}
