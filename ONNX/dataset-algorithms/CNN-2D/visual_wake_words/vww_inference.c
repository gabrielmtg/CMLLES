#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

#define BATCH_SIZE 1
#define CHANNELS 3
#define HEIGHT 96
#define WIDTH 96

const OrtApi* g_ort = NULL;

#define CHECK_STATUS(expr) do { \
    OrtStatus* onnx_status = (expr); \
    if (onnx_status != NULL) { \
        const char* msg = g_ort->GetErrorMessage(onnx_status); \
        fprintf(stderr, "\n[ERRO FATAL ONNX]: %s\n\n", msg); \
        g_ort->ReleaseStatus(onnx_status); \
        exit(1); \
    } \
} while (0)

int main() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Falha ao inicializar a ONNX Runtime API.\n");
        return 1;
    }

    OrtEnv* env;
    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "VWW_Inference", &env));

    OrtSessionOptions* session_options;
    CHECK_STATUS(g_ort->CreateSessionOptions(&session_options));
    
    OrtSession* session;
    
    printf("Carregando modelo ONNX...\n");
    CHECK_STATUS(g_ort->CreateSession(env, "model/vww_96_model.onnx", session_options, &session));

    size_t input_tensor_size = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH;
    float* input_tensor_values = (float*)malloc(input_tensor_size * sizeof(float));
    
    FILE* file = fopen("imagem_teste.bin", "rb");
    if (file == NULL) {
        fprintf(stderr, "Erro ao abrir imagem_teste.bin. Rode o script Python primeiro!\n");
        return 1;
    }
    
    size_t elementos_lidos = fread(input_tensor_values, sizeof(float), input_tensor_size, file);
    fclose(file);
    
    if (elementos_lidos != input_tensor_size) {
        fprintf(stderr, "Aviso: O tamanho lido nao bate com o tensor esperado.\n");
    } else {
        printf("Imagem carregada com sucesso do disco!\n");
    }

    OrtMemoryInfo* memory_info;
    CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    int64_t input_shape[] = {BATCH_SIZE, CHANNELS, HEIGHT, WIDTH};
    OrtValue* input_tensor = NULL;
    
    CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, 
        input_tensor_values, 
        input_tensor_size * sizeof(float), 
        input_shape, 4, 
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
        &input_tensor
    ));

    const char* input_names[] = {"input_image"};
    const char* output_names[] = {"prediction"};
    OrtValue* output_tensor = NULL;

    printf("Rodando inferencia no modelo...\n");
    CHECK_STATUS(g_ort->Run(
        session, NULL, 
        input_names, (const OrtValue* const*)&input_tensor, 1, 
        output_names, 1, &output_tensor
    ));

    float* out_arr;
    CHECK_STATUS(g_ort->GetTensorMutableData(output_tensor, (void**)&out_arr));
    
    printf("\n--- Resultados (Logits) ---\n");
    printf("Classe 0 (No Person): %.4f\n", out_arr[0]);
    printf("Classe 1 (Person):    %.4f\n", out_arr[1]);
    
    if (out_arr[1] > out_arr[0]) {
        printf("\n=> PREVISAO FINAL: Pessoa detectada!\n");
    } else {
        printf("\n=> PREVISAO FINAL: Nenhuma pessoa detectada.\n");
    }

    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    free(input_tensor_values);

    return 0;
}
