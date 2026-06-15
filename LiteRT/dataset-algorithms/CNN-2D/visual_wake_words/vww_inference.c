#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/lite/c/c_api.h>

#define BATCH_SIZE 1
#define CHANNELS 3
#define HEIGHT 96
#define WIDTH 96

int main() {
    TfLiteModel* model = TfLiteModelCreateFromFile("model/vww_96_model.tflite");
    if (model == NULL) {
        fprintf(stderr, "Falha ao carregar modelo TFLite.\n");
        return 1;
    }

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        fprintf(stderr, "Falha ao alocar tensores.\n");
        return 1;
    }

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

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    float* in_ptr = (float*)TfLiteTensorData(input_tensor);
    for(size_t i = 0; i < input_tensor_size; i++) {
        in_ptr[i] = input_tensor_values[i];
    }

    printf("Rodando inferencia no modelo...\n");
    if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
        fprintf(stderr, "Falha ao rodar inferencia.\n");
        return 1;
    }

    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    float* out_arr = (float*)TfLiteTensorData(output_tensor);
    
    printf("\n--- Resultados (Logits) ---\n");
    printf("Classe 0 (No Person): %.4f\n", out_arr[0]);
    printf("Classe 1 (Person):    %.4f\n", out_arr[1]);
    
    if (out_arr[1] > out_arr[0]) {
        printf("\n=> PREVISAO FINAL: Pessoa detectada!\n");
    } else {
        printf("\n=> PREVISAO FINAL: Nenhuma pessoa detectada.\n");
    }

    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
    free(input_tensor_values);

    return 0;
}
