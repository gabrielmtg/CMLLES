#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/lite/c/c_api.h>

int main() {
    TfLiteModel* model = TfLiteModelCreateFromFile("model/cmapss_lstm.tflite");
    if (model == NULL) {
        fprintf(stderr, "Falha critica ao carregar o modelo LiteRT.\n");
        return -1;
    }

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    if (interpreter == NULL) {
        fprintf(stderr, "Falha critica ao inicializar a API do LiteRT.\n");
        return -1;
    }

    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        fprintf(stderr, "Falha critica ao alocar os tensores.\n");
        return -1;
    }

    size_t input_tensor_size = 1 * 30 * 14;
    float* input_tensor_values = (float*)malloc(input_tensor_size * sizeof(float));
    for (size_t i = 0; i < input_tensor_size; i++) {
        input_tensor_values[i] = 0.5f; 
    }

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    float* in_ptr = (float*)TfLiteTensorData(input_tensor);
    for(size_t i = 0; i < input_tensor_size; i++) {
        in_ptr[i] = input_tensor_values[i];
    }

    printf("Executando inferencia...\n");
    if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
        fprintf(stderr, "Erro durante a inferencia.\n");
        return -1;
    }

    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    float* output_arr = (float*)TfLiteTensorData(output_tensor);
    float rul_em_ciclos = output_arr[0] * 125.0f;

    printf("========================================\n");
    printf("Vida Util Restante Prevista (RUL): %.2f ciclos\n", rul_em_ciclos);
    printf("========================================\n");

    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
    free(input_tensor_values);

    return 0;
}
