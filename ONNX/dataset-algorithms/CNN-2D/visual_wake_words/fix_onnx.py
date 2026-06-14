import onnx

onnx_path = "model/vww_96_model.onnx"
print(f"Carregando {onnx_path}...")

model = onnx.load(onnx_path)

model.ir_version = 9

onnx.save(model, onnx_path)

print("Versão alterada para 9 e arquivo salvo com sucesso!")
