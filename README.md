# CMLLES

Benchmark de inferência de redes neurais em sistemas embarcados, comparando seis bibliotecas de ML em uma Raspberry Pi 4B via porta serial UART.

**Frameworks:** FANN · Genann · NCNN · LiteRT · ONNX Runtime · CMSIS-NN  
**Algoritmos:** MLP · Autoencoder · CNN-2D · LSTM  
**Plataforma alvo:** Raspberry Pi 4B (Cortex-A72, AArch64)

---

## Pré-requisitos

**No PC (Ubuntu/Debian):**

```bash
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
                 cmake make python3 python3-pip minicom
pip install torch torchvision scikit-learn pandas numpy
```

**Datasets** (colocar em `datasets/`):
| Algoritmo | Dataset |
|---|---|
| MLP Iris | `datasets/MLP/iris/iris.data` |
| MLP IAES | `datasets/MLP/IAES-dataset/` |
| Autoencoder | `datasets/Autoencoder/NSL_KDD_Dataset/` |
| CNN-2D | `datasets/CNN-2D/coco/` (imagens + anotações VWW) |
| LSTM | `datasets/LSTM/CMAPSSData/` |

---

## Passo 1 — Treinar os modelos

```bash
make train
```

Treina todos os frameworks e algoritmos em sequência. Os modelos são salvos em `<framework>/benchmarks/<algoritmo>/model/`.

> O treino do FANN pode demorar mais que os outros por usar RPROP full-batch sem vetorização.

---

## Passo 2 — Gerar a imagem do SD card

```bash
./scripts/build.sh
```

O script faz automaticamente:
1. Baixa o Raspberry Pi OS Lite (arm64, Bookworm) se necessário
2. Cross-compila as bibliotecas ML para AArch64
3. Cross-compila todos os benchmarks
4. Monta a imagem e injeta os executáveis em `/usr/bin/cmlles_tests/`
5. Copia os arquivos `model/` de cada framework
6. Habilita o UART serial (`enable_uart=1`, `dtoverlay=disable-bt`)
7. Instala o serviço systemd que executa os benchmarks automaticamente no boot

> Se libs e benchmarks já estiverem compilados, o script pula as etapas 2 e 3 automaticamente.

A imagem final fica em:
```
output_rpi_image/images/cmlles-raspios.img
```

---

## Passo 3 — Gravar no SD card

```bash
./scripts/deploy_to_sdcard.sh /dev/sdX   # substitua sdX pelo seu dispositivo
```

Para identificar o dispositivo correto, compare o `lsblk` antes e depois de inserir o SD card:

```bash
lsblk -d -o NAME,SIZE,MODEL,TRAN | grep -v loop
```

---

## Passo 4 — Conectar o cabo serial UART

| Adaptador USB-Serial | RPi 4B (GPIO) |
|---|---|
| TX | Pin 10 — RXD (GPIO 15) |
| RX | Pin 8  — TXD (GPIO 14) |
| GND | Pin 6  — GND |

> Não conecte o VCC do adaptador — a alimentação vem pela fonte USB-C do RPi.

---

## Passo 5 — Abrir o minicom

```bash
minicom -D /dev/ttyUSB0 -b 115200
```

Se não souber a porta: `ls /dev/ttyUSB* /dev/ttyACM*` antes e depois de conectar o cabo.

---

## Passo 6 — Ligar a Raspberry Pi

Insira o SD card e ligue a alimentação. O boot do sistema aparece no minicom em cerca de 20–40 segundos. Após o boot, o serviço `cmlles-bench.service` dispara automaticamente e os resultados aparecem progressivamente no terminal.

Para ver os logs após o boot via shell:

```bash
journalctl -u cmlles-bench.service
```

---

## Estrutura do projeto

```
<FRAMEWORK>/benchmarks/<ALGORITMO>/
    train.py              # treina e exporta o modelo
    benchmark_<algo>.c    # executa 1000 inferências, salva CSV de latências
    Makefile              # targets: train, all (compila benchmark), clean
    model/                # modelos treinados, test_data.txt, training_metrics.json
    latencies/            # gerado no RPi: latencies_<label>.csv (1000 linhas)
```

**FANN / Genann:** usam `prepare_data.py` + `train_models.c` em vez de `train.py`.  
**CMSIS-NN:** `train.py` quantiza para INT8 e gera `model/*_weights.h` + `model/*_params.h`.

---

## Passo 7 — Coletar os resultados

Após os benchmarks terminarem no RPi (serviço com status `exited`), desligue o RPi, insira o SD card no PC e rode em sequência:

```bash
# 1. Copia cmlles_results/ e latencies_rpi/ do SD card → results/
./scripts/collect/collect_from_sdcard.sh

# 2. Agrega PMU por inferência (1000 medições individuais) → results/per_inference_perf_summary.csv
./scripts/collect/collect_per_inference_perf.sh

# 3. Parseia perf stat externo (totais por benchmark) → results/perf_metrics_all.csv
./scripts/collect/collect_perf_metrics.sh

# 4. Métricas de treinamento (training_metrics.json) → results/training_metrics_all.csv
./scripts/collect/collect_training_metrics.sh

# 5. Tamanho dos binários (size, objdump) → results/binary_metrics.csv
./scripts/collect/collect_binary_metrics.sh
```

Todos os CSVs gerados ficam em `results/`. As latências individuais (1000 linhas por benchmark) ficam em `results/latencies_rpi/`.

---

## Estrutura de resultados

```
results/
├── latencies_rpi/                    # 1000 latências por benchmark (cycles, instructions, L1, etc.)
├── cmlles_results/                   # saída bruta do perf stat por benchmark
├── per_inference_perf_summary.csv    # sumário PMU por inferência (mean, std, p95, p99)
├── perf_metrics_all.csv              # totais do perf stat por benchmark
├── training_metrics_all.csv          # métricas de treinamento de todos os frameworks
└── binary_metrics.csv                # tamanho de texto/dados/bss por executável
```
