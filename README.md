<div align="center">

## Masked Face Recognition Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ATRRCVrzh37h7hFUbNL8BefTb4DW39Uu?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/PUCRS-Escola_Politécnica-maroon.svg)](https://www.pucrs.br/)

*Projeto prático e teórico desenvolvido para a disciplina de **Introdução à Visão Computacional** na **PUCRS**, abordando os desafios do reconhecimento facial na era da COVID-19 e a implementação de um pipeline de busca vetorial.*

---

</div>

## Visão Geral

Com a pandemia da COVID-19, o uso maciço de máscaras faciais introduziu um desafio crítico para os sistemas de Visão Computacional. A oclusão da metade inferior do rosto atua como **ruído estruturado**, fazendo com que modelos tradicionais percam sinal útil e ganhem informações irrelevantes (texturas, dobras e estampas da máscara).

Este repositório analisa os avanços científicos apontados pela literatura para contornar essa limitação e demonstra uma **implementação prática de um pipeline de reconhecimento facial baseado em embeddings de 128 dimensões e bancos de dados vetoriais**.

---

## Síntese da Pesquisa Teórica

Baseado no artigo científico *"A survey on computer vision based human analysis in the COVID-19 era"*:

### 1. Detecção de Rosto e Máscara
A estratégia consolida-se em um pipeline de duas etapas (detecção + classificação):
* **Detectores:** Adaptação de modelos como MTCNN, RetinaFace, SSD, Faster R-CNN e variações da YOLO (v2, v3 e v5).
* **Modelos Notáveis:** 
  * `SSDMNV2`: Combinação de SSD com MobileNetV2 para uso leve em dispositivos móveis.
  * `SE-YOLOv3`: Uso de blocos Squeeze-and-Excitation e *focal loss*.
  * `SRCNet`: Aplicação de super-resolução em imagens com menos de 150×150 px.

### 2. Estratégias para Reconhecimento Facial com Máscara
1. **In-painting:** Apaga a máscara e reconstrói a parte inferior da face digitalmente.
2. **Template Unmasking:** Transforma o vetor de características (*template*) para que se comporte como um rosto sem máscara.
3. **Otimização do Modelo:** Retreinamento de redes usando funções de perda adaptadas (ArcFace multitarefa, *triplet loss* e maiores margens).
4. **Reconhecimento Periocular:** Foco exclusivo no vetor das regiões dos olhos e sobrancelhas.

### 3. Datasets Analisados
* **Bases Reais:** `MAFA` (30.811 imagens com oclusões), `ISL-UFMD/ISL-UFHD` (diversidade étnica e variações de uso correto/incorreto) e `BAFMD`.
* **Bases Sintéticas:** `MaskedFace-Net` (137k imagens) e `MS1MV2-Masked` (~57,5M de imagens simuladas).

---

## 🛠️ Pipeline Prático Implementado

O notebook implementa um fluxo completo de identificação e busca por similaridade vetorial:

```
┌────────────────────┐
│ Imagem de Entrada  │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Detector de Face  │ ──► Localiza a ROI (Region of Interest)
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Rede Neural CNN   │ ──► Converte a face em um Embedding (Vetor de 128D)
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Banco Vetorial   │ ──► Realiza busca por similaridade (1-NN / Cosseno / Euclidiana)
└────────────────────┘
```

### Resultados dos Testes Práticos
* **Métrica de Distância Euclidiana (`face_recognition`):** Limiar oficial menor que 0.6 indica a mesma pessoa.
* **ChromaDB:** Identificou com precisão a pessoa correspondente no banco vetorial (*distância de 0.119*).
* **FaceDB:** Classificou e reconheceu a face com **88% de confiança**.

---

## Estrutura do Repositório

```
masked-face-recognition-pipeline/
├── data/                           # Imagens de exemplo utilizadas no teste
├── notebooks/
│   └── reconhecimento_facial.ipynb # Notebook principal do Colab
├── .gitignore                      # Arquivo para ignorar os bancos locais gerados
├── LICENSE                         # Licença do repositório
└── README.md                       # Documentação do projeto
```

## Como Executar

### Opção 1: Executar Direto no Google Colab
Clique no selo no topo deste README ou acesse o notebook no Colab para rodar o código na nuvem.

### Opção 2: Executar Localmente

1. **Clone o repositório:**
   ```
   git clone [https://github.com/lucasnk1/masked-face-recognition-pipeline.git](https://github.com/lucasnk1/masked-face-recognition-pipeline.git)
   cd masked-face-recognition-pipeline
   ```

2. **Instale as dependências necessárias:**
   ```
   pip install face_recognition chromadb facedb gdown
   ```
3. **Inicie o Jupyter Notebook:**
   ```
   jupyter notebook notebooks/reconhecimento_facial.ipynb
   ```

## Stacks utilizadas

* **Linguagem:** Python 3.10+

* **Visão Computacional & Embeddings:** face_recognition, Pillow, NumPy

* **Bancos de Dados Vetoriais:** ChromaDB, FaceDB

* **Download de Dados:** gdown

## Autor

**Lucas Leuck de Oliveira**

* **Instituição:** Pontifícia Universidade Católica do Rio Grande do Sul (PUCRS)

* **Curso:** Escola Politécnica / Introdução à Visão Computacional

* **Professora:** Daniela Oliveira Ferreira do Amaral

   
