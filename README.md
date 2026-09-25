# 🐱🐶 Transfer Learning com Deep Learning — Gatos vs Cachorros

[![Notebook Check](https://github.com/SEU_USUARIO/SEU_REPOSITORIO/actions/workflows/validate-notebook.yml/badge.svg)](https://github.com/SEU_USUARIO/SEU_REPOSITORIO/actions/workflows/validate-notebook.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_USUARIO/SEU_REPOSITORIO/blob/main/notebook/transfer_learning_cats_vs_dogs.ipynb)

Projeto desenvolvido como conclusão do **Desafio de Projeto — Transfer Learning**, parte da
trilha de Deep Learning da **Digital Innovation One (DIO)**.

> Autor: **José Wagner Blanco Júnior** — Principal AI Systems Architect | Executive Manager
> Consultoria & Mentoria Blanco — Maringá, PR

---

## 🎯 Objetivo do desafio

Aplicar o método de **Transfer Learning** em uma rede de Deep Learning, em Python, no
ambiente Google Colab, documentando todo o processo técnico e publicando o resultado em
um repositório público no GitHub.

## 🧠 O que é Transfer Learning?

Transfer Learning é a técnica de reaproveitar o conhecimento aprendido por uma rede neural
já treinada em uma tarefa/dataset grande (geralmente o ImageNet, com milhões de imagens) e
adaptá-lo para um novo problema, geralmente com um dataset muito menor. Isso economiza tempo
de treinamento, poder computacional e costuma gerar resultados melhores do que treinar uma
rede do zero com poucos dados.

Neste projeto, o processo é feito em duas etapas clássicas:

1. **Feature Extraction**: o modelo base (`MobileNetV2`, pré-treinado no ImageNet) é
   congelado e usado apenas como extrator de características. Apenas uma nova "cabeça" de
   classificação é treinada sobre ele.
2. **Fine-tuning**: as últimas camadas do modelo base são descongeladas e re-treinadas com
   uma taxa de aprendizado muito baixa, ajustando finamente os pesos ao domínio específico
   do problema (gatos vs cachorros).

## 📦 Dataset

Utilizamos o dataset **[`cats_vs_dogs`](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)**,
disponível nativamente via `tensorflow_datasets`, contendo aproximadamente 23.000 imagens
divididas em duas classes: **gato** e **cachorro**.

> 💡 O pipeline foi construído para ser facilmente adaptado a **qualquer outro par de
> classes** (ex.: fotos próprias, de familiares, de animais de estimação, etc.) — basta
> trocar a fonte dos dados na etapa de carregamento, mantendo o restante do notebook.

Split utilizado:
- 80% treino
- 10% validação
- 10% teste

## 🏗️ Arquitetura utilizada

- **Modelo base:** MobileNetV2 (pré-treinado no ImageNet, `include_top=False`)
- **Cabeça de classificação:** `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(1)` (logit binário)
- **Data augmentation:** flip horizontal + rotação aleatória leve
- **Otimizador:** Adam
  - `1e-4` na fase de feature extraction
  - `1e-5` na fase de fine-tuning (últimas camadas do modelo base descongeladas a partir da camada 100)
- **Loss:** Binary Crossentropy (from logits)

## 📁 Estrutura do repositório

```
.
├── README.md
├── notebook/
│   └── transfer_learning_cats_vs_dogs.ipynb   # Notebook principal (roda no Google Colab)
├── images/
│   ├── training_curves.png                    # Gráficos de acurácia/loss (gerado ao rodar o notebook)
│   └── inference_examples.png                 # Exemplos de inferência (gerado ao rodar o notebook)
└── .github/
    └── workflows/
        └── validate-notebook.yml               # Automação: valida a estrutura do notebook a cada push
```

## ▶️ Como executar

### Opção 1 — Google Colab (recomendado)
Clique no badge **"Open In Colab"** no topo deste README, ou acesse diretamente:

```
https://colab.research.google.com/github/SEU_USUARIO/SEU_REPOSITORIO/blob/main/notebook/transfer_learning_cats_vs_dogs.ipynb
```

Recomenda-se ativar GPU em **Ambiente de execução → Alterar tipo de ambiente de execução → GPU**.

### Opção 2 — Ambiente local
```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO
pip install tensorflow tensorflow-datasets matplotlib numpy jupyter
jupyter notebook notebook/transfer_learning_cats_vs_dogs.ipynb
```

## 📊 Resultados

Após a execução completa (feature extraction + fine-tuning), o modelo atinge alta acurácia
de validação/teste na classificação binária gato vs cachorro. Os gráficos de acurácia/loss
e os exemplos de inferência gerados durante a execução ficam salvos em `images/` (ou na raiz
de execução do Colab) como `training_curves.png` e `inference_examples.png`.

## 🤖 Automação incluída

Este repositório conta com um workflow de **GitHub Actions** (`.github/workflows/validate-notebook.yml`)
que roda automaticamente a cada `push`/`pull request` e valida:
- A integridade estrutural do notebook (JSON válido no formato `.ipynb`);
- A presença das seções obrigatórias (dataset, modelo, treinamento, avaliação).

Isso garante que o notebook entregue nunca fique corrompido ou incompleto no repositório,
sem exigir infraestrutura de treinamento pesada rodando em CI (o treinamento em si continua
sendo feito no Colab, com GPU).

## 🛠️ Tecnologias utilizadas

- Python 3
- TensorFlow / Keras
- TensorFlow Datasets
- Matplotlib / NumPy
- Google Colab
- GitHub Actions

## 📚 Referências

- [Transfer Learning Notebook (MNIST) — ml4a-guides](https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb)
- [Dataset cats_vs_dogs — TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
- [Guia oficial de Transfer Learning — TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
- Desafio de Projeto — Digital Innovation One (DIO)

---

Desenvolvido por **José Wagner Blanco Júnior** · [Consultoria & Mentoria Blanco](https://consultoriablanco.com)
