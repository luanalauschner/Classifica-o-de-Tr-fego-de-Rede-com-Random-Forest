# Classificação de Tráfego de Rede com Random Forest

Este projeto utiliza aprendizado de máquina para classificar fluxos de tráfego de rede do dataset CICIDS2017.

O modelo utilizado é o **Random Forest Classifier**, implementado com a biblioteca Scikit-learn.

O programa pode executar duas tarefas:

- **Classificação binária:** identifica se o fluxo é `BENIGN` ou `ATTACK`;
- **Classificação multiclasse:** identifica se o fluxo é benigno ou o tipo específico de ataque registrado no dataset.

O código utiliza características estatísticas dos fluxos, como duração, quantidade de pacotes, volume de bytes, tamanho dos pacotes e intervalos de tempo.

---

## Estrutura do projeto

A pasta do projeto deve seguir esta organização:

```text
trabalho_cicids2017_ml/
│
├── dados/
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   └── Wednesday-workingHours.pcap_ISCX.csv
│
├── cicids2017_ml.py
├── executar_experimento.py
├── requirements.txt
└── README.md
```

O programa procura automaticamente os arquivos `.csv` dentro da pasta `dados`.

Não é necessário informar o caminho dos dados no comando de execução.

---

## Requisitos

Recomenda-se utilizar Python 3.10 ou superior.

As principais bibliotecas utilizadas são:

- Pandas;
- NumPy;
- Scikit-learn;
- Matplotlib;
- Joblib;
- Tabulate;
- SciPy.

Para instalar as dependências, execute:

```powershell
pip install pandas numpy scikit-learn matplotlib joblib tabulate scipy
```

Caso exista um arquivo `requirements.txt`, também é possível executar:

```powershell
pip install -r requirements.txt
```

---

## Dataset

O projeto utiliza o **CICIDS2017**, um dataset público de tráfego de rede que contém fluxos previamente rotulados.

Cada linha dos arquivos CSV representa um fluxo de comunicação e possui:

- características estatísticas do fluxo;
- uma coluna chamada `Label`, que informa a classificação verdadeira;
- registros benignos;
- registros associados a diferentes tipos de ataque.

Alguns exemplos de rótulos encontrados no dataset são:

```text
BENIGN
DDoS
PortScan
DoS Hulk
DoS GoldenEye
Bot
FTP-Patator
SSH-Patator
Infiltration
Heartbleed
Web Attack Brute Force
Web Attack SQL Injection
Web Attack XSS
```

---

## Classificação binária

Na tarefa binária, todos os ataques são agrupados em uma única classe chamada `ATTACK`.

A transformação ocorre da seguinte forma:

```text
Rótulo original                 Classe utilizada

BENIGN                          BENIGN
DDoS                            ATTACK
PortScan                        ATTACK
DoS Hulk                        ATTACK
Bot                             ATTACK
FTP-Patator                     ATTACK
Qualquer outro ataque           ATTACK
```

Para executar a tarefa binária:

```powershell
python executar_experimento.py --tarefa binaria --saida "resultados_binarios" --cv 10 --max-por-classe 100000
```

Esse comando utiliza, no máximo:

```text
100.000 registros BENIGN
100.000 registros ATTACK
```

Total máximo:

```text
200.000 registros
```

---

## Classificação multiclasse

Na tarefa multiclasse, os tipos de ataque permanecem separados.

Exemplo:

```text
BENIGN
DDoS
PortScan
DoS Hulk
Bot
FTP-Patator
```

Para executar a tarefa multiclasse:

```powershell
python executar_experimento.py --tarefa multiclasse --saida "resultados_multiclasse" --cv 10 --max-por-classe 20000
```

Nesse caso, o programa utiliza até 20.000 registros de cada classe.

Algumas classes podem ter menos registros porque o CICIDS2017 possui poucos exemplos de determinados ataques.

---

## Preparação dos dados

Durante a preparação, o código:

1. localiza a coluna `Label`;
2. corrige diferenças de escrita nos rótulos;
3. cria a classificação binária ou multiclasse;
4. remove registros sem rótulo válido;
5. separa os atributos da resposta correta;
6. remove a coluna `Label` do conjunto de atributos;
7. remove o nome do arquivo de origem;
8. remove colunas duplicadas;
9. converte os atributos para valores numéricos;
10. substitui valores infinitos por valores ausentes;
11. remove colunas completamente vazias;
12. remove colunas constantes.

A coluna `Label` é usada apenas para ensinar e avaliar o modelo.

Ela não é fornecida ao Random Forest como uma característica de entrada.

---

## Treinamento do modelo

O modelo utilizado é:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
)
```

O Random Forest possui 100 árvores de decisão.

Durante o treinamento, o modelo estuda a relação entre:

```text
Características estatísticas do fluxo
                    ↓
Classe registrada no Label
```

Depois do treinamento, o modelo recebe somente os atributos e produz uma previsão:

```text
BENIGN
ATTACK
```

ou, na tarefa multiclasse:

```text
BENIGN
DDoS
PortScan
Bot
...
```

---

## Tratamento de valores ausentes

O projeto utiliza um pipeline contendo:

```text
SimpleImputer
      ↓
Random Forest
```

O `SimpleImputer` substitui valores ausentes pela mediana da respectiva coluna.

Essa transformação é calculada somente com os dados de treinamento, evitando o uso indevido de informações do conjunto de teste.

---

## Validação cruzada

O código utiliza validação cruzada estratificada.

A estratificação procura preservar a proporção das classes em cada fold.

As métricas calculadas incluem:

- acurácia;
- precisão macro;
- recall macro;
- F1 macro;
- F1 ponderado;
- tempo de treinamento.

Na tarefa binária, também são calculadas:

- ROC-AUC;
- Average Precision.

---

## Conjunto de teste final

Além da validação cruzada, o programa separa um conjunto de teste final.

Por padrão:

```text
70% → treinamento
30% → teste
```

O conjunto de teste não é utilizado durante o treinamento final.

Ele serve para avaliar como o modelo se comporta diante de registros que não foram utilizados para ajustar o Random Forest.

---

## Arquivos gerados

Os resultados são salvos na pasta indicada em `--saida`.

### `configuracao.json`

Contém os parâmetros utilizados na execução.

---

### `distribuicao_classes.csv`

Contém a quantidade de registros de cada classe.

---

### `distribuicao_classes.png`

Gráfico da distribuição das classes.

---

### `distribuicao_por_arquivo.csv`

Apresenta a quantidade de registros por arquivo CSV e por classe.

---

### `metricas_cv_por_fold.csv`

Contém as métricas obtidas em cada fold da validação cruzada.

---

### `resumo_metricas_cv.csv`

Contém as médias das métricas da validação cruzada.

---

### `boxplot_acuracia_cv.png`

Mostra a distribuição da acurácia obtida nos folds.

---

### `metricas_holdout.csv`

Contém as métricas obtidas no conjunto de teste final.

---

### `relatorio_classificacao_RF.csv`

Apresenta precisão, recall, F1-score e quantidade de exemplos para cada classe.

---

### `matriz_confusao_RF.png`

Mostra a comparação entre as classes verdadeiras e as classes previstas.

Na tarefa binária, permite identificar:

- verdadeiros positivos;
- verdadeiros negativos;
- falsos positivos;
- falsos negativos.

---

### `curva_roc_RF.png`

Gerada somente na tarefa binária.

Apresenta a relação entre a taxa de verdadeiros positivos e a taxa de falsos positivos.

---

### `curva_pr_RF.png`

Gerada somente na tarefa binária.

Apresenta a relação entre precisão e recall.

---

### `importancia_atributos_rf.csv`

Apresenta todos os atributos utilizados pelo Random Forest, ordenados por importância.

---

### `importancia_atributos_rf.png`

Mostra os 20 atributos mais importantes para as decisões do modelo.

A importância indica o quanto cada atributo contribuiu para as divisões realizadas pelas árvores.

Ela não representa necessariamente uma relação de causa e efeito.

---

### `modelo_random_forest.joblib`

Armazena o modelo treinado.

O arquivo contém:

- o `SimpleImputer`;
- o Random Forest;
- os nomes e a ordem dos atributos;
- a tarefa utilizada;
- os nomes das classes.

Esse arquivo permite reutilizar o modelo sem executar novamente todo o treinamento.

---

### `resumo_resultados.md`

Contém um resumo textual das métricas e das configurações do experimento.

---

### `resumo_execucao.json`

Apresenta no formato JSON as principais informações da execução.

Exemplo:

```json
{
  "task": "binaria",
  "rows": 200000,
  "features": 69,
  "model": "Random Forest",
  "cv_f1_macro": 0.9983699995734497,
  "cv_accuracy": 0.9983700000000001,
  "output_dir": "resultados_binarios"
}
```

---