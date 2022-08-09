# RFG

Implementação do experimento do paper [_Automated Malware Detection in Mobile App Stores Based on Robust Feature Generation_](https://doi.org/10.3390/electronics9030435) (RFG). O experimento é composto pelas seguintes etapas:

1. Feature selection por meio do Chi-quadrado e o ANOVA;
    - A quantidade `k` de características varia de forma incremental. Por padrão, incia-se com `k = 10` e segue-se incrementando em 20 até a quantidade total de features do dataset. Você pode definir o valor do incremento ou suprir uma lista com os valores de `k` a serem utilizados.
2. Treino e teste dos seguintes modelos através de validação cruzada _K-fold_: Naive Bayes,
KNN, Random Forest, J48, Sequential Minimal Optimization (SMO), Logistic Regression, AdaBoost decision-stump, Random Committee, JRip e Simple Logistics.

A nossa implementação considera apenas o modelo do Random Forest na segunda etapa, pois ele apresentou os melhores resultados no paper do RFG.

### Dependências

O `rfg.py` foi desenvolvido e testado no sistema operacional Ubuntu 20.04 LTS, com as seguintes versões das linguagens Python e Java:

- Python3 versão 3.8.10;

### Instalação

Após clonar o repositório, instale os seguintes pacotes com o pip: 
```
pip3 install pandas scikit-learn
```
## Como rodar

Mude para o diretório raiz deste repositório (i.e.: `cd feature_selection`).

Para rodar o experimento do RFG sobre algum dataset, use o módulo do script `rfg.py`, como no exemplo:
```
python3 -m methods.RFG.rfg -d Drebin215.csv
```

**IMPORTANTE:** o módulo deve ser executado a partir do diretório pai do diretório `methods`. Caso contrário você receberá o seguinte erro: `ModuleNotFoundError: No module named 'utils'`. Isso acontece devido ao funcionamento de módulos em Python.

## Arquivos de saída

- O `rfg.py` sempre exporta dois datasets `.csv` referentes às melhores features selecionadas com métodos `chi2` (Chi-quadrado) e com o `f_classif` (ANOVA);

- Se a opção `--feature-selection-only` não for passada (saiba mais em "Detalhes de uso" a seguir), o `rfg.py` também irá exportar um arquivo `.csv` com os resultados da avaliação do experimento. (Você pode usar o notebook `RFG_plot_results.ipynb` para visualizar os resultados)

## Detalhes de uso

```
usage: rfg.py [-h] -d DATASET [-i INCREMENT] [-f LIST] [-k N_FOLDS] [-t THRESHOLD] 
              [-n N_SAMPLES] [--feature-selection-only]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset (csv file). It should be already preprocessed, comma separated, with the last feature being the class.
  -i INCREMENT, --increment INCREMENT
                        Increment. Default: 20
  -f LIST               List of number of features to select. If provided, Increment is ignored. Usage example: -f="10,50,150,400"
  -k N_FOLDS, --n-folds N_FOLDS
                        Number of folds to use in k-fold cross validation. Default: 10.
  -t THRESHOLD, --prediction-threshold THRESHOLD
                        Prediction threshold for Weka classifiers. Default: 0.6
  -n N_SAMPLES, --n-samples N_SAMPLES
                        Use a subset of n samples from the dataset. RFG uses the whole dataset by default.
  --feature-selection-only
                        If set, the experiment is constrained to the feature selection phase only. The program always returns the best K features, where K is the maximum value in the features list.
```