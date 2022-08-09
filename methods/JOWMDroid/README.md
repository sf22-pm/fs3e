# JOWMDroid
Espaço para a reprodução do trabalho "[JOWMDroid: Android malware detection based on feature weighting with joint optimization of weight-mapping and classifier parameters](https://www.sciencedirect.com/science/article/pii/S016740482030359X)".

## Como instalar

```
## 1) Clone o respositório:
git clone https://github.com/Malware-Hunter/feature_selection.git

## 2) Instale as dependências:
pip install numpy scikit-learn scipy pandas
```

## Como rodar

Mude para o diretório raiz deste repositório (i.e.: `cd feature_selection`).

Para rodar o experimento sobre algum dataset (e.g. `data.csv`) execute o seguinte comando:

```
python3 -m methods.JOWMDroid.JOWMDroid -d data.csv
```

Note: o JOWMDroid assume que o dataset já está pré-processado, conforme consta na seção a seguir.

## Arquivos de saída

- O `JOWMDroid.py` sempre gera um dataset com as features selecionadas.

- Se a opção `--feature-selection-only` não for passada, então os resultados do experimento são exportados também. O nome do arquivo pode ser especificado com a opção `--output-file`. Para mais detalhes, veja os Detalhes de uso a seguir.

## Detalhes de uso
```
usage: JOWMDroid.py [-h] -d DATASET [--sep SEPARATOR] [-c CLASS_COLUMN] [-n N_SAMPLES] [-o OUTPUT_FILE] [--exclude-hyperparameter] [-m LIST] [-t MI_THRESHOLD]
                    [--train-size TRAIN_SIZE] [--cv INT] [--feature-selection-only]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset (csv file). It should be already preprocessed.
  --sep SEPARATOR       Dataset feature separator. Default: ","
  -c CLASS_COLUMN, --class-column CLASS_COLUMN
                        Name of the class column. Default: "class"
  -n N_SAMPLES, --n-samples N_SAMPLES
                        Use a subset of n samples from the dataset. By default, all samples are used.
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output file name. Default: results.csv
  --exclude-hyperparameter
                        If set, the ML hyperparameter will be excluded in the Differential Evolution. By default it's included
  -m LIST, --mapping-functions LIST
                        List of mapping functions to use. Default: "power, exponential, logarithmic, hyperbolic, S_curve"
  -t MI_THRESHOLD, --mi-threshold MI_THRESHOLD
                        Threshold to select features with Mutual Information. Default: 0.05. Only features with score greater than or equal to this value will be selected
  --train-size TRAIN_SIZE
                        Proportion of samples to use for train. Default: 0.8
  --cv INT              Number of folds to use in cross validation. Default: 5
  --feature-selection-only
                        If set, the experiment is constrained to the feature selection phase only.
```