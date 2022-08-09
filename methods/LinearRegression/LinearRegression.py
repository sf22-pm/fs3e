import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from argparse import ArgumentParser
from methods.utils import get_base_parser, get_dataset, get_X_y, get_filename
import time
import logging

def parse_args(argv):
    parser = ArgumentParser(parents=[get_base_parser()])
    args = parser.parse_args(argv)
    return args

# FOLDERS
def KFolders():
    kf = KFold(n_splits=10, shuffle=False) # set a divisão em 10 folds
    kf.get_n_splits(X) # retorna o número de iterações divididas na validação cruzada
    return (kf)

# FILTRAGEM DAS FEATURES
def FilterFeatures(ft_names, coef):
  i=0
  n1 = len(coef)
  features_list_to_delete = []

  for i in range(0, n1):
     if coef[i] < 0.1 and coef[i] > -0.1:
       features_list_to_delete.append(ft_names[i])
  return (features_list_to_delete)

# APLICAÇÃO DA REGRESSÃO LINEAR
def LinearR():
    for train_index, test_index in KFolders().split(X):
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)
        coef_in = model.coef_
        ft_to_delete = FilterFeatures(features_names, coef_in)
        fold_ft_num.append(len(ft_to_delete))
        fold_ft_to_delete.append(ft_to_delete)

        for ft in ft_to_delete:
            index = list(features_names).index(ft)
            #print(index)
            fold_count[index]-=1
    return(fold_ft_num, fold_ft_to_delete)

# VALOR MÁXIMO DE FEATURES PARA EXCLUSÃO
def MaxValue():
    max_value = None
    index = None

    for idx, num in enumerate(fold_ft_num):
        if max_value is None or num > max_value:
            max_value = num
            index = idx
    #print('value', max_value, 'index', index)
    #print(fold_ft_to_delete[index])
    return(index)

# NOVO DATASET
def NewDataset():
    new_X = X.drop(columns=fold_ft_to_delete[MaxValue()])
    #print(new_X)
    df2 = pd.DataFrame(new_X)
    df2['class'] = y
    return df2

if __name__=="__main__":
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_lr
    logger_lr = logging.getLogger('LinearRegression')
    logger_lr.setLevel(logging.INFO)

    args = parse_args(sys.argv[1:])
    X, y = get_X_y(args, get_dataset(args))

    # VETORES AUXILIARES
    features_names = np.array(X.columns.values.tolist())
    fold_count = [10] * len(features_names)
    fold_ft_num = []
    fold_ft_to_delete = []

    # CHAMADA DAS FUNÇÕES
    logger_lr.info("SETANDO KFOLDS EM 10")
    KFolders()
    time.sleep(3)
    logger_lr.info("APLICANDO A REGRESSÃO LINEAR")
    LinearR()
    MaxValue()
    time.sleep(3)
    logger_lr.info("GERANDO NOVO DATASET...")

    NewDataset().to_csv(get_filename(args.output_file, prefix=args.output_prefix), index=False)
    logger_lr.info("DATASET GERADO COM SUCESSO")
