from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from argparse import ArgumentParser
import sys
from methods.utils import get_base_parser, get_dataset, get_X_y, get_filename
import logging

heuristic_metrics = ['precision', 'accuracy', 'recall', 'f-measure']

def parse_args(argv):
    base_parser = get_base_parser()
    parser = ArgumentParser(parents=[base_parser])
    parser.add_argument(
        '-i', '--increment',
        help = 'Increment. Default: 20',
        type = int,
        default = 20)
    parser.add_argument(
        '-f',
        metavar = 'LIST',
        help = 'List of number of features to select. If provided, Increment is ignored. Usage example: -f="10,50,150,400"',
        type = str,
        default = "")
    parser.add_argument(
        '-k', '--n-folds',
        help = 'Number of folds to use in k-fold cross validation. Default: 10.',
        type = int,
        default = 10)
    parser.add_argument(
        '-t', '--threshold',
        help = 'Threshold to choose the best dataset of selected features based on --heuristic-metric. Default: 0.95.',
        type = float,
        default = 0.95)
    parser.add_argument(
        '-s', '--decrement-step',
        help = "If the heuristic couldn't find't the dataset of features selected, try again decreasing the threshold by this amount. Default: 0.05.",
        type = float,
        default = 0.05)
    parser.add_argument(
        '-m', '--heuristic-metric',
        help = f"Metric to base the choice of the best dataset of selected features. Options: {','.join(heuristic_metrics)}. Default: 'recall'.",
        choices=heuristic_metrics,
        default = 'recall')
    parser.add_argument('--feature-selection-only', action='store_true',
        help="If set, the experiment is constrained to the feature selection phase only. The program always returns the best K features, where K is the maximum value in the features list.")
    args = parser.parse_args(argv)
    return args

def run_experiment(X, y, classifiers, is_feature_selection_only = False,
                   score_functions=[chi2, f_classif],
                   n_folds=10,
                   k_increment=20,
                   k_list=[]):
    """
    Esta função implementa um experimento de classificação binária usando validação cruzada e seleção de características.
    Os "classifiers" devem implementar as funções "fit" e "predict", como as funções do Scikit-learn.
    Se o parâmetro "k_list" for uma lista não vazia, então ele será usado como a lista das quantidades de características a serem selecionadas.
    """
    global logger_rfg
    results = []
    feature_rankings = {}
    if(len(k_list) > 0):
        k_values = k_list
    else:
        k_values = range(1, X.shape[1], k_increment)
    for k in k_values:
        if(k > X.shape[1]):
            logger_rfg.warning(f"Skipping K = {k}, since it's greater than the number of features available ({X.shape[1]})")
            continue

        logger_rfg.info("K = %s" % k)
        for score_function in score_functions:
            if(k == max(k_values)):
                selector = SelectKBest(score_func=score_function, k=k).fit(X, y)
                X_selected = X.iloc[:, selector.get_support(indices=True)].copy()
                feature_scores_sorted = pd.DataFrame(list(zip(X_selected.columns.values.tolist(), selector.scores_)), columns= ['features','score']).sort_values(by = ['score'], ascending=False)
                X_selected_sorted = X_selected.loc[:, list(feature_scores_sorted['features'])]
                X_selected_sorted['class'] = y
                feature_rankings[score_function.__name__] = X_selected_sorted
                if(X_selected.shape[1] == 1):
                    logger_rfg.warning("Nenhuma caracteristica selecionada")
            if(is_feature_selection_only):
                continue
            X_selected = SelectKBest(score_func=score_function, k=k).fit_transform(X, y)
            kf = KFold(n_splits=n_folds, random_state=256, shuffle=True)
            fold = 0
            for train_index, test_index in kf.split(X_selected):
                X_train, X_test = X_selected[train_index], X_selected[test_index]
                y_train, y_test = y[train_index], y[test_index]

                for classifier_name, classifier in classifiers.items():
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division = 0)
                    results.append({'n_fold': fold,
                                    'k': k,
                                    'score_function':score_function.__name__,
                                    'algorithm': classifier_name,
                                    'accuracy': report['accuracy'],
                                    'precision': report['macro avg']['precision'],
                                    'recall': report['macro avg']['recall'],
                                    'f-measure': report['macro avg']['f1-score']
                                })
                fold += 1

    return pd.DataFrame(results), feature_rankings

def get_best_result(results, threshold=0.95, heuristic_metric='recall', decrement_step=0.05):
    averages = results.groupby(['k','score_function']).mean().drop(columns=['n_fold'])
    maximun_score = max(averages.max())
    th = threshold
    while th > 0:
        for k, score_function in averages.index:
            if(averages.loc[(k, score_function)][heuristic_metric] > th * maximun_score):
                return (k, score_function)
        th -= decrement_step

    logger_rfg.error("Não foi possível encontrar o dataset de características selecionadas, tente novamente variando o --heuristic_metric e/ou --threshold")

def get_best_features_dataset(best_result, feature_rankings, class_column):
    k, score_function = best_result
    X = feature_rankings[score_function].drop(columns=[class_column])
    y = feature_rankings[score_function][class_column]
    X_selected = X.iloc[:, :k]
    X_selected = X_selected.join(y)

    return X_selected

def main():
    global logger_rfg
    parsed_args = parse_args(sys.argv[1:])
    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    k_list = [int(value) for value in parsed_args.f.split(",")] if parsed_args.f != "" else []

    classifiers = {
        'RandomForest': RandomForestClassifier(),
    }

    logger_rfg.info("Executando experimento")
    results, feature_rankings = run_experiment(
        X, y,
        classifiers,
        n_folds = parsed_args.n_folds,
        k_increment = parsed_args.increment,
        k_list=k_list,
        is_feature_selection_only=parsed_args.feature_selection_only
    )

    filename = get_filename(parsed_args.output_file, prefix=parsed_args.output_prefix)
    logger_rfg.info("Selecionando as melhores caracteristicas")
    get_best_features_dataset(get_best_result(results, parsed_args.threshold, parsed_args.heuristic_metric, parsed_args.decrement_step), feature_rankings, parsed_args.class_column).to_csv(filename, index=False)

if __name__ == '__main__':
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_rfg
    logger_rfg = logging.getLogger('RFG')
    logger_rfg.setLevel(logging.INFO)
    main()
