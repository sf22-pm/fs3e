import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy.optimize import differential_evolution
import timeit
import argparse
import sys
import inspect
from methods.utils import get_base_parser, get_dataset, get_X_y, get_filename
import logging

def parse_args():
    parser = argparse.ArgumentParser(parents=[get_base_parser()])
    parser.add_argument('--exclude-hyperparameter', action='store_true',
        help="If set, the ML hyperparameter will be excluded in the Differential Evolution. By default it's included")
    parser.add_argument( '-m', '--mapping-functions', metavar = 'LIST', type = str,
        default = "power, exponential, logarithmic, hyperbolic, S_curve",
        help = 'List of mapping functions to use. Default: "power, exponential, logarithmic, hyperbolic, S_curve"')
    parser.add_argument( '-t', '--mi-threshold', type = float, default = 0.2,
        help = 'Threshold to select features with Mutual Information. Default: 0.2. Only features with score greater than or equal to this value will be selected')
    parser.add_argument('--train-size', type = float, default = 0.8,
        help = 'Proportion of samples to use for train. Default: 0.8')
    parser.add_argument('--cv', metavar = 'INT', type = int, default = 5,
        help="Number of folds to use in cross validation. Default: 5")
    parser.add_argument('--feature-selection-only', action='store_true',
        help="If set, the experiment is constrained to the feature selection phase only.")

    return parser.parse_args(sys.argv[1:])


def select_features_with_mi(X, y, threshold=0.2):
    mi_model = mutual_info_regression(X, y, random_state = 1)
    scores = pd.Series(mi_model, index=np.array(X.columns))
    t = scores.max() * threshold
    selected_features = [feature for feature,
                         score in scores.items() if score >= t]
    return X[selected_features]

def get_weights_from_classifiers(X, y,
                                 classifiers,
                                 weights_attributes=[
                                     'coef_', 'feature_importances_'],
                                 train_size=0.8, random_state=1):
    global logger_jowmdroid
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=train_size, random_state=random_state)
    weights_list = []
    for classifier in classifiers.values():
        classifier.fit(X_train, y_train)
        for weights_attribute in weights_attributes:
            is_found = False
            if(weights_attribute in dir(classifier)):
                weights = classifier.__getattribute__(weights_attribute)
                # A linha abaixo garante que weights não seja um vetor de vetor (e.g.: [2,3,4,4], e não [[2,3,4,4]]).
                weights = weights if isinstance(
                    weights[0], np.float64) else weights[0]
                weights_list.append(weights)
                is_found = True
                break
        if(not is_found):
            logger_jowmdroid.warning(
                f"Vetor de pesos para o classificador {classifier.__class__.__name__} não foi encontrado. Verifique o parametro weights_attributes")
    return weights_list

def get_normalized_weights_average(weights_list):
    normalized_weights = [[]] * len(weights_list)
    for i, weights in enumerate(weights_list):
        max_value = weights.max()
        min_value = weights.min()
        if max_value != min_value:
            normalized_weights[i] = (
                weights - min_value) / (max_value - min_value)
        else:
            normalized_weights[i] = np.array([0.5] * len(weights))
    return np.average(normalized_weights, axis=0)

def power(v):
    a, y, x = v
    result = y * np.power(x, a)
    return result

power.bounds = [(0.0, 10.0), (10.0 ** -6, 10.0), (0.0, 1.0)]

def exponential(v):
    a, b, y, x = v
    result = y * (np.power(a, (b*x)) - 1) / ((a ** b) - 1)
    return result

exponential.bounds = [(0.0, 10.0), (10.0 ** -6, 10.0), (10.0 ** -6, 10.0), (0.0, 1.0)]

def logarithmic(v):
    a, y, x = v
    result = y * ((np.log(1 + a*x)) / (np.log(1 + a)))
    return result

logarithmic.bounds = [(0.0, 10.0), (10.0 ** -6, 10.0), (0.0, 1.0)]

def hyperbolic(v):
    a, b, y, x = v
    result = (y * ((a*x) / (1 + (b*x)))) / (a / (1 + b))
    return result

hyperbolic.bounds = [(0.0, 10.0), (10.0 ** -6, 10.0), (10.0 ** -6, 10.0), (0.0, 1.0)]

def S_curve(v):
    a, y, x = v
    result = (y * ((1 / (1 + (a * np.exp(-x)))) - (1 / (1 + a)))) / \
        ((1 / (1 + (a*np.exp(-1)))) - (1 / (1+a)))
    return result

S_curve.bounds = [(0.0, 10.0), (10.0 ** -6, 10.0), (0.0, 1.0)]

def objective_function(parameters, *args):
    mapping_function, initial_weights, classifier, X, y, cv, metric = args
    mapped_weights = mapping_function(
        list(parameters[:len(mapping_function.bounds) - 1]) + list([initial_weights]))
    optimized_classifier = classifier['model'].set_params(
        **{classifier['parameter_name']: parameters[-1]}) if 'parameter_name' in classifier else classifier['model']
    result = cross_val_score(optimized_classifier, np.multiply(
        X, mapped_weights), y, cv=cv, scoring=metric)
    return result.mean()

def run_jowmdroid(X, y, weight_classifiers, evaluation_classifiers, mapping_functions,
                  train_size=0.8, random_state=1,
                  popsize=40, maxiter=30, recombination=0.3, disp=False, mutation=0.5, seed=1,
                  cv=5, scoring=["accuracy", "precision", "recall", "f1"],
                  include_hyperparameter=True):
    global logger_jowmdroid
    inspect_frame(inspect.currentframe())
    logger_jowmdroid.info('Calculando initial_weights...')
    initial_weights = get_normalized_weights_average(
        get_weights_from_classifiers(X, y, weight_classifiers))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state)
    solutions = {}
    for classifier in evaluation_classifiers:
        solutions[classifier['name']] = {}
        logger_jowmdroid.info('Otimizando parâmetros com o DE. Classificador: ' + classifier['name'])
        for mapping_function in mapping_functions:
            bounds = mapping_function.bounds + [classifier['bound']] if include_hyperparameter == True and (
                'parameter_name' in classifier or 'bound' in classifier) else mapping_function.bounds

            solution = differential_evolution(objective_function, bounds=bounds,
                                              args=(
                                                  mapping_function, initial_weights, classifier, X_train, y_train, cv, scoring[0]),
                                              popsize=popsize, maxiter=maxiter, recombination=recombination,
                                              disp=disp, mutation=mutation, seed=seed).x
            solutions[classifier['name']][mapping_function.__name__] = solution
        logger_jowmdroid.info(
            f"Melhores parâmetros das funções de mapeamento para o classificador {classifier['name']}: {solutions}")
    results = []
    for classifier in evaluation_classifiers:
        logger_jowmdroid.info('Avaliando o classificador ' + classifier['name'])
        for mapping_function_name, solution in solutions[classifier['name']].items():
            optimized_classifier = classifier['model'].set_params(
                **{classifier['parameter_name']: solution[-1]}) if 'parameter_name' in classifier else classifier['model']

            scores = cross_validate(optimized_classifier, X=X_test, y=y_test, cv=cv, scoring=scoring)
            result = {'classifier': classifier['name'],
                      'mapping_function': mapping_function_name}
            for metric in scoring:
                result[metric] = scores["test_" + metric].mean()
            results.append(result)
    return pd.DataFrame(results)


def inspect_frame(frame):
    args, _, _, values = inspect.getargvalues(frame)
    for i in args:
        print(f'{i} = {values[i]}')

if __name__ == "__main__":
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_jowmdroid
    logger_jowmdroid = logging.getLogger('JOWMDroid')
    logger_jowmdroid.setLevel(logging.INFO)

    parsed_args = parse_args()

    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    init_size = X.shape[1]
    start_time = timeit.default_timer()
    X = select_features_with_mi(X, y, threshold=parsed_args.mi_threshold)
    end_time = timeit.default_timer()
    logger_jowmdroid.info("Elapsed Time: {}".format(end_time - start_time))
    if(X.shape[1] == 0):
        logger_jowmdroid.warning("AVISO: 0 features selecionadas")
    features_dataset = X
    features_dataset['class'] = y
    features_dataset.to_csv(get_filename(parsed_args.output_file, prefix=parsed_args.output_prefix), index = False)
    if(parsed_args.feature_selection_only):
        logger_jowmdroid.info("Selected Features >> %s of %s" % (features_dataset.shape[1]-1, init_size))
        exit(0)

    weight_classifiers = {"SVM": SVC(
        kernel='linear'), "RF": RandomForestClassifier(), "LR": LogisticRegression()}

    functions = {'power': power, 'exponential': exponential, 'logarithmic': logarithmic, 'hyperbolic': hyperbolic, 'S_curve': S_curve}
    mapping_functions = [functions[name] for name in parsed_args.mapping_functions.replace(' ', '').split(",")]
    evaluation_classifiers = [
        {"name": "KNN", "model": KNeighborsClassifier(n_neighbors=1)},
        {"name": "SVM", "model": SVC(kernel='linear'), "parameter_name": "C", "bound": (1.0, 5.0)},
        {"name": "MLP", "model": MLPClassifier(), "parameter_name": "learning_rate_init", "bound": (0.0001, 0.01)},
        {"name": "LR", "model": LogisticRegression(), "parameter_name": "C", "bound": (1.0, 5.0)}
    ]

    results = run_jowmdroid(X, y, weight_classifiers, evaluation_classifiers,
                            mapping_functions, cv=parsed_args.cv, train_size=parsed_args.train_size,
                            include_hyperparameter = not parsed_args.exclude_hyperparameter)

    results.to_csv(get_filename(parsed_args.output_file, prefix=parsed_args.output_prefix, suffix='evaluation_results'), index = False)
