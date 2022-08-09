#!/usr/bin/python3 

from argparse import ArgumentParser
import argparse
import sys
import glob
import asyncio
from itertools import chain
import pandas as pd
import re
import seaborn as sns
import logging
import os

def create_executable(program_name):
    async def executable(*args):
        global logger
        process = await asyncio.create_subprocess_exec('/bin/bash', program_name, *args)
        await process.wait()
        if(process.returncode != 0):
            msg = f"Program '{program_name}' called with args '{' '.join(args)}' returned with error"
            logger.warning(msg)
        return program_name, args
    return executable

def get_fs_methods():
    program_names = glob.glob('methods/*/run.sh')
    fs_methods = {}
    for program_name in program_names:
        method_name = program_name.split('/')[1].lower()
        fs_methods[method_name] = create_executable(program_name)
    return fs_methods

ml_models = ['svm','rf']
fs_methods = get_fs_methods()

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        global logger
        self.print_usage()
        logger.error(message)
        sys.exit(2)

def parse_args():
    parser = DefaultHelpParser(description="Suite to run feature selection (FS) methods and evaluation of machine learning (ML) algorithms")
    #parser = ArgumentParser(description="Suite to run feature selection (FS) methods and evaluation of machine learning (ML) algorithms")
    subparsers = parser.add_subparsers(title='Available commands', dest="command")

    list_parser = subparsers.add_parser('list', help='List available feature selection methods and/or machine learning models')
    list_group = list_parser.add_mutually_exclusive_group(required=True)
    list_group.add_argument("--all", action='store_true')
    list_group.add_argument("--fs-methods", action='store_true')
    list_group.add_argument("--ml-models", action='store_true')

    run_parser = subparsers.add_parser("run", help='Run experiment with feature selection methods and ML models')
    run_parser.add_argument('-d', '--datasets', required=True, help='Datasets to run the experiment', nargs='+')
    run_parser.add_argument(f'--fs-methods', help=f'Feature selection methods to include. Default: all', choices=list(fs_methods.keys()) + ['all'], nargs='*', default='all')
    run_parser.add_argument(f'--ml-model', help=f'Machine learning model for evaluation of datasets resulting from feature selection. Default: all', choices=ml_models + ['all'], default='all')
    run_parser.add_argument(f'--output-prefix', help="Prefix of output file names. Should not be an empty string. Default: 'results'", default='results')
    run_parser.add_argument(f'--plot-fs-methods', help=f'Feature selection methods to plot. Default: all', choices=list(fs_methods.keys()) + ['all'], nargs='*', default='all')
    run_parser.add_argument(f'--plot-ml-models', help=f'Machine learning model to plot. Default: all', choices=ml_models + ['all'], default='all', nargs='*')
    #run_parser.add_argument(f'--output-dir', metavar = 'DIRECTORY', help = f'Directory For Output Data. Default: outputs', type = str, default = 'outputs')
    args = parser.parse_args(sys.argv[1:])
    return args

async def run_fs_methods(output_prefix, chosen_methods, datasets):
    global logger
    tasks = []
    for method in chosen_methods:
        msg = f"STARTING {method}"
        logger.info(msg)
        tasks.append(asyncio.create_task(fs_methods[method](output_prefix, ' '.join(datasets))))
    for task in tasks:
        await task

async def run_ml_model(output_prefix, model, datasets):
    model_executable = create_executable('run_evaluation.sh')
    await model_executable(output_prefix, model, ' '.join(datasets))

def graph_metrics(df, filename, output_prefix):
    current_method = filename.split('_')[-3]
    current_dataset = filename.split('_')[-2]
    models_index = list(df['model'].str.upper())
    metrics_dict = dict()
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics_list:
        metrics_dict[metric] = list(df[metric] * 100.0)

    df = pd.DataFrame(metrics_dict, index = models_index)
    df.columns = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AuC']
    ax = df.plot.bar(rot = 0, edgecolor='white', linewidth = 1)
    ax.set_xlabel('Model')
    ax.set_ylabel('Values (%)')
    ax.legend(ncol = 1, loc = 'lower left')
    ax.set_ylim(0, 100)
    ax.set_xlim(-1, len(models_index))
    ax.set_title(f'Results for {current_method} with dataset {current_dataset}')
    path_graph_file = f"{output_prefix}_metrics_of_{filename.replace('.csv', '')}.png"
    ax.figure.savefig(path_graph_file, dpi = 300)


def graph_class(df, filename, output_prefix):
    current_method = filename.split('_')[-3]
    current_dataset = filename.split('_')[-2]
    models_index = list(df['model'].str.upper())
    classification_dict = dict()
    classification_list = ['TP', 'FP', 'TN', 'FN']
    for classification in classification_list:
        classification_dict[classification] = list(df[classification.lower()])

    df = pd.DataFrame(classification_dict, index = models_index)
    stacked_data = df.apply(lambda x: x*100/sum(x), axis = 1)
    ax = stacked_data.plot.barh(rot = 0, stacked = True)
    ax.set_xlabel('Values (%)')
    ax.set_ylabel('Model')
    ax.set_ylim(-1, len(models_index))
    ax.legend(ncol = len(classification_list), loc = 'upper center')
    for container in ax.containers:
        if container.datavalues[0] > 2.5:
            ax.bar_label(container, label_type = 'center', color = 'black', weight='bold', fmt = '%.2f')
    ax.set_title(f'Classification to {current_method} with dataset {current_dataset}')
    path_graph_file = f"{output_prefix}_class_of_{filename.replace('.csv', '')}.png"
    ax.figure.savefig(path_graph_file, dpi = 300)
def plot_results(all_ml_results_filenames, chosen_methods, chosen_models, output_prefix):
    chosen_results_filenames = [results_filename for results_filename in all_ml_results_filenames if re.search('|'.join(chosen_methods), results_filename)]
    for filename in chosen_results_filenames:
        df = pd.read_csv(filename)
        df = df[df['model'].isin(chosen_models)]
        graph_metrics(df, filename, output_prefix)
        graph_class(df, filename, output_prefix)
async def run_command(parsed_args):
    chosen_methods = list(fs_methods.keys()) if 'all' in parsed_args.fs_methods else parsed_args.fs_methods
    await run_fs_methods(parsed_args.output_prefix, chosen_methods, parsed_args.datasets)

    # [IMPORTANTE]
    # para obter os datasets de features selecionadas, a linha a seguir assume que eles possuem o nome no formato especificado
    dataset_filenames = chain(*[glob.glob(f"{parsed_args.output_prefix}_dataset_{method}*.csv") for method in chosen_methods])
    await run_ml_model(parsed_args.output_prefix, parsed_args.ml_model, dataset_filenames)

    ml_results_filenames = glob.glob(f"{parsed_args.output_prefix}_ml_results*.csv")
    chosen_methods_to_plot = list(fs_methods.keys()) if 'all' in parsed_args.plot_fs_methods else parsed_args.plot_fs_methods
    chosen_models_to_plot = ml_models if 'all' in parsed_args.plot_ml_models else parsed_args.plot_ml_models
    plot_results(ml_results_filenames, chosen_methods_to_plot, chosen_models_to_plot, parsed_args.output_prefix)

def list_command(parsed_args):
    if(parsed_args.fs_methods):
        print(', '.join(fs_methods))
    elif(parsed_args.ml_models):
        print(', '.join(ml_models))
    else:
        print('methods:', ', '.join(fs_methods.keys()))
        print('models:', ', '.join(ml_models))

command = {
    'run' : lambda parsed_args: asyncio.run(run_command(parsed_args)),
    'list': list_command
}

def main():
    global logger
    parsed_args = parse_args()
    if(parsed_args.command == None):
        msg = "You must use one of these commands:", ', '.join(command.keys())
        logger.error(msg)
        exit(1)
    command[parsed_args.command](parsed_args)

if __name__ == '__main__':
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger
    logger = logging.getLogger('FS3E')
    logger.setLevel(logging.INFO)
    main()
