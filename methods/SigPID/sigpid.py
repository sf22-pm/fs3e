import pandas as pd
import numpy  as np
import timeit
import argparse
import csv
import os, sys
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from methods.SigPID.spinner import Spinner
from methods.utils import get_base_parser, get_dataset, get_filename
import logging
from tqdm import tqdm

B = None
M = None

def parse_args(argv):
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    args = parser.parse_args(argv)
    return args

def S_B(j):
    sigmaBij = B.sum(axis = 0, skipna = True)[j]
    sizeBj = B.shape[0]
    sizeMj = M.shape[0]
    return (sigmaBij/sizeBj)*sizeMj

def PRNR(j):
    sigmaMij = (M.sum(axis = 0, skipna = True)[j]) * 1.0
    S_Bj = S_B(j)
    r = (sigmaMij - S_Bj)/(sigmaMij + S_Bj) if sigmaMij > 0.0 and S_Bj > 0.0 else 0.0
    return r

def check_dirs():
    import shutil
    root_path = os.getcwd()
    root_path = os.path.join(root_path, 'MLDP')
    #print(root_path)
    if os.path.exists(root_path):
        shutil.rmtree('MLDP')
    dirs = ['PRNR', 'SPR', 'PMAR']
    for dir in dirs:
        path = os.path.join(root_path, dir)
        os.makedirs(path)
        #print('Directory', dir, 'Created.')

def calculate_PRNR(dataset, filename, class_column):
    permissions = dataset.drop(columns=[class_column])
    with open(filename,"w", newline='') as f:
        f_writer = csv.writer(f)
        for p in permissions:
            permission_PRNR_ranking = PRNR(p)
            if permission_PRNR_ranking != 0:
                f_writer.writerow([p,permission_PRNR_ranking])

def permission_list(filename, asc):
    colnames = ['permission','rank']
    list = pd.read_csv(filename, names = colnames)
    list = list.sort_values(by = ['rank'], ascending = asc)
    #print(list)
    return list

def SVM(dataset_df, class_column):
    from sklearn import metrics
    state = np.random.randint(100)
    Y = dataset_df[class_column]
    X = dataset_df.drop([class_column], axis = 1)

    start_time = timeit.default_timer()
    #split between train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.3,random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train,y_train)

    #prediction labels for X_test
    y_pred=svm.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    elapsed_time = timeit.default_timer() - start_time

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, zero_division = 0)
    recall = metrics.recall_score(y_test, y_pred, zero_division = 0)
    f1_score = metrics.f1_score(y_test, y_pred, zero_division = 0)
    fpr = fp/(fp+tn)

    return elapsed_time, accuracy, precision, recall, f1_score, fpr

def run_PMAR(dataset, prnr_malware, class_column):
    global logger_sigpid
    features_name = dataset.columns.values.tolist()
    class_apk = dataset[class_column]
    features_dataset = dataset.drop([class_column], axis=1)
    num_apk = features_dataset.shape[0] - 1
    num_features = features_dataset.shape[1]

    logger_sigpid.info("Mining Association Rules")
    #spn = Spinner("Mining Association Rules")
    #spn.start()
    records = []
    for i in range(0,num_apk):
        if class_apk[i] in [0, 1]:
            i_list = []
            for j in range(0,num_features):
                if features_dataset.values[i][j] == 1:
                    i_list.append(features_name[j])
            records.append(i_list)
    #print(records)

    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    freq_items = apriori(df,
                        min_support = 0.1,
                        use_colnames = True,
                        max_len = 2,
                        verbose = 0)
    if not freq_items.empty:
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.965)
    else:
        rules = []
    #rules = rules[['antecedents', 'consequents', 'support','confidence']]
    #spn.stop()
    PMAR_df = dataset
    deleted_ft = []
    for i in range(0,len(rules)):
        ant = list(rules.loc[i,'antecedents'])[0]
        con = list(rules.loc[i,'consequents'])[0]
        rank_ant = prnr_malware.loc[(prnr_malware['permission'] == ant)].values[0,1]
        rank_con = prnr_malware.loc[(prnr_malware['permission'] == con)].values[0,1]
        #print(ant, rank_ant, con, rank_con)
        to_delete = ant if rank_ant < rank_con else con
        if to_delete not in deleted_ft:
            #print(to_delete)
            PMAR_df = PMAR_df.drop([to_delete], axis=1)
            deleted_ft.append(to_delete)

    return PMAR_df

def plot_results(step, annotation, annx, anny):
    colnames = ['num_permissions', 'elapsed_time', 'accuracy', 'precision', 'recall', 'f1_score', 'fpr']
    f = "MLDP/" + step + "/svm_results.csv"
    plot_ = pd.read_csv(f, names = colnames)
    plot_ = plot_.drop(['elapsed_time', 'fpr'], axis=1)
    markers = ['o', 's', '^', 'x']
    ax = plot_.plot(x='num_permissions', xticks=plot_.num_permissions, rot=90)
    x_lbl = list(plot_['num_permissions'])[-1] * 0.4
    y_lbl = min(list(plot_.min())) + 0.1
    txt = annotation + "\n" + "{}, {:.3f}".format(annx, anny)
    ax.annotate(txt,
                xy = (annx, anny),
                xycoords='data',
                xytext=(x_lbl, y_lbl),
                textcoords='data',
                ha='center',
                arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
    plt.title('PIS + ' + step)
    plt.xlabel('Number of Permissions')
    plt.ylabel('Metrics')
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(markers[i])
    plt.savefig('pis_' + step + '.png')

def drop_internet(dataset):
    cols = []
    to_drop = ['android.permission.INTERNET', 'INTERNET']
    features = dataset.columns.values.tolist()
    cols = list(set(features).intersection(to_drop))
    #print(cols)
    ds = dataset.drop(columns=cols)
    return ds

if __name__=="__main__":

    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_sigpid
    logger_sigpid = logging.getLogger('SigPID')
    logger_sigpid.setLevel(logging.INFO)

    check_dirs()
    args = parse_args(sys.argv[1:])

    try:
        initial_dataset = get_dataset(args)
    except BaseException as e:
        logger_sigpid.exception(e)
        exit(1)

    dataset = drop_internet(initial_dataset)
    B = dataset[(dataset[args.class_column] == 0)]
    M = dataset[(dataset[args.class_column] == 1)]

    calculate_PRNR(B, "MLDP/PRNR/PRNR_B_List.csv", args.class_column)
    calculate_PRNR(M, "MLDP/PRNR/PRNR_M_List.csv", args.class_column)

    benigns_permissions = permission_list("MLDP/PRNR/PRNR_B_List.csv", True)
    malwares_permissions = permission_list("MLDP/PRNR/PRNR_M_List.csv", False)

    num_permissions = dataset.shape[1] - 1 #CLASS

    logger_sigpid.info('PRNR Generating Subset of Permissions')
    #spn = Spinner('PRNR Generating Subset of Permissions')
    #spn.start()
    counter = increment = 3
    while counter < num_permissions/2 + increment:
        malwares_head_perms = malwares_permissions['permission'].head(counter).values
        benigns_head_perms = benigns_permissions['permission'].head(counter).values
        subset_permissions = list(set(malwares_head_perms) | set(benigns_head_perms))
        #print(subset_permissions)
        #print(len(subset_permissions))
        subset_permissions.append(args.class_column)
        subset = dataset[subset_permissions]
        evaluated_ft = counter * 2
        evaluated_ft = num_permissions if evaluated_ft > num_permissions else evaluated_ft
        subset.to_csv("MLDP/PRNR/subset_" + str(evaluated_ft) + ".csv", index = False)
        counter += increment
    #spn.stop()

    counter = increment = 6
    best_PRNR_accuracy = 0.0
    best_PRNR_counter = 0
    #spn = Spinner('Running PIS + PRNR')
    #spn.start()
    with open("MLDP/PRNR/svm_results.csv","w", newline='') as f:
        f_writer = csv.writer(f)
        logger_sigpid.info('Running PIS + PRNR')
        pbar = tqdm(range(num_permissions), disable = (logger_sigpid.getEffectiveLevel() > logging.INFO))
        while counter < num_permissions + increment:
            evaluated_ft = num_permissions if counter > num_permissions else counter
            pbar.set_description("With %s Features" % str(evaluated_ft))
            pbar.n = evaluated_ft
            #txt = "Running PIS + PRNR With {} Features".format(evaluated_ft)
            #spn = Spinner(txt)
            #spn.start()
            dataset_df = pd.read_csv('MLDP/PRNR/subset_' + str(evaluated_ft) + '.csv', encoding = 'utf8')
            results = list(SVM(dataset_df, args.class_column))
            if results[1] > best_PRNR_accuracy:
                best_PRNR_accuracy = results[1]
                best_PRNR_counter = evaluated_ft
            f_writer.writerow([evaluated_ft] + results)
            #spn.stop()
            counter += increment
        pbar.close()
    #spn.stop()
    #print("PRNR:", best_PRNR_counter, "Permissions", ">>", "Accuracy ({:.3f})".format(best_PRNR_accuracy))

    #Plot PIS + PRNR
    plot_results("PRNR", "Best Accuracy", best_PRNR_counter, best_PRNR_accuracy)

    #SPR
    PRNR_df = pd.read_csv("MLDP/PRNR/subset_" + str(best_PRNR_counter) + ".csv", encoding = 'utf8')
    PRNR_df = PRNR_df.drop(columns=[args.class_column])

    #calculates the support of each permission
    supp = PRNR_df.sum(axis = 0)
    supp = supp.sort_values(ascending=False)

    logger_sigpid.info('SPR Generating Subset of Permissions')
    #spn = Spinner('SPR Generating Subset of Permissions')
    #spn.start()
    counter = increment = 5
    while counter < best_PRNR_counter + increment:
        subset_permissions = list(supp.head(counter).index)
        #print(subset_permissions)
        #print(len(subset_permissions))
        subset_permissions.append(args.class_column)
        subset = dataset[subset_permissions]
        evaluated_ft = best_PRNR_counter if counter > best_PRNR_counter else counter
        subset.to_csv("MLDP/SPR/subset_" + str(evaluated_ft) + ".csv", index = False)
        counter += increment
    #spn.stop()

    counter = increment = 5
    best_SPR_accuracy = best_PRNR_accuracy
    best_SPR_counter = best_PRNR_counter
    #spn = Spinner('Running PIS + SPR')
    #spn.start()
    with open("MLDP/SPR/svm_results.csv","w", newline='') as f:
        f_writer = csv.writer(f)
        logger_sigpid.info('Running PIS + SPR')
        pbar_spr = tqdm(range(best_PRNR_counter), disable = (logger_sigpid.getEffectiveLevel() > logging.INFO))
        while counter < best_PRNR_counter + increment:
            evaluated_ft = best_PRNR_counter if counter > best_PRNR_counter else counter
            pbar_spr.set_description("With %s Features" % str(evaluated_ft))
            pbar_spr.n = evaluated_ft
            #txt = "Running PIS + SPR With {} Features".format(evaluated_ft)
            #spn = Spinner(txt)
            #spn.start()
            dataset_df = pd.read_csv('MLDP/SPR/subset_' + str(evaluated_ft) + '.csv', encoding = 'utf8')
            results = list(SVM(dataset_df, args.class_column))
            if results[1] >= 0.9 and evaluated_ft < best_SPR_counter:
                best_SPR_accuracy = results[1]
                best_SPR_counter = evaluated_ft
            f_writer.writerow([evaluated_ft] + results)
            #spn.stop()
            counter += increment
        pbar_spr.close()
    #spn.stop()
    #print("SPR:", best_SPR_counter, "Permissions", ">>", "Accuracy ({:.3f})".format(best_SPR_accuracy))

    #Plot PIS + SPR
    plot_results("SPR", "Pruning Point", best_SPR_counter, best_SPR_accuracy)

    #PMAR
    SPR_df = pd.read_csv("MLDP/SPR/subset_" + str(best_SPR_counter) + ".csv", encoding = 'utf8')
    final_dataset = run_PMAR(SPR_df, malwares_permissions, args.class_column)

    final_dataset.to_csv(get_filename(args.output_file, prefix=args.output_prefix), index=False)
    final_perms = len(final_dataset.columns) - 1
    num_permissions = initial_dataset.shape[1] - 1
    pct = (1.0 - (final_perms/num_permissions)) * 100.0
    logger_sigpid.info("%s to %s Permissions. Reduction of %.2f%%" % (num_permissions, final_perms, pct))
    #print(num_permissions, "to", final_perms, "Permissions. Reduction of {:.3f}%".format(pct))

"""
    #Testing Final Dataset
    result = list(SVM(final_dataset))
    msg = "Dataset Final\n"
    msg += "Accuracy: {:.3f}\n".format(result[1])
    msg += "Precision: {:.3f}\n".format(result[2])
    msg += "Recall: {:.3f}\n".format(result[3])
    msg += "F1 Score: {:.3f}\n".format(result[4])
    msg += "FPR: {:.3f}".format(result[5])
    print(msg)
"""
