#!/bin/bash
. methods/utils.sh
verify_pip_packages pandas scikit-learn mlxtend matplotlib

[[ $1 && $2 ]] || { echo "Uso: bash $0 OUTPUT_PREFIX DATASET [DATASET...]" && exit 1;}
OUTPUT_PREFIX=$1; shift
for DATASET in $*
do
    D_NAME=$(echo $DATASET | awk -F/ '{print $NF}')
    TS=$(date +%Y%m%d%H%M%S)
    #OUT_FILENAME="dataset_sigpid_${D_NAME}_$TS"
    OUT_FILENAME="dataset_sigpid_${D_NAME}"
    #echo "python3 -m methods.SigPID.sigpid -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX"
    python3 -m methods.SigPID.sigpid -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX
    #{ time python3 -m methods.SigPID.sigpid -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX; } 2> time_$OUT_FILENAME.txt
done
