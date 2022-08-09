#!/bin/bash
. methods/utils.sh
verify_pip_packages pandas numpy scikit-learn

[[ $1 && $2 ]] || { echo "[Linear Regression] Usage: bash $0 OUTPUT_PREFIX DATASET [DATASET...]" && exit 1;}
OUTPUT_PREFIX=$1; shift
for DATASET in $*
do
    D_NAME=$(echo $DATASET | awk -F/ '{print $NF}')
    TS=$(date +%Y%m%d%H%M%S)
    #OUT_FILENAME="dataset_linearregression_${D_NAME}_$TS"
    OUT_FILENAME="dataset_linearregression_${D_NAME}"
    #echo "python3 -m methods.LinearRegression.LinearRegression -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX"
    python3 -m methods.LinearRegression.LinearRegression -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX
    #{ time python3 -m methods.LinearRegression.LinearRegression -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX; } 2> time_$OUT_FILENAME.txt
done
