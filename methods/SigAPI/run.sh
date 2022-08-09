#!/bin/bash
. methods/utils.sh
verify_pip_packages pandas numpy scikit-learn

set_increment(){
    TOTAL_FEATURES=$1
    [[ $TOTAL_FEATURES -lt 50 ]] && INCREMENT=1 && return
    [[ $TOTAL_FEATURES -lt 1000 ]] && INCREMENT=5 && return
    INCREMENT=10
}

sigapi(){
    DATASET=$1
    D_NAME=$2
}

[[ $1 && $2 ]] || { echo "Uso: bash $0 OUTPUT_PREFIX DATASET [DATASET...]" && exit 1;}
OUTPUT_PREFIX=$1; shift
for DATASET in $*
do
    D_NAME=$(echo $DATASET | awk -F/ '{print $NF}')
    set_increment `head -1 $DATASET | awk -F, '{print NF-1}'`
    TS=$(date +%Y%m%d%H%M%S)
    OUT_FILENAME="dataset_sigapi_${D_NAME}_$TS"
    #OUT_FILENAME="dataset_sigapi_${D_NAME}"    
    #echo "python3 -m methods.SigAPI.main -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX -i $INCREMENT"
    python3 -m methods.SigAPI.main -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX -i $INCREMENT
    #{ time python3 -m methods.SigAPI.main -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX -i $INCREMENT; } 2> time_$OUT_FILENAME.txt
done
