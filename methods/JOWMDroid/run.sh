. methods/utils.sh
verify_pip_packages numpy scipy pandas scikit-learn

[[ $1 && $2 ]] || { echo "[JOWMDroid] Usage: bash $0 OUTPUT_PREFIX DATASET [DATASET...]" && exit 1;}
OUTPUT_PREFIX=$1; shift
for DATASET in $*
do
    D_NAME=$(echo $DATASET | awk -F/ '{print $NF}')
    TS=$(date +%Y%m%d%H%M%S)
    #OUT_FILENAME="dataset_jowmdroid_${D_NAME}_$TS"
    OUT_FILENAME="dataset_jowmdroid_${D_NAME}"
    #echo "python3 -m methods.JOWMDroid.JOWMDroid -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX --output-prefix $OUTPUT_PREFIX --output-prefix $OUTPUT_PREFIX --output-prefix $OUTPUT_PREFIX --feature-selection-only --exclude-hyperparameter"
    python3 -m methods.JOWMDroid.JOWMDroid -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX --feature-selection-only --exclude-hyperparameter
    #{ time python3 -m methods.JOWMDroid.JOWMDroid -d $DATASET -o $OUT_FILENAME --output-prefix $OUTPUT_PREFIX --feature-selection-only --exclude-hyperparameter; } 2> time_$OUT_FILENAME.txt
done
