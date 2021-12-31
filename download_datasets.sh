DATASETS=("abalone" "bodyfat" "cpusmall" "housing" "mg" "mpg" "space_ga")
DATASETS_DIR="datasets"
mkdir -p "${DATASETS_DIR}"
BASE_LINK="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression"
for dataset in "${DATASETS[@]}"
    do [[ ! -f "${dataset}" ]] && wget -O "${DATASETS_DIR}/${dataset}" "${BASE_LINK}/${dataset}_scale"
done
