#!/bin/bash

#cd ../..

# custom config
DATA="/fs-computility/ai-shen/path/to/CLIP/"
TRAINER=ZeroshotCLIP
DATASETS=("imagenet" "caltech101" "dtd" "eurosat" "oxford_pets" "oxford_flowers" "fgvc_aircraft" "food101" "stanford_cars" "sun397" "ucf101")
CFG=vit_l14  # rn50, rn101, vit_b32 or vit_b16
attacks=$1

for DATASET in "${DATASETS[@]}"; do
    DIR=./output/${TRAINER}/${attacks}/${CFG}/${DATASET}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"

        python train.py \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/zeroshot/${CFG}.yaml \
        --output-dir ${DIR} \
        --attacks ${attacks} \
        --eval-only
    fi
done
