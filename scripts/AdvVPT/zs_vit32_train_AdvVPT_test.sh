
# evaluation

DATA="/fs-computility/ai-shen/path/to/CLIP/"
TRAINER=AdvVPT

DATASETS=("imagenet" "caltech101" "dtd" "eurosat" "oxford_pets" "oxford_flowers" "fgvc_aircraft" "food101" "stanford_cars" "sun397" "ucf101")
SEED=1
EPOCHS=(100)  # ($(seq 10 10 100)) Generate sequence from 0 to 100 with steps of 10 

CFG=vit_b32_c2_ep100_batch32_2ctx_9depth
SHOTS=16
attacks=$1

for DATASET in "${DATASETS[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
        DIR=./output/evaluation/${attacks}/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}/${EPOCH}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"

            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ./output/train/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
            --load-epoch ${EPOCH} \
            --attacks ${attacks} \
            --eval-only
        fi

    done
done