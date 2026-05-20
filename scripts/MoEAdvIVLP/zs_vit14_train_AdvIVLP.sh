# custom config
DATA="/path/to/CLIP/"
TRAINER=MoEAdvIVLP

DATASET=("imagenet")
SEED=1

CFG=vit_l14_c2_ep100_batch32_2+2ctx_18depth
SHOTS=16



DIR=./output/train/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi



# evaluation

DATA="/path/to/CLIP/"
TRAINER=MoEAdvIVLP

DATASETS=("imagenet" "caltech101" "dtd" "eurosat" "oxford_pets" "oxford_flowers" "fgvc_aircraft" "food101" "stanford_cars" "sun397" "ucf101")
SEED=1
EPOCHS=(20)  # MoE VLI canonical recipe for ViT-L/14 ("L18WU10"): depth=18, warmup=10, load_epoch=20
ATTACKS=("clean" "pgd" "auto" "di" "ti" "cw")

CFG=vit_l14_c2_ep100_batch32_2+2ctx_18depth
SHOTS=16

for ATTACK in "${ATTACKS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for EPOCH in "${EPOCHS[@]}"; do
            DIR=./output/evaluation/${ATTACK}/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}/${EPOCH}
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
                --attacks ${ATTACK} \
                --eval-only
            fi

        done
    done
done
