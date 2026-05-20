# custom config
DATA="/path/to/CLIP/"
TRAINER=AdvCoOp

DATASETS=("imagenet" "caltech101" "dtd" "eurosat" "oxford_pets" "oxford_flowers" "fgvc_aircraft" "food101" "stanford_cars" "sun397" "ucf101")
CFG=vit_b16_c2_ep100_batch32    # config file
CTP=end                         # class token position (end or middle)
NCTX=32                         # number of context tokens
SHOTS=16                        # number of shots (1, 2, 4, 8, 16)
CSC=False                       # class-specific context (False or True)

SEED=1

for DATASET in "${DATASETS[@]}"; do
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
        TRAINER.AdvCoOp.N_CTX ${NCTX} \
        TRAINER.AdvCoOp.CSC ${CSC} \
        TRAINER.AdvCoOp.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
