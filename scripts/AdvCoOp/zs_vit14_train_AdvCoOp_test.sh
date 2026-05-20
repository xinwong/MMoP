# evaluation

DATA="/fs-computility/ai-shen/path/to/CLIP/"
TRAINER=AdvCoOp
SHOTS=16                        # number of shots (1, 2, 4, 8, 16)
NCTX=32                         # number of context tokens
CSC=False                       # class-specific context (False or True)
CTP=end                         # class token position (end or middle)

DATASETS=("imagenet" "caltech101" "dtd" "eurosat" "oxford_pets" "oxford_flowers" "fgvc_aircraft" "food101" "stanford_cars" "sun397" "ucf101")
CFG=vit_l14_c2_ep100_batch32    # config file

SEED=1
EPOCHS=(100)  # ($(seq 10 10 100)) Generate sequence from 0 to 100 with steps of 10 
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
            --model-dir ./output/train/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
            --load-epoch ${EPOCH} \
            --eval-only \
            --attacks ${attacks} \
            TRAINER.AdvCoOp.N_CTX ${NCTX} \
            TRAINER.AdvCoOp.CSC ${CSC} \
            TRAINER.AdvCoOp.CLASS_TOKEN_POSITION ${CTP}
        fi

    done
done