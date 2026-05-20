#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA="/path/to/CLIP/"
TRAINER=AdvCoOp
CFG=vit_b32_c2_ep100_batch32
CTP=end
NCTX=32
SHOTS=16
CSC=False
SEED=1
EPOCH=100
FORCE_RUN="${FORCE_RUN:-0}"

ATTACKS_STR="${ATTACKS:-pgd cw di autoattack}"
DATASETS_STR="${DATASETS:-imagenet caltech101 dtd eurosat oxford_pets oxford_flowers fgvc_aircraft food101 stanford_cars sun397 ucf101}"

read -r -a ATTACKS <<< "${ATTACKS_STR}"
read -r -a DATASETS <<< "${DATASETS_STR}"

MODEL_DIR="${REPO_ROOT}/output/train/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}"
MODEL_FILE="${MODEL_DIR}/prompt_learner/model.pth.tar-${EPOCH}"
OUTPUT_ROOT="${REPO_ROOT}/output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots"
SUMMARY_FILE="${SUMMARY_FILE:-${OUTPUT_ROOT}/summary_vit32_zeroshot11_4attacks.tsv}"

is_complete() {
    local log_file="$1"
    [ -f "${log_file}" ] || return 1
    local result_count
    result_count=$(grep -c '^=> result$' "${log_file}" 2>/dev/null || true)
    [ "${result_count}" -ge 2 ]
}

mkdir -p "${OUTPUT_ROOT}"

if [ ! -f "${MODEL_FILE}" ]; then
    echo "[ERROR] missing checkpoint: ${MODEL_FILE}" >&2
    exit 1
fi

cd "${REPO_ROOT}" || exit 1

echo -e "attack\tdataset\tstatus\toutput_dir" > "${SUMMARY_FILE}"

ok_count=0
skip_count=0
fail_count=0

for ATTACK in "${ATTACKS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        OUT_DIR="${OUTPUT_ROOT}/${ATTACK}/${DATASET}/seed${SEED}/${EPOCH}"
        LOG_FILE="${OUT_DIR}/log.txt"

        if [ "${FORCE_RUN}" != "1" ] && is_complete "${LOG_FILE}"; then
            echo "[SKIP] ${ATTACK} ${DATASET} -> ${OUT_DIR}"
            echo -e "${ATTACK}\t${DATASET}\tskipped\t${OUT_DIR}" >> "${SUMMARY_FILE}"
            skip_count=$((skip_count + 1))
            continue
        fi

        mkdir -p "${OUT_DIR}"
        echo "[RUN] ${ATTACK} ${DATASET} -> ${OUT_DIR}"

        if python train.py \
            --root "${DATA}" \
            --seed "${SEED}" \
            --trainer "${TRAINER}" \
            --dataset-config-file "configs/datasets/${DATASET}.yaml" \
            --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
            --output-dir "${OUT_DIR}" \
            --model-dir "${MODEL_DIR}" \
            --load-epoch "${EPOCH}" \
            --eval-only \
            --attacks "${ATTACK}" \
            TRAINER.AdvCoOp.N_CTX "${NCTX}" \
            TRAINER.AdvCoOp.CSC "${CSC}" \
            TRAINER.AdvCoOp.CLASS_TOKEN_POSITION "${CTP}"
        then
            if is_complete "${LOG_FILE}"; then
                echo "[OK] ${ATTACK} ${DATASET}"
                echo -e "${ATTACK}\t${DATASET}\tok\t${OUT_DIR}" >> "${SUMMARY_FILE}"
                ok_count=$((ok_count + 1))
            else
                echo "[FAIL] ${ATTACK} ${DATASET} finished without a complete log" >&2
                echo -e "${ATTACK}\t${DATASET}\tincomplete\t${OUT_DIR}" >> "${SUMMARY_FILE}"
                fail_count=$((fail_count + 1))
            fi
        else
            status=$?
            echo "[FAIL:${status}] ${ATTACK} ${DATASET}" >&2
            echo -e "${ATTACK}\t${DATASET}\tfail_${status}\t${OUT_DIR}" >> "${SUMMARY_FILE}"
            fail_count=$((fail_count + 1))
        fi
    done
done

echo "[DONE] ok=${ok_count} skipped=${skip_count} failed=${fail_count}"
echo "[SUMMARY] ${SUMMARY_FILE}"
