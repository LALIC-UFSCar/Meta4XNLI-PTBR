#!/bin/bash

USER_PROMPTS=(
    resources/user_prompts/upt1.txt
)

SYSTEM_PROMPTS=(
    resources/system_prompts/spt1.txt
    resources/system_prompts/spt2.txt
)

CONFIGS=(
    resources/request_configs/r1.yaml
)

MODELS=(
    meta-llama/llama-4-maverick-17b-128e-instruct
    mistral-saba-24b
    meta-llama/llama-4-scout-17b-16e-instruct
    llama-3.3-70b-versatile
    gemma2-9b-it
    llama-3.1-8b-instant
    allam-2-7b
    llama3-70b-8192
    llama3-8b-8192
)

for USER_PROMPT in "${USER_PROMPTS[@]}"
do
    for SYSTEM_PROMPT in "${SYSTEM_PROMPTS[@]}"
    do
        for CONFIG in "${CONFIGS[@]}"
        do
            for MODEL in "${MODELS[@]}"
            do

                USER_PROMPT_NAME="${USER_PROMPT##*/}"
                USER_PROMPT_NAME="${USER_PROMPT_NAME%.*}"
                SYSTEM_PROMPT_NAME="${SYSTEM_PROMPT##*/}"
                SYSTEM_PROMPT_NAME="${SYSTEM_PROMPT_NAME%.*}"
                CONFIG_NAME="${CONFIG##*/}"
                CONFIG_NAME="${CONFIG_NAME%.*}"

                NAME=${MODEL//\//_}_${CONFIG_NAME}_${SYSTEM_PROMPT_NAME}_${USER_PROMPT_NAME}_meta4xnli_dev_head_100
                echo "Running ${NAME}"

                # translation
                python3 -m get_answers \
                --dataset data/meta4xnli/detection/splits/samples/en/meta4xnli_dev_head_100.jsonl \
                --system "$SYSTEM_PROMPT" \
                --user "$USER_PROMPT" \
                --config "$CONFIG" \
                --model "$MODEL" \
                --output data/meta4xnli_ptbr/en2es/${NAME}.jsonl \
                #--sleep 2

                # evaluation
                python3 -m evaluate_translations \
                --source data/meta4xnli/detection/splits/samples/en/meta4xnli_dev_head_100.jsonl \
                --reference data/meta4xnli/detection/splits/samples/es/meta4xnli_dev_head_100.jsonl \
                --hypothesis data/meta4xnli_ptbr/en2es/${NAME}.jsonl \
                --metrics bleu chrf chrf2 ter meteor rouge \
                --full_results results/en2es/full_results/${NAME}.tsv \
                --summary_results results/en2es/summary_results.tsv \
                --index ${NAME}
            done
        done
    done
done


