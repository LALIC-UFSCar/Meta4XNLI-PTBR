python get_annotations.py \
--dataset data/meta4xnli/detection/splits/samples/es/meta4xnli_train_metaphorical_sample_1000.jsonl \
--system resources_annotation/system_prompts/spa1.txt \
--user resources_annotation/user_prompts/upa1.txt \
--config resources_annotation/request_configs/ra1.yaml \
--model meta-llama/llama-4-scout-17b-16e-instruct \
--output data/meta4xnli_ptbr/annotations/es/meta4xnli_train_metaphorical_sample_1000_head_10.jsonl \
--sleep 0 \
--sample_size 2 \
--client groq
