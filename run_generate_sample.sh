size=1000
labels=(literal metaphorical)
split=train
splits_path=data/meta4xnli/detection/splits

# create folder "samples", if it doesn't exist
if [ ! -d ${splits_path}/samples ]; then
  mkdir -p ${splits_path}/samples
fi
# create folder "en", if it doesn't exist
if [ ! -d ${splits_path}/samples/en ]; then
  mkdir -p ${splits_path}/samples/en
fi
# create folder "es", if it doesn't exist
if [ ! -d ${splits_path}/samples/es ]; then
  mkdir -p ${splits_path}/samples/es
fi

for label in "${labels[@]}"
do
    python -m generate_sample \
    --source_input ${splits_path}/en/meta4xnli_${split}.jsonl \
    --target_input ${splits_path}/es/meta4xnli_${split}.jsonl \
    --source_output ${splits_path}/samples/en/meta4xnli_${split}_${label}_sample_${size}.jsonl \
    --target_output ${splits_path}/samples/es/meta4xnli_${split}_${label}_sample_${size}.jsonl \
    --label ${label} \
    --size ${size}
done
