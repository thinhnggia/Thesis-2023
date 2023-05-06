seed=12 # Seed for parameters random init

# Training
# for prompt_id in {3..8}
for prompt_id in 6 8
do
    PYTHONPATH=./ python src/tools/train.py \
    --seed ${seed} \
    --test_prompt_id ${prompt_id}
done