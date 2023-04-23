seed=12 # Seed for parameters random init

# Training
for prompt_id in {1..8}
# for prompt_id in 1
do
    PYTHONPATH=./ python src/tools/train.py \
    --seed ${seed} \
    --test_prompt_id ${prompt_id}
done

# PYTHONPATH=./ python src/tools/train.py \
# --seed ${seed} \
# --test_prompt_id 4