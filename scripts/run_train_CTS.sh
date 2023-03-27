seed=12 # Seed for parameters random init
model_name="CTS"

# Training
# for prompt_id in {1..8}
# do
#     PYTHONPATH=./ python src/tools/train.py \
#     --model_name ${model_name} \
#     --seed ${seed} \
#     --test_prompt_id ${prompt_id}
# done

PYTHONPATH=./ python src/tools/train.py \
--model_name ${model_name} \
--seed ${seed} \
--test_prompt_id 1