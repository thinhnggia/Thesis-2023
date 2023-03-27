set -e

PYTHONPATH=./ python src/tools/prepare_ASAP_data.py \
    --traits_path src/data/ASAP/traits \
    --paes_path src/data/ASAP/PAES \
    --orig_path src/data/ASAP/training_set_rel3.tsv \
    --output_path src/data/ASAP/final_data