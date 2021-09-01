#!/usr/bin/env bash
dataset=../data/AMR/amr_2.0
for a in $(seq 1.5 0.1 2.0); do
    echo $a
    python3 work.py --test_data ${dataset}/dev.txt.features.preproc.json\
        --test_batch_size 2000\
        --load_path $1\
        --beam_size 12\
        --alpha $a\
        --max_time_step 100\
        --output_suffix _dev_out
    python3 postprocess.py --golden_file ../data/AMR/amr_2.0/dev.txt.features \
        --pred_file ${1}_dev_out \
        --output
done
