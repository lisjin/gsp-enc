dataset=../data/AMR/amr_2.0
python3 work.py --test_data ${dataset}/test.txt.features.preproc.json\
               --test_batch_size 2000\
               --load_path $1\
               --beam_size 12\
               --alpha 1.7\
               --max_time_step 100\
               --output_suffix _test_out
