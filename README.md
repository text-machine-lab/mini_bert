To run this script please first do pip install -e .

Main folders:
1. `./data_filtration_script`: this folder contains python scripts used for filtering various text corpora based on a vocabulary file
2. `./tokenizer_selection_script`: this folder contains python scripts written for training tokenizers and measure various metrics based on the tokenized text. The experiments conducted to find best-suited tokenizer were conducted using these scripts.
3. `./output_dir/honest-leaf-127`: this folder consists of the best performing model
4. `./`: parent directory, consists of all pre-training and glue evaluation scripts
    - the train_wnli.py script was initially used for fine tuning, but it has some issues.
    - The run_glue.py script is now used for fine tuning.
    - The data processing script prepares the raw data to the hugging face format we need.
    - The train.py script uses a Lamdba scheduler and was used for the wandb sweeps.
    - train_.py script has gradient accumulation and NOAM scheduler


Sample command for train.py
`python3 train.py --beta2=0.95 --learning_rate=0.00005 --max_train_steps=10 --output_dir=output_dir --tokenizer_path=tokenizers/Sentence_13k --batch_size=10 --debug --dataset_path=data/formatted_data_new`

Sample command for train_.py
`python3 train_.py --dataset_path=./formatted_data_new --tokenizer_path=./tokenizer_selection_scripts/Tokenizer_files/roberta-base_17000 --device_index=0 --fixed_seed_val=2 --max_train_steps=2000 --warmup_percent=0.05 --batch_size=320 --grad_acc_steps=1 --beta2=0.95 --learning_rate=8 --weight_decay=0.1 --eval_every_steps=300 --logging_steps=1 --wandb_project=mini_bert_overfitting_check --save_checkpoint_evey_steps=3000`

Sample command for run_glue.py
`python3 run_glue_nofilter.py --model_name=output_dir/wobbly-shadow-5 --tokenizer=./tokenizer_selection_scripts/Tokenizer_files/roberta-base_17000 --task_name="qnli" --do_train --do_eval --max_seq_length=128 --per_gpu_train_batch_size=32 --learning_rate=2e-4 --weight_decay=0 --num_train_epochs=5 --output_dir="output_dir/QNLI" --report_to=wandb --overwrite_output_dir --eval_steps=100 --seed=0`


The create_tokenizer script can be used to generate a tokenizer.
sample script:
`python3 create_tokenizer.py --vocab_size=13000 --load_dir=<path_to_training_vocab> --save_dir=<where_to_save_tokenizer>
    --tokenizer_type=<roberta, byte_level, sentence_piece, if none of these, then defaults to BPE>`
