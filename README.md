To run this script please first do pip install -e .

The data processing script preps the raw data to the hugging face format we need.
The train.py script uses a Lamdba scheduler and was used for the wandb sweeps.
train_.py script has gradienta accumulation and a scheduler that behaves more like the noam scheduler.

Sample command for train.py

python3 train.py --beta2=0.95 --learning_rate=0.00005 --max_train_steps=10 --output_dir=output_dir --tokenizer_path=tokenizers/Sentence_13k --batch_size=10 --debug --dataset_path=data/formatted_data_new

the train_wnli.py script was initially used for fine tuning, but it has some issues.

The run_glue.py script is now used for fine tuning.

The create_tokenizer script can be used to generate a tokenizer.
sample script:
python3 create_tokenizer.py --vocab_size=13000 --load_dir=<path_to_training_vocab> --save_dir=<where_to_save_tokenizer>
    --tokenizer_type=<roberta, byte_level, sentence_piece, if none of these, then defaults to BPE>