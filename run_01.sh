#!/bin/bash

declare -a datasets=("cola" "sst2" "mrpc" "qqp" "stsb" "mnli" "qnli" "rte" "wnli")

declare -a modelx=("devoted-capybara-3" "laced-sun-4")

for h in 1 0
do
	for i in "${datasets[@]}"
	do
		for j in "${modelx[@]}"
		do
			for k in 0 1 2
			do
				if [[ "$j" == "devoted-capybara-3" ]]
				then
					d="checkpoint_at_16000"
				elif [[ "$j" == "laced-sun-4" ]]
				then
					d="checkpoint_at_8000"
				fi
				echo "$i $j $k $h $d"
				python3 run_glue_CustomConfigModel.py --model_name=output_dir/$j/other_checkpoints/$d --filter_glue=$h --tokenizer=./tokenizer_selection_scripts/Tokenizer_files/roberta-base_19000 --task_name="$i" --do_train --do_eval --max_seq_length=128 --per_gpu_train_batch_size=32 --learning_rate=2e-4 --weight_decay=0 --num_train_epochs=5 --output_dir="output_dir/$i-$j-vE-s$k-f$h-$d-new" --report_to=wandb --overwrite_output_dir --eval_steps=100 --seed=$k
			done
		done
	done
done