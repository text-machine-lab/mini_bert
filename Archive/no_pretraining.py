import argparse

from transformers import AutoTokenizer, RobertaTokenizerFast, RobertaForMaskedLM


def parse_args():
    """This function creates argument parser and parses the scrip input arguments.
        This is the most common way to define input arguments in python.

        To change the parameters, pass them to the script, for example:

        python cli/train.py \
            --output_dir output_dir \
            --weight_decay 0.01

        """
    parser = argparse.ArgumentParser(
        description="Train machine translation transformer model"
    )

    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Where to store the final model. "
            "Should contain the source and target tokenizers in the following format: "
            r"output_dir/{source_lang}_tokenizer and output_dir/{target_lang}_tokenizer. "
            "Both of these should be directories containing tokenizer.json files."
        ),
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="path to tokenizer.  If not provided, default BERT tokenizer will be used.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path) if args.tokenizer_path else RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    model = RobertaForMaskedLM.from_pretrained('phueb/BabyBERTa-3')
    config = model.config
    config.vocab_size = len(tokenizer) + 1
    model = RobertaForMaskedLM(config)
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
