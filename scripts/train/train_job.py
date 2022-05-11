import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Script to run train job for seq2seq (TST) or classifier (STI) models.")
    parser.add_argument('-t', '--task', type=str, help='Select which task to run: seq2seq or classifier.', choices=["seq2seq", "classifier"])
    parser.add_argument('-n', '--name', type=str, help='Specify the name of the model and directory where output will be save.')

    args = parser.parse_args()

    if args.task == "seq2seq":
        print(f'Running seq2seq training. Model Name: {args.name}')
        os.system(f"./scripts/train/seq2seq/train_seq2seq.sh {args.name}")

    elif args.task == "classifier":
        print(f'Running classifier training. Model Name: {args.name}')
        os.system(f"./scripts/train/classifier/train_classifier.sh {args.name}")

    else:
        raise ValueError('Must select either seq2seq or classifier.')

if __name__ == "__main__":
    main()