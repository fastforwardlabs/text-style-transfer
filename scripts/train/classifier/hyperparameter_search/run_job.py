import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Script to run train job for classifier (STI) models for hyperparameter search")
    parser.add_argument('-n', '--name', type=str, help='Specify the name of the trial script to run. (ex. trial-a.sh)')

    args = parser.parse_args()

    print(f'Running {args.name}')
    os.system(f"./scripts/train/classifier/hyperparameter_search/{args.name}")

if __name__ == "__main__":
    main()