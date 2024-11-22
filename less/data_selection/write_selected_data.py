import argparse
import os
import torch
from datasets import load_dataset, Features, Value
import json


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str,
                           nargs='+', help='The names of the training datasets on HF')
    argparser.add_argument('--target_task_names', type=str,
                           nargs='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    assert args.percentage is not None or args.max_samples is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = Features({
        "text": Value("string"),
        "source": Value("string")
    })

    # Load datasets from Hugging Face
    train_datasets = []
    dataset = load_dataset("UDACA/Code-Mixed-Dataset", split="train", features=features)
    train_datasets.append(dataset)

    for target_task in args.target_task_names:
        output_path = os.path.join(args.output_path, target_task)

        # Assuming scores are already computed for the datasets
        score_paths = [os.path.join(
            output_path, f"{task_name}_influence_score.pt") for task_name in args.train_file_names]

        num_samples = [len(dataset) for dataset in train_datasets]
        cumsum_num_samples = torch.cumsum(torch.tensor(num_samples), dim=0)

        total_samples = sum(num_samples)
        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
        else:
            data_amount_name = f"num{args.max_samples}"

        # Load scores
        all_scores = []
        for score_path in score_paths:
            score = torch.load(score_path, map_location=device)
            all_scores.append(score)
        all_scores = torch.cat(all_scores, dim=0)

        # Sort scores
        file_specific_index = torch.cat(
            [torch.arange(line_num) for line_num in num_samples]).to(device)
        data_from = torch.cat([torch.ones(line_num, dtype=torch.long)
                              * i for i, line_num in enumerate(num_samples)]).to(device)
        sorted_scores, sorted_index = torch.sort(
            all_scores, dim=0, descending=True)

        # Output sorted scores
        sorted_score_file = os.path.join(output_path, f"sorted.csv")
        data_from = data_from[sorted_index]
        sorted_index = file_specific_index[sorted_index]
        
        if not os.path.exists(sorted_score_file):
            with open(sorted_score_file, 'w', encoding='utf-8') as file:
                file.write("dataset name, index, score\n")
                for score, index, name in zip(sorted_scores, sorted_index, data_from):
                    file.write(
                        f"{args.train_file_names[name.item()]}, {index.item()}, {round(score.item(), 6)}\n")

        # Select top-k samples
        topk_scores, topk_indices = torch.topk(
            all_scores.float(), args.max_samples, dim=0, largest=True)

        final_index_list = sorted_index[:args.max_samples].tolist()
        final_data_from = data_from[:args.max_samples].tolist()

        with open(os.path.join(output_path, f"top_{data_amount_name}.jsonl"), 'w', encoding='utf-8', errors='ignore') as file:
            for index, data_from in zip(final_index_list, final_data_from):
                sample = train_datasets[data_from][index]
                
                # Write each sample as a JSON object with a 'text' key
                json_line = {"text": sample['text']}
                file.write(json.dumps(json_line) + "\n")

