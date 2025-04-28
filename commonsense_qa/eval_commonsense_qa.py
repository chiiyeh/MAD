import json
import openai
import numpy as np
import time
import re
from pathlib import Path
import argparse

def parse_answer(input_str):
    pattern = r'is .*?\((\w)\)'
    # pattern = r'###### (\w)'
    matches = re.findall(pattern, input_str)

    solution = None
    # print("predicted solution")
    # print(input_str)
    # print("matches")
    # print(matches)

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    return solution

def compute_accuracy(answer_key, extracted_answers):
    pred_answer = most_frequent(extracted_answers)
    if answer_key == pred_answer:
        return 1
    else:
        return 0

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run scoring for file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Filepath for scoring."
    )

    parser.add_argument(
        "--round",
        type=int,
        default=0,
        help="Round number for scoring."
    )
    args = parser.parse_args()
    # Load the JSON file
    json_file = args.input_file
    response_dict = json.load(json_file.open("r"))
    questions = list(response_dict.keys())
    print("total questions:", len(questions))

    round = args.round
    if round > 0:
        print("Evaluating at round:", round)
    else:
        print("Evaluating at final round")

    accuracies = []

    for question in questions:
        question_details = response_dict[question]

        pred_solutions = []
        responses = question_details["agent_contexts"]
        answer = question_details["answer"]
        for response in responses:
            if round > 0:
                pred_solution = response[2*(round-1) + 1]['content']
            else:
                pred_solution = response[-1]['content']
            pred_solutions.append(pred_solution)

        accurate = compute_accuracy(answer, question_details["extracted_answers"])

        if accurate is not None:
            accuracies.append(float(accurate))
        else:
            import pdb
            pdb.set_trace()

    print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
