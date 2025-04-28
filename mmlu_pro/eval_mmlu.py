import json
import openai
import numpy as np
import time
import re
from pathlib import Path
import argparse

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


def solve_math_problems(input_str):
    # pattern = r"\d+\.?\d*"
    pattern = r"\\boxed{([A-Z])}"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

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
            return solution
    pattern = r"correct option is (\w)"
    matches = re.findall(pattern, input_str)
    if matches:
        solution = matches[-1].upper()
        return solution

    return solution


def compute_accuracy(gt, pred_solutions):
    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)

            if pred_answer is not None:
                pred_answers.append(pred_answer)

        if not pred_answers:
            print(pred_solutions, gt)
            return 0
        # print(gt, pred_answers)
        pred_answer = most_frequent(pred_answers)
        # pred_answer = pred_answers[0]
    else:
        pred_answer = parse_answer(pred_solutions)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solutions)

    if gt == pred_answer:
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

def check_same_ans(responses):
    pred_answers = []
    for response in responses:
        pred_solution = response[-1]['content']

        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        pred_answers.append(pred_answer)

    print(pred_answers)

    return len(set(pred_answers)) == 1

def select_multi_step_questions(json_file: str):
    response_dict = json.load(open(json_file, "r"))
    questions = list(response_dict.keys())
    question_ans_pair = []
    for question in questions:
        responses, gt = response_dict[question]
        if len(responses[0]) > 3:
            question_ans_pair.append((question, gt))
    return question_ans_pair

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

        accurate = compute_accuracy(answer, pred_solutions)

        if accurate is not None:
            accuracies.append(float(accurate))
        else:
            import pdb
            pdb.set_trace()
            print(gt)

    print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
