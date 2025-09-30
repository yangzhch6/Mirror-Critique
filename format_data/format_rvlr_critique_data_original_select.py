from datasets import load_dataset
import pandas as pd
import json
from math_verify import parse, verify
from tqdm import tqdm
import argparse


no_template_critique_input = """## Instruct:
You are an expert mathematics tutor who always thinks step-by-step. You will be shown: Question and its Solution.
Your task:
* Analyze the Solution according to the Question
* Produce a numbered step-by-step analysis of the Solution, explaining why it is correct or incorrect.
* End with a single line containing only
\\boxed{True}  — if the boxed answer in the Solution is correct,
\\boxed{False} — otherwise.


## Question:
{QUESTION}

## Solution:
{SOLUTION}
"""

no_template_critique_output = """Now I will give the step-by-step Analysis followed by the boxed judgment.
## Analysis:
{ANALYSIS}
"""


no_template_gt_input = """## Instruct:
You are a student who is learning how to solve questions step by step. You should study the Ground Truth Solution and commit it firmly to memory.

## Question:
{QUESTION}

Now I will study and memorize the Ground Truth Solution.
"""

no_template_gt_output = """## Ground Truth Solution:
{GROUND_TRUTH}
"""

## ------------------------

template_critique_sys = """You are an expert mathematics tutor who always thinks step-by-step. You will be shown: Question and its Solution.
Your task:
* Analyze the Solution according to the Question
* Produce a numbered step-by-step analysis of the Solution, explaining why it is correct or incorrect.
* End with a single line containing only
\\boxed{True}  — if the boxed answer in the Solution is correct,
\\boxed{False} — otherwise."""

template_critique_user = """## Question:
{QUESTION}

## Solution:
{SOLUTION}
"""

template_critique_assistant = """Now I will give the step-by-step Analysis followed by the boxed judgment.
## Analysis:
{ANALYSIS}
"""


template_gt_sys = """You are a student who is learning how to solve questions step by step. You should study the Ground Truth Solution and commit it firmly to memory."""

template_gt_user = """## Question:
{QUESTION}

Now you should study and memorize the Ground Truth Solution.
"""

template_gt_assistant = """## Ground Truth Solution:
{GROUND_TRUTH}
"""


def load_json(file_path):
    """
    Load a JSON file and return its content.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def find_original_line(question, original_data):
    for line in original_data:
        if question.strip() in line['problem']:
            return line
    return None

def delete_redundant(candidate_solution, candidate_score):
    collected_solution = []
    collected_score = []
    collected_ans = []
    # 选择K个正样本
    for idx, solution in enumerate(candidate_solution):
        if candidate_score[idx] > 0 and len(collected_solution) < 1:
            collected_solution.append(solution)
            collected_score.append(candidate_score[idx])
            collected_ans.append(parse(solution))
    
    # 选择parse结果不同的负样本
    for idx, solution in enumerate(candidate_solution):
        if candidate_score[idx] > 0:
            continue
        
        boxed_answer = extract_final_boxed_answer(solution)
        
        if boxed_answer == "":
            continue
        
        parsed_ans = parse(solution)
        not_exist = True
        for ans in collected_ans:
            if verify(parsed_ans, ans):
                not_exist = False
                break
        if not_exist:
            collected_solution.append(solution)
            collected_score.append(candidate_score[idx])
            collected_ans.append(parsed_ans)

    return collected_solution, collected_score

def extract_final_boxed_answer(solution_str: str):
    """
    Extracts the final boxed answer from the solution string.
    Assumes the answer is formatted as \\boxed{answer}.
    """
    if "boxed{" in solution_str:
        return solution_str.split("boxed{")[-1].split("}")[0].strip()
    else:
        return ""


def format_no_template_critique_input(problem, solution):
    return no_template_critique_input.replace('{QUESTION}', problem).replace('{SOLUTION}', solution)

def format_no_template_critique_output(output):
    return no_template_critique_output.replace('{ANALYSIS}', output.strip())


def format_no_template_gt_input(problem):
    return no_template_gt_input.replace('{QUESTION}', problem)

def format_no_template_gt_output(ground_truth):
    return no_template_gt_output.replace('{GROUND_TRUTH}', ground_truth.strip())


def format_template_critique_input(problem, solution):
    return [
        {
            "role": "system",
            "content": template_critique_sys
        },
        {
            "role": "user",
            "content": template_critique_user.replace('{QUESTION}', problem).replace('{SOLUTION}', solution)
        }
    ]

def format_template_critique_output(output):
    return [
        {
            "role": "assistant",
            "content": template_critique_assistant.replace('{ANALYSIS}', output.strip())
        }
    ]

def format_template_gt_input(problem):
    return [
        {
            "role": "system",
            "content": template_gt_sys
        },
        {
            "role": "user",
            "content": template_gt_user.replace('{QUESTION}', problem)
        }
    ]

def format_template_gt_output(ground_truth):
    return [
        {
            "role": "assistant",
            "content": template_gt_assistant.replace('{GROUND_TRUTH}', ground_truth.strip())
        }
    ]

def find_original_data_acc(question, original_data):
    for line in original_data:
        if question.strip() in line['input']:
            return sum(line["score"])/len(line["score"])
    return -1

def get_args():
    parser = argparse.ArgumentParser(description="Generate verify data parquet.")
    parser.add_argument("--use_template", type=str, choices=["True", "False"],
                        default="True", help="Format of input data")
    parser.add_argument("--original_data_path", type=str, required=True)
    parser.add_argument("--up_threshold", type=float, required=True)
    parser.add_argument("--down_threshold", type=float, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--start_step", type=int, required=True)
    parser.add_argument("--round_step_count", type=int, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    if args.use_template == 'False':
        format_gt_input_func = format_no_template_gt_input
        format_gt_output_func = format_no_template_gt_output
        format_critique_input_func = format_no_template_critique_input
        format_critique_output_func = format_no_template_critique_output
    elif args.use_template == 'True':
        format_gt_input_func = format_template_gt_input
        format_gt_output_func = format_template_gt_output
        format_critique_input_func = format_template_critique_input
        format_critique_output_func = format_template_critique_output
    else:
        raise ValueError("Invalid use_template type. Choose 'True' or 'False'.")

    # original_data = load_json(args.original_data_path)
    original_data = []
    for i in range(args.start_step+1, args.start_step + args.round_step_count + 1):
        original_data_path = args.original_data_path.format(str(i))
        step_data = load_json(original_data_path)
        original_data += step_data
    print(len(original_data))
    # print(original_data[0])
    # exit()

    
    formatted_data = []

    judge_true_count = 0
    judge_false_count = 0

    for line in tqdm(original_data):
        line_acc = sum(line["score"]) / len(line["score"])
        if line_acc >= args.up_threshold or line_acc <= args.down_threshold:
            continue

        if args.use_template == "True":
            question = line["input"].split("assistant\n")[0].split("user\n")[-1].strip()
        else:
            question = line["input"].split("## Question:\n")[1].split("\nYou should include the final answer in \\boxed{} for closed-form results like multiple choices or mathematical results.")[0].strip()

        candidate_solution_lst = []
        candidate_score_lst = []
        for idx in range(len(line["output"])):
            if line["response_length"][idx] > 2560:
                continue
            candidate_solution_lst.append(line["output"][idx])
            candidate_score_lst.append(line["score"][idx])

        for candidate_solution, candidate_score in zip(candidate_solution_lst, candidate_score_lst):
            ## add critique data
            for idx in range(len(line["output"])):
                output = line["output"][idx]
                formatted_data.append({
                    "data_source": "rlvr_critique_select",
                    "prompt": format_critique_input_func(question, candidate_solution),
                    # "target": "",
                    'ability': 'math',
                    'reward_model': {
                        'ground_truth': 'true' if candidate_score > 0 else 'false',
                        'style': 'rule'
                    },
                    'extra_info': {'index': len(formatted_data), 'split': 'default'}
                })
                if candidate_score > 0:
                    judge_true_count += 1
                elif candidate_score == 0:
                    judge_false_count += 1
                else:
                    raise ValueError("Invalid answer type. Choose 'true' or 'false'.")
                break

    print(judge_true_count, judge_false_count)
    # print(formatted_data[0])
    print("Formatted critique data length:", len(formatted_data))

    # save
    df = pd.DataFrame(formatted_data)
    df.to_parquet(args.output_path)

