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


def get_args():
    parser = argparse.ArgumentParser(description="Generate verify data parquet.")
    parser.add_argument("--use_template", type=str, choices=["True", "False"],
                        default="False", help="Format of input data")
    parser.add_argument("--critique_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

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

    critique_data = load_json(args.critique_data_path)
    
    formatted_sft_gt_data = []
    formatted_sft_critique_data = []
    for line in tqdm(critique_data):
        question = line["input"].split("## Question:\n")[1].split("## Ground Truth (for your reference only—do not reveal):")[0].strip()
        solution = line["input"].split("## Solution:\n")[1].split("Now give your step-by-step Analysis followed by the boxed judgment.")[0].strip()
        ground_truth = line["input"].split("## Ground Truth (for your reference only—do not reveal):")[1].split("## Solution:")[0].strip()

        ## add gt data
        formatted_sft_gt_data.append({
            "data_source": "sft_gt",
            "input": format_gt_input_func(question),
            "output": format_gt_output_func(ground_truth),
            'ability': 'math',
            'extra_info': {'index': len(formatted_sft_gt_data), 'split': 'default'}
        })

        ## add critique data
        for idx in range(len(line["output"])):
            if line["score"][idx] > 0:
                output = line["output"][idx]
                formatted_sft_critique_data.append({
                    "data_source": "sft_critique",
                    "input": format_critique_input_func(question, solution),
                    "output": format_critique_output_func(ground_truth, output),
                    'ability': 'math',
                    'extra_info': {'index': len(formatted_sft_critique_data), 'split': 'default'}
                })
                break
            else:
                continue

    formatted_data = formatted_sft_gt_data + formatted_sft_critique_data
    print(formatted_data[0])
    print("Formatted critique data length:", len(formatted_data))

    # save
    df = pd.DataFrame(formatted_data)
    df.to_parquet(args.output_path)

    # save validation set
    val_df = df.sample(frac=0.001, random_state=42)
    val_output_path = args.output_path.replace(".parquet", "_val.parquet")
    val_df.to_parquet(val_output_path)
    print("Validation set saved to:", val_output_path)