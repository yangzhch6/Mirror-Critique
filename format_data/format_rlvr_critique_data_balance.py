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


def format_no_template_critique_input(problem, solution):
    return no_template_critique_input.replace('{QUESTION}', problem).replace('{SOLUTION}', solution)

def format_no_template_critique_output(output):
    return no_template_critique_output.replace('{ANALYSIS}', output.strip())



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

def format_instruct_verify(problem, ground_truth, solution):
    return [
        {
            "role": "system",
            "content": instruct_sys_verify
        },
        {
            "role": "user",
            "content": instruct_user_verify.replace('{QUESTION}', problem).replace('{GROUND_TRUTH}', ground_truth).replace('{SOLUTION}', solution)
        }
    ]



def get_args():
    parser = argparse.ArgumentParser(description="Generate verify data parquet.")
    # parser.add_argument("--data_source", type=str, required=True,
    #                     help="Name of the data source, e.g. Qwen2.5-7B-openr1")
    parser.add_argument("--verify_file_path", type=str, required=True,
                        help="Path template for verify files, e.g. '/path/{}.json'")
    parser.add_argument("--ori_data_path", type=str,
                        default="/mnt/weka/home/renxi.wang/yxwang/lark/RLVR-Data/openr1.json",
                        help="Path to the original openr1.json")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save the output parquet")
    parser.add_argument("--max_item_count", type=int, default=128)
    parser.add_argument("--start_step", type=int, required=True)
    parser.add_argument("--round_step_count", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--response_data_use_template", choices=["True", "False"],
                        default="False", help="Format of input data")
    parser.add_argument("--rlvr_verify_data_use_template", choices=["True", "False"],
                        default="False", help="Format of input data")
    parser.add_argument("--delete_redundant", choices=["True", "False"],
                        default="False", help="Format of input data")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.rlvr_verify_data_use_template == 'False':
        format_critique_input_func = format_no_template_critique_input
        format_critique_output_func = format_no_template_critique_output
    elif args.rlvr_verify_data_use_template == 'True':
        format_critique_input_func = format_template_critique_input
        format_critique_output_func = format_template_critique_output
    else:
        raise ValueError("Invalid use_template type. Choose 'True' or 'False'.")

    data_range = [args.start_step, args.start_step + args.round_step_count]

    original_data = load_json(args.ori_data_path)

    verify_data = []
    for i in range(data_range[0]+1, data_range[1]+1):
        print(i)
        verify_file_path_ = args.verify_file_path.format(str(i))
        step_data = load_json(verify_file_path_)
        verify_data += step_data
        print(verify_file_path_)
    print("Loaded verify data length:", len(verify_data))

    formatted_verify_data = []
    scores = []
    for line in tqdm(verify_data):
        scores += line["score"]

        line_acc = sum(1 for score in line["score"] if score > 0) / len(line["score"])
        if line_acc > args.threshold:
            continue 
        
        if line_acc < 1e-9 or line_acc > 1 - 1e-9:
            continue
            
        if args.response_data_use_template == "True":
            question = line["input"].split("assistant\n")[0].split("user\n")[-1].strip()
        else:
            question = line["input"].split("## Question:\n")[1].split("\nYou should include the final answer in \\boxed{} for closed-form results like multiple choices or mathematical results.")[0].strip()

        original_line = find_original_line(question, original_data)
        assert original_line is not None, f"Original line not found for question."

        ground_truth = original_line['solution']
        answer = original_line['answer']
        answer_ = line["answer"]
        # assert answer == answer_, f"Answer mismatch: {answer} != {answer_}"

        candidate_solution = []
        candidate_score = []
        for idx in range(len(line["output"])):
            if line["response_length"][idx] > 2560: # 2560
                continue
            candidate_solution.append(line["output"][idx])
            candidate_score.append(line["score"][idx])
        
        # if args.delete_redundant == "True":
        #     candidate_solution, candidate_score = delete_redundant(candidate_solution, candidate_score)
        
        # filter one correct and one wrong sample
        if all(score == 0 for score in candidate_score):
            continue
        if all(score > 0 for score in candidate_score):
            continue
        
        filtered_solution = []
        filtered_score = []
        for idx, score in enumerate(candidate_score):
            if score > 0:
                filtered_solution.append(candidate_solution[idx])
                filtered_score.append(candidate_score[idx])
                break
            else:
                continue
        for idx, score in enumerate(candidate_score):
            if score <= 0:
                filtered_solution.append(candidate_solution[idx])
                filtered_score.append(candidate_score[idx])
                break
            else:
                continue

        candidate_solution = filtered_solution
        candidate_score = filtered_score

        line_add_count = 0
        for idx, solution in enumerate(candidate_solution):
            # formatted_prompt = format_instruct_verify(question, ground_truth, solution)
            # formatted_verify_data.append({
            #     "data_source": "",
            #     "prompt": formatted_prompt,
            #     'ability': 'math',
            #     'reward_model': {
            #         'ground_truth': 'true' if candidate_score[idx] > 0 else 'false',
            #         'style': 'rule'
            #     },
            #     'extra_info': {'index': len(formatted_verify_data), 'split': 'default'}
            # })
            formatted_verify_data.append({
                "data_source": "rlvr_critique",
                "prompt": format_critique_input_func(question, solution),
                # "output": format_critique_output_func(ground_truth, output),
                'ability': 'math',
                'reward_model': {
                    'ground_truth': 'true' if candidate_score[idx] > 0 else 'false',
                    'style': 'rule'
                },
                'extra_info': {'index': len(formatted_verify_data), 'split': 'default'}
            })
            line_add_count += 1
            if line_add_count == args.max_item_count:
                break


    print(formatted_verify_data[0])
    print("Formatted verify data length:", len(formatted_verify_data))

    positive_count_before = sum(scores) 
    negative_count_before = len(scores) - positive_count_before

    postive_count = sum(1 for item in formatted_verify_data if item['reward_model']['ground_truth'] == 'true')
    negative_count = len(formatted_verify_data) - postive_count
    print(f"Positive samples before filtering: {positive_count_before}, after filtering: {postive_count}")
    print(f"Negative samples before filtering: {negative_count_before}, after filtering: {negative_count}")

    # save
    df = pd.DataFrame(formatted_verify_data)
    df.to_parquet(args.output_path)
