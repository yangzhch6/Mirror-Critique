from datasets import load_dataset
import pandas as pd
import json
from math_verify import parse, verify
from tqdm import tqdm
import argparse

instruct_sys_verify = """You are an expert mathematics tutor who always thinks step-by-step. You will be shown: Question, Ground Truth (hidden from the student), Solution.
Your task:
* Analyze the Solution according to the Ground Truth. But do not mention "ground truth", "correct answer", "official solution", etc.
* Produce a numbered step-by-step analysis of the Solution, explaining why it is correct or incorrect.
* End with a single line containing only
\\boxed{True}  — if the boxed answer in the Solution is correct,
\\boxed{False} — otherwise.
"""

instruct_user_verify = """## Question:
{QUESTION}

## Ground Truth (for your reference only—do not reveal):
{GROUND_TRUTH}

## Solution:
{SOLUTION}

Now give your step-by-step Analysis followed by the boxed judgment.
"""

nothink_bsft_sys_prompt = 'You are a student learning how to solve problems step by step. You should study the Expert Thought Process and commit it firmly to memory. There is a significant gap between the text style of the Expert Thought Process and your own problem-solving text style, but you still need to study it diligently and memorize the Expert\'s problem-solving process.'

nothink_bsft_user_prompt = '## Question:\n{QUESTION}\n\nNow, you should recite the Expert Thought Process although it is significantly different from your own problem-solving text style.'

# nothink_bsft_assistant_prompt = '## Expert Thought Process:\n{GROUND_TRUTH}'


no_template_gt_input = 'You are a student learning how to solve problems step by step. You should study the Expert Thought Process and commit it firmly to memory. There is a significant gap between the text style of the Expert Thought Process and your own problem-solving text style, but you still need to study it diligently and memorize the Expert\'s problem-solving process.\n\n\n## Question:\n{QUESTION}\n\nNow, you should recite the Expert Thought Process although it is significantly different from your own problem-solving text style.'

no_template_gt_output = '## Expert Thought Process:\n{GROUND_TRUTH}'



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



def format_no_template_gt_input(problem):
    return no_template_gt_input.replace('{QUESTION}', problem)

def format_no_template_gt_output(ground_truth):
    return no_template_gt_output.replace('{GROUND_TRUTH}', ground_truth.strip())

def format_template_gt_input(problem):
    return [
        {
            "role": "system",
            "content": nothink_bsft_sys_prompt
        },
        {
            "role": "user",
            "content": nothink_bsft_user_prompt.replace('{QUESTION}', problem)
        }
    ]

def format_template_gt_output(ground_truth):
    return [
        {
            "role": "assistant",
            # "content": nothink_bsft_assistant_prompt.replace('{GROUND_TRUTH}', ground_truth.strip())
            "content": ground_truth.strip()
        }
    ]



def get_args():
    parser = argparse.ArgumentParser(description="Generate verify data parquet.")
    # parser.add_argument("--data_source", type=str, required=True,
    #                     help="Name of the data source, e.g. Qwen2.5-7B-openr1")
    parser.add_argument("--verify_file_path", type=str, required=True,
                        help="Path template for verify files, e.g. '/path/{}.json'")
    parser.add_argument("--ori_data_path", type=str,
                        default="/mnt/weka/home/yongxin.wang/workspace/lark/RLVR-Data/original_datasets/openr1.json",
                        help="Path to the original openr1.json")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save the output parquet")
    parser.add_argument("--start_step", type=int, required=True)
    parser.add_argument("--round_step_count", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--use_template", choices=["True", "False"],
                        default="False", help="Format of input data")
    parser.add_argument("--think", choices=["True", "False"],
                        default="False", help="Format of input data")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.use_template == 'False':
        format_gt_input_func = format_no_template_gt_input
        format_gt_output_func = format_no_template_gt_output
    elif args.use_template == 'True':
        format_gt_input_func = format_template_gt_input
        format_gt_output_func = format_template_gt_output


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


    formatted_sft_gt_data = []
    scores = []
    for line in tqdm(verify_data):
        scores += line["score"]

        line_acc = sum(1 for score in line["score"] if score > 0) / len(line["score"])
        if line_acc > args.threshold:
            continue 
            
        if args.use_template == "True":
            question = line["input"].split("assistant\n")[0].split("user\n")[-1].strip()
        else:
            question = line["input"].split("## Question:\n")[1].split("\nYou should include the final answer in \\boxed{} for closed-form results like multiple choices or mathematical results.")[0].strip()

        original_line = find_original_line(question, original_data)
        assert original_line is not None, f"Original line not found for question."

        if args.think == "True":
            solution_key = 'think_solution'
        else:
            solution_key = 'solution'
        ground_truth = original_line[solution_key] # think_solution
        # answer = original_line['answer']
        # answer_ = line["answer"]
        # assert answer == answer_, f"Answer mismatch: {answer} != {answer_}"

        formatted_sft_gt_data.append({
            "data_source": "sft_gt",
            "input": format_gt_input_func(question),
            "output": format_gt_output_func(ground_truth),
            'ability': 'math',
            'extra_info': {'index': len(formatted_sft_gt_data), 'split': 'default'}
        })

    formatted_data = formatted_sft_gt_data
    print(formatted_data[0])
    print("Formatted sft data length:", len(formatted_data))

    # save
    df = pd.DataFrame(formatted_data)
    df.to_parquet(args.output_path)

    # save validation set
    val_df = df.sample(frac=0.001, random_state=42)
    val_output_path = args.output_path.replace(".parquet", "_val.parquet")
    val_df.to_parquet(val_output_path)
    print("Validation set saved to:", val_output_path)