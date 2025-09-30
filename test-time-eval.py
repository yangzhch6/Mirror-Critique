import pandas as pd
import json
from math_verify import parse, verify
from tqdm import tqdm
import argparse

def load_json(file_path):
    """
    Load a JSON file and return its content.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_final_boxed_answer(solution_str: str):
    """
    Extracts the final boxed answer from the solution string.
    Assumes the answer is formatted as \\boxed{answer}.
    """
    if "boxed{" in solution_str:
        return solution_str.split("boxed{")[-1].split("}")[0].strip()
    else:
        return ""


def compute_verify_score(line):
    verify_score = []
    for output in line["output"]:
        output_final_boxed = extract_final_boxed_answer(output).lower()
        if output_final_boxed == "":
            continue
        if "true" == output_final_boxed:
            verify_score.append(1)
        elif "false" == output_final_boxed:
            verify_score.append(0)
        else:
            continue
    return verify_score #sum(verify_score) / len(verify_score)

K = 16
N = 16
file_path = "./checkpoints/rlvr-verify/Qwen2.5-Math-1.5B-L-sft-ckpt-balance-bsz1k/val_generations/240.json"
print(file_path)
data = load_json(file_path)

# pass_1
print("## Pass@1 Performance ##")
acc_count = 0
acc_count_source = {}
count_source = {}
for line in data:
    if line["data_source"] in count_source:
        count_source[line["data_source"]] += 1
    else:
        count_source[line["data_source"]] = 1

    if line["answer"] == "true":
        acc_count += 1
        if line["data_source"] in acc_count_source:
            acc_count_source[line["data_source"]] += 1
        else:
            acc_count_source[line["data_source"]] = 1

print(f"Overall Pass@1: {acc_count / len(data):.4f}")
for source in count_source:
    source_acc = acc_count_source[source] if source in acc_count_source else 0
    print(f"Source: {source}, | Pass@1: {source_acc / count_source[source]:.4f}")
print("-"*60)


# verify acc
print("## Verify Accuracy ##")
acc_count = 0
acc_count_source = {}
count_source = {}
for line in data:
    if line["data_source"] in count_source:
        count_source[line["data_source"]] += N
    else:
        count_source[line["data_source"]] = N

    acc_count += sum(line["score"])
    if line["data_source"] in acc_count_source:
        acc_count_source[line["data_source"]] += sum(line["score"])
    else:
        acc_count_source[line["data_source"]] = sum(line["score"])

print(f"Overall Verify Acc: {acc_count / N / len(data):.4f}")
for source in count_source:
    source_acc = acc_count_source[source] if source in acc_count_source else 0
    print(f"Source: {source}, | Verify Acc: {source_acc / count_source[source]:.4f}")
# print("-"*60)



# verify bin
print("## Verify Bin Accuracy ##")
verify_score_bin_count = [0] * (N+1) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
verify_score_bin_acc = [0] * (N+1) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for line in data:
    score = sum(line["score"])
    if line["answer"] == "true":
        score = score
        verify_score_bin_acc[int(score)] += 1
    else:
        score = N - score
    verify_score_bin_count[int(score)] += 1

for i in range(N+1):
    acc = verify_score_bin_acc[i] / verify_score_bin_count[i] if verify_score_bin_count[i] > 0 else 0
    print(f"Score: {i}, | {i/N:.3f}, | Count: {verify_score_bin_count[i]}, | Acc: {acc:.4f}")
print("-"*60)


# processing data
aggregated_data = {}
for line in data:
    key = line["input"].split("## Solution:")[0]
    if key in aggregated_data:
        aggregated_data[key].append(line)
    else:
        aggregated_data[key] = [line]

print("# lenth of aggregated_data:", len(aggregated_data))


# PassK
print("## Pass@K Performance ##")
acc_count = 0
acc_count_source = {}
count_source = {}
for key in aggregated_data:
    lines = aggregated_data[key]
    answers = [line["answer"] for line in lines[:K]]
    if "true" in answers:
        acc_count += 1
        source = lines[0]["data_source"]
        if source in acc_count_source:
            acc_count_source[source] += 1
        else:
            acc_count_source[source] = 1
    source = lines[0]["data_source"]
    if source in count_source:
        count_source[source] += 1
    else:
        count_source[source] = 1

print(f"Overall Pass@{K}: {acc_count / len(aggregated_data):.4f}")
for source in count_source:
    source_acc = acc_count_source[source] if source in acc_count_source else 0
    print(f"Source: {source}, | Pass@{K}: {source_acc / count_source[source]:.4f}")
print("-"*60)


# exit()

# aggregate to file
major_acc_count = 0
all_count = 0

voting_data = []
for key in tqdm(aggregated_data):
    # print(len(aggregated_data[key]))
    assert len(aggregated_data[key]) >= K
    ans_list = []
    ans_correct = []
    ans_count = []
    ans_verify_score_list = []
    avg_ans_verify_score_list = []

    for line in aggregated_data[key][:K]:
        solution = line["input"].split("## Solution:")[1].strip()
        boxed_answer = extract_final_boxed_answer(solution)
        if boxed_answer == "":
            continue
        
        verify_score_lst = compute_verify_score(line)
        if len(verify_score_lst) == 0:
            continue
        avg_verify_score = sum(verify_score_lst) / len(verify_score_lst)

        parsed_ans = parse(solution)

        not_exist = True
        for idx in range(len(ans_list)):
            if verify(parsed_ans, ans_list[idx]):
                ans_count[idx] += 1
                ans_verify_score_list[idx].append(verify_score_lst)
                avg_ans_verify_score_list[idx].append(avg_verify_score)
                not_exist = False
                break
        if not_exist:
            ans_list.append(parsed_ans)
            ans_correct.append(line["answer"] == "true")
            ans_count.append(1)
            ans_verify_score_list.append([verify_score_lst])
            avg_ans_verify_score_list.append([avg_verify_score])

    if len(ans_count) != 0:
        max_acc_count = max(ans_count)
        max_acc_index = ans_count.index(max_acc_count)
        if ans_correct[max_acc_index]:
            major_acc_count += 1
    all_count += 1

    voting_data.append({
        "input": key,
        "ans_list": [item[1] for item in ans_list],
        "ans_correct": ans_correct,
        "ans_count": [float(item) for item in ans_count],
        # "ans_verify_score_list": ans_verify_score_list,
        "avg_ans_verify_score_list": avg_ans_verify_score_list,
        "data_source": aggregated_data[key][0]["data_source"],
    })

# save as json
with open(file_path.replace(".json", "_k{}_n{}_vote.json".format(K,N)), "w") as f:
    json.dump(voting_data, f, indent=4)




data = voting_data
# Pass@K
print("## Pass@K Performance ##")
acc_count = 0
acc_count_source = {}
count_source = {}
for line in data:
    if line["data_source"] in count_source:
        count_source[line["data_source"]] += 1
    else:
        count_source[line["data_source"]] = 1

    if True in line["ans_correct"]:
        acc_count += 1
        if line["data_source"] in acc_count_source:
            acc_count_source[line["data_source"]] += 1
        else:
            acc_count_source[line["data_source"]] = 1
print(f"Overall Pass@K Acc: {acc_count/len(data)}")
for source in acc_count_source:
    print(f"Data Source: {source}, Count: {count_source[source]}, Pass@K Acc: {acc_count_source[source]/count_source[source]}")
print('-'*60)


# major@K & vote@K
print("## Major@K & Vote@K Performance ##")
major_acc_count = 0
major_acc_count_source = {}
major_acc_count_source = {}
count_source = {}
# print(data[0])
for line in data:
    if line["data_source"] in count_source:
        count_source[line["data_source"]] += 1
    else:
        count_source[line["data_source"]] = 1

    if len(line["ans_count"]) == 0:
        continue
    
    max_acc_count = max(line["ans_count"])
    # max_acc_index = line["ans_count"].index(max_acc_count)
    max_acc_index_lst = get_index(line["ans_count"], max_acc_count)
    extract_result = [line["ans_correct"][idx] for idx in max_acc_index_lst]
    
    if True in extract_result:
        major_acc_count += 1/len(extract_result)
        if line["data_source"] in major_acc_count_source:
            major_acc_count_source[line["data_source"]] += 1/len(extract_result)
        else:
            major_acc_count_source[line["data_source"]] = 1/len(extract_result)

print(f"Overall Major@K Acc: {major_acc_count/len(data)}")
for source in major_acc_count_source:
    print(f"Data Source: {source}, Count: {count_source[source]}, Major@K Acc: {major_acc_count_source[source]/count_source[source]}")
print('-'*60)


# vote_acc_count = 0
# for line in data:
#     if len(line["ans_count"]) == 0:
#         continue
    
#     avg_ans_verify_score_list = [sum(item)/len(item) for item in line["avg_ans_verify_score_list"]]
#     weighted_vote = [a*b for a,b in zip(line["ans_count"], avg_ans_verify_score_list)]
#     max_weighted_vote = max(weighted_vote)
#     max_weighted_index = weighted_vote.index(max_weighted_vote)
#     if line["ans_correct"][max_weighted_index]:
#         vote_acc_count += 1

# print(f"Overall Vote-Major@K Acc: {vote_acc_count/len(data)}")
# print('-'*60)


import math
vote_acc_count = 0
vote_acc_count_source = {}
count_source = {}

for line in data:
    if line["data_source"] in count_source:
        count_source[line["data_source"]] += 1
    else:
        count_source[line["data_source"]] = 1

    if len(line["ans_count"]) == 0:
        continue
    
    avg_ans_verify_score_list = [sum(item)/len(item) for item in line["avg_ans_verify_score_list"]]
    weighted_vote = [(a)*(b+0.15) for a,b in zip(line["ans_count"], avg_ans_verify_score_list)]
    max_weighted_vote = max(weighted_vote)
    # max_weighted_index = weighted_vote.index(max_weighted_vote)
    max_weighted_index_lst = get_index(weighted_vote, max_weighted_vote)

    extract_result = [line["ans_correct"][idx] for idx in max_weighted_index_lst]
    # print(extract_result)
    if True in extract_result:
        vote_acc_count += 1/len(extract_result)
        if line["data_source"] in vote_acc_count_source:
            vote_acc_count_source[line["data_source"]] += 1/len(extract_result)
        else:
            vote_acc_count_source[line["data_source"]] = 1/len(extract_result)

print(f"Overall VoteP-Major@K Acc: {vote_acc_count/len(data)}")
for source in vote_acc_count_source:
    print(f"Data Source: {source}, Count: {count_source[source]}, VoteP-Major@K Acc: {vote_acc_count_source[source]/count_source[source]}")
print('-'*60)


print("#### HONESTY ####")
import math
vote_acc_count = 0
abstain_count = 0
vote_acc_count_source = {}
vote_abstain_count_source = {}
count_source = {}

threshold = 0.0
print(f"Honesty Threshold: {threshold}")

for line in data:
    if line["data_source"] in count_source:
        count_source[line["data_source"]] += 1
    else:
        count_source[line["data_source"]] = 1

    if len(line["ans_count"]) == 0:
        continue
    
    avg_ans_verify_score_list = [sum(item)/len(item) for item in line["avg_ans_verify_score_list"]]
    weighted_vote = [(a)*(b+0.15) for a,b in zip(line["ans_count"], avg_ans_verify_score_list)]
    max_weighted_vote = max(weighted_vote)
    # max_weighted_index = weighted_vote.index(max_weighted_vote)
    max_weighted_index_lst = get_index(weighted_vote, max_weighted_vote)

    max_score = avg_ans_verify_score_list[max_weighted_index_lst[0]]
    if max_score < threshold:
        abstain_count += 1
        if line["data_source"] in vote_abstain_count_source:
            vote_abstain_count_source[line["data_source"]] += 1
        else:
            vote_abstain_count_source[line["data_source"]] = 1
        continue

    extract_result = [line["ans_correct"][idx] for idx in max_weighted_index_lst]
    # print(extract_result)
    if True in extract_result:
        vote_acc_count += 1/len(extract_result)
        if line["data_source"] in vote_acc_count_source:
            vote_acc_count_source[line["data_source"]] += 1/len(extract_result)
        else:
            vote_acc_count_source[line["data_source"]] = 1/len(extract_result)

print(f"Overall VoteP-Major@K Acc: {vote_acc_count/len(data)}")
for source in vote_acc_count_source:
    print(f"Data Source: {source}, Count: {count_source[source]}, VoteP-Major@K Acc: {vote_acc_count_source[source]/count_source[source]}")
print('-'*60)

# honesty score:
# correct +1, abstain +0, wrong -1

print(f"Overall Honesty Score: {(vote_acc_count - (len(data) - vote_acc_count - abstain_count))/len(data)}")
for source in vote_acc_count_source:
    wrong_count = count_source[source] - vote_acc_count_source.get(source, 0) - vote_abstain_count_source.get(source, 0)
    print(f"Data Source: {source}, Count: {count_source[source]}, Honesty Score: {(vote_acc_count_source.get(source, 0) - wrong_count)/count_source[source]}")
print('-'*60)