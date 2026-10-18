import sys
import os
import matplotlib.pyplot as plt
import pandas as pd 

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from zebra_logic_analysis._uni_figure import load_data, search_space_size
import json 

size_ranges = {
        "Small": (0, 10**3),
        "Medium": (10**3, 10**6),
        "Large": (10**6, 10**10),
        "X-Large": (10**10, float("inf"))
}
    
def compute_puzzle_accuracy_by_size(bon_files, K, mode):
    global size_ranges
    accuracy_by_size = {key: {"correct": 0, "total": 0} for key in size_ranges}
    # we also need to return the following metrics: cell-wise accuracy, no_answer rates, reasoning lens
    cell_wise_accuracies = []
    total_cells_all = 0 
    correct_cells_all = 0
    no_answer_counts = 0
    reasoning_lens = []
    solved_counts = 0 
    for file_path in bon_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(f"Loaded {len(data)} entries from {file_path}")
            for entry in data:
                # Use the imported search_space_size function
                search_space_size_value = search_space_size(entry.get("size", "1*1"))
                solved = entry.get("solved", False)
                solved_counts += 1 if solved else 0
                parsed_bool = entry.get("parsed", True)
                total_cells, correct_cells = entry.get("total_cells"), entry.get("correct_cells", 0) 
                if not parsed_bool:
                    no_answer_counts += 1
                else:
                    # reasoning_lens.append(len(entry["output"][0]))
                    cell_wise_accuracy = (correct_cells / total_cells * 100) if total_cells > 0 else 0
                    cell_wise_accuracies.append(cell_wise_accuracy)
                total_cells_all += total_cells
                correct_cells_all += correct_cells
                
                

                for size_label, (min_size, max_size) in size_ranges.items():
                    if min_size <= search_space_size_value < max_size:
                        accuracy_by_size[size_label]["total"] += 1
                        if solved:
                            accuracy_by_size[size_label]["correct"] += 1
                        break
                

    for size_label, counts in accuracy_by_size.items():
        total = counts["total"]
        correct = counts["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"K={K}, {size_label} Search Space: {accuracy:.2f}% accuracy ({correct}/{total})")
    print("-"*100)
    # avg_cell_wise_accuracy = sum(cell_wise_accuracies) / len(cell_wise_accuracies) if cell_wise_accuracies else 0
    avg_cell_wise_accuracy = (correct_cells_all / total_cells_all * 100) if total_cells_all > 0 else 0
    no_answer_rate = no_answer_counts / len(data) * 100 if len(data) > 0 else 0
    overall_acc = solved_counts / len(data)
    # avg_reasoning_length = sum(reasoning_lens) / len(reasoning_lens) if reasoning_lens else 0
    return overall_acc, accuracy_by_size, avg_cell_wise_accuracy, no_answer_rate

# Load bon files
base_path = "result_dirs_parsed/zebra-grid/" 
models = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "o1-preview-2024-09-12", "o1-mini-2024-09-12-v3",  
          "Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.1-70B-Instruct", "Llama-3.1-405B-Instruct-Turbo",
        #   "Meta-Llama-3-8B-Instruct", "Meta-Llama-3-70B-Instruct",
          "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct",
          "gemma-2-27b-it@nvidia", "gemma-2-9b-it@nvidia", "gemma-2-2b-it", 
          "gemini-1.5-flash-exp-0827",  "gemini-1.5-pro-exp-0827", 
          "claude-3-5-sonnet-20241022", "Mistral-Large-2", "Mixtral-8x7B-Instruct-v0.1", 
          "Phi-3.5-mini-instruct", "deepseek-v2.5-0908", "Llama-3.2-3B-Instruct@together",
          "o1-2024-12-17", "deepseek-v3", "deepseek-r1"
        ]

models_llama = [
    # "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "o1-preview-2024-09-12", "o1-mini-2024-09-12-v3",  
          "Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.1-70B-Instruct", "Llama-3.1-405B-Instruct-Turbo", "Llama-3.2-3B-Instruct@together"
        #   "Meta-Llama-3-8B-Instruct", "Meta-Llama-3-70B-Instruct",
        #   "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct",
        #   "gemma-2-27b-it@nvidia", "gemma-2-9b-it@nvidia", "gemma-2-2b-it", 
        #   "gemini-1.5-flash-exp-0827",  "gemini-1.5-pro-exp-0827", 
        #   "claude-3-5-sonnet-20241022", "Mistral-Large-2", "Mixtral-8x7B-Instruct-v0.1", 
        #   "Phi-3.5-mini-instruct", "deepseek-v2.5-0908", 
        ]

# models = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", 
#             # "o1-preview-2024-09-12", "o1-mini-2024-09-12", 
#             "neg_feedback/gpt-4o-mini-2024-07-18.neg_feedback.T=1",
#             "neg_feedback/gpt-4o-2024-08-06.neg_feedback.T=1",
#             "zebra_oracle/gpt-4o-mini-2024-07-18.zebra_oracle.T=1",
#             "zebra_oracle/gpt-4o-2024-08-06.zebra_oracle.T=1",
#             "rm_32/gpt-4o-2024-08-06_rm_scores.rm_bon.K=32",
#             "rm_32/gpt-4o-mini-2024-07-18_rm_scores.rm_bon.K=32",
#             "bon_all/gpt-4o-2024-08-06.best_of_n.K=32",
#             "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=32",
#             "bon_all/gpt-4o-2024-08-06.most_common_of_n.K=32",
#             "bon_all/gpt-4o-mini-2024-07-18.most_common_of_n.K=32",
#             ]

# models = [ 
#             "bon_all/gpt-4o-2024-08-06.best_of_n.K=128",
#             "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=128",
#             "bon_all/gpt-4o-2024-08-06.most_common_of_n.K=128",
#             "bon_all/gpt-4o-mini-2024-07-18.most_common_of_n.K=128",
#             ]

# models = [
#     "neg_feedback_v2/gpt-4o-2024-08-06.neg_feedback_v2.T=1", # self-verify 
#     "neg_feedback_v2/gpt-4o-2024-08-06.neg_feedback_v2.T=2", # self-verify 
#     "neg_feedback_v2/gpt-4o-mini-2024-07-18.neg_feedback_v2.T=1",
# ]

    # "o1-preview-2024-09-12", "o1-mini-2024-09-12"]

# Add this mapping dictionary after the existing model definitions
model_name_mapping = {
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "o1-preview-2024-09-12-v2": "o1-preview",
    "o1-mini-2024-09-12-v3": "o1-mini",
    "bon_all/gpt-4o-2024-08-06.best_of_n.K=128": "gpt-4o BoN-Oracle (N=128)",
    "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=128": "gpt-4o-mini BoN-Oracle (N=128)",
    "bon_all/gpt-4o-2024-08-06.most_common_of_n.K=128": "gpt-4o Majority (N=128)",
    "bon_all/gpt-4o-mini-2024-07-18.most_common_of_n.K=128": "gpt-4o-mini Majority (N=128)",
    "Llama-3.2-3B-Instruct@together": "Llama-3.2-3B-Instruct",
    
}
# print the table of accuracies in the following shape 
# each row is for a particular model and mode
# then the columns are model names, small, medium, large, x-large, cell-wise, no_answer rates, reasoning lens.

# run the evaluation and then present the table 

# Initialize a DataFrame to store the results
results_df = pd.DataFrame(columns=[
    "Model", "All", "Small", "Medium", "Large", "X-Large", 
    "Cell Acc", 
    # "No Answer Rate",
    #  "Reasoning Length"
])

# Iterate over each model
for model in models:
    bon_files = [f"{base_path}/{model}.json"]
    overall_acc, accuracy_by_size, avg_cell_wise_accuracy, no_answer_rate = compute_puzzle_accuracy_by_size(bon_files, K=1, mode="default")
    
    # Map model name
    model_display_name = model_name_mapping.get(model, model)
    overall_acc = overall_acc * 100  # Convert to percentage
    # Extract accuracies for each size range
    small_accuracy = accuracy_by_size["Small"]["correct"] / accuracy_by_size["Small"]["total"] * 100 if accuracy_by_size["Small"]["total"] > 0 else 0
    medium_accuracy = accuracy_by_size["Medium"]["correct"] / accuracy_by_size["Medium"]["total"] * 100 if accuracy_by_size["Medium"]["total"] > 0 else 0
    large_accuracy = accuracy_by_size["Large"]["correct"] / accuracy_by_size["Large"]["total"] * 100 if accuracy_by_size["Large"]["total"] > 0 else 0
    xlarge_accuracy = accuracy_by_size["X-Large"]["correct"] / accuracy_by_size["X-Large"]["total"] * 100 if accuracy_by_size["X-Large"]["total"] > 0 else 0
    
    


    # Append the results to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([{
        "Model": model_display_name,
        "All": f"{overall_acc:.2f}%",
        "Small": f"{small_accuracy:.2f}%",
        "Medium": f"{medium_accuracy:.2f}%",
        "Large": f"{large_accuracy:.2f}%",
        "X-Large": f"{xlarge_accuracy:.2f}%",
        "Cell Acc": f"{avg_cell_wise_accuracy:.2f}%",
        # "No Answer Rate": f"{no_answer_rate:.2f}%",
        # "Reasoning Length": f"{avg_reasoning_length:.2f}"
    }])], ignore_index=True)
    # sort by the overall_acc in their float value 
    results_df["Overall Acc"] = results_df["All"].str.rstrip('%').astype(float)
    results_df = results_df.sort_values(by="Overall Acc", ascending=False).drop(columns=["Overall Acc"])

# Print the table
# print(results_df.to_string(index=False))
# use tabulate to print the table in a more readable format
from tabulate import tabulate


print(tabulate(results_df, headers='keys', tablefmt='pretty', showindex=False))

# use latex format to print the table
print(results_df.to_latex(index=False))
# 