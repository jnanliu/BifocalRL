import json
import math 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# load the file from "result_dirs_parsed/zebra-grid/o1-mini-2024-09-12-v3.json"

data_by_model = {}

model_list = ["o1-preview-2024-09-12-v2", "o1-mini-2024-09-12-v3", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"] 
output_file_name = "zebra_logic_analysis/openai_reason_lens.png"
max_space_size = 17.5 

# model_list = ["Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.1-70B-Instruct", "Llama-3.1-405B-Instruct-Turbo"] 
# output_file_name = "zebra_logic_analysis/llama_reason_lens.png"
# max_space_size = 17.5 

# model_list = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct"]
# output_file_name = "zebra_logic_analysis/qwen_reason_lens.png"
# max_space_size = 17.5


for model in model_list:
  with open(f"result_dirs_parsed/zebra-grid/{model}.json") as f:
    data = json.load(f)
  data_by_model[model] = data



 
# search space size can be calculated by the `size` key by using the formula: (N!)^M where N*M is from the size key, write a function to get arg like "3*4" to return the search space size
def search_space_size(size):
    N, M = map(int, size.split("*"))
    # factorial of N and times M 
    return (math.factorial(N))**M 


# draw a figure to show the reasoning len (y) vs the search space size (x)
# the y-axis is the reasoning len in characters for the group of the size interval
# the x-axis is the search space size
# Output Length is the number of solved puzzles divided by the total number of puzzles in the group

df = pd.DataFrame(data) 
# Plot the data
plt.figure(figsize=(10, 6))

for model in model_list:
    model_data = data_by_model[model]
    df = pd.DataFrame(model_data)
    df["search_space_size"] = df["size"].apply(search_space_size)
    
    # Calculate the average reasoning length for each search space size
    reasoning_length_data = df.groupby("search_space_size", as_index=False).apply(
        lambda group: pd.Series({
            # "average_reasoning_length": group["reasoning"].apply(len).mean()
            "average_reasoning_length": group["output"].apply(lambda x: len(x[0])).mean()
        })
    ).reset_index(drop=True)
    
    # Plot the data for the current model
    sns.lineplot(data=reasoning_length_data, x="search_space_size", y="average_reasoning_length", marker="o", label=model)


# Set the x-axis to a logarithmic scale
plt.xscale("log")
# set the x-axis with the maximum value of the search space size
plt.xlim(1, 10**max_space_size)

# Add labels and title
plt.xlabel("Search Space Size (log scale)")
plt.ylabel("Output Length (in characters)")
plt.title("Output Length vs. Search Space Size for Multiple Models")
plt.grid(True)

# Add a legend to distinguish between models
plt.legend(title="Model")
plt.savefig(output_file_name)
# plt.show()

 