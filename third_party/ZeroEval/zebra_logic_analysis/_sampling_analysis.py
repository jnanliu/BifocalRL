import sys
import os
import matplotlib.pyplot as plt

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

    for file_path in bon_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for entry in data:
                # Use the imported search_space_size function
                search_space_size_value = search_space_size(entry.get("size", "1*1"))
                solved = entry.get("solved", False)
                
                for size_label, (min_size, max_size) in size_ranges.items():
                    if min_size <= search_space_size_value < max_size:
                        accuracy_by_size[size_label]["total"] += 1
                        if solved:
                            accuracy_by_size[size_label]["correct"] += 1
                        break
    
    print(f"Accuracy by size for K={K}, mode={mode}, bon_files={bon_files}:")
    for size_label, counts in accuracy_by_size.items():
        total = counts["total"]
        correct = counts["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"K={K}, {size_label} Search Space: {accuracy:.2f}% accuracy ({correct}/{total})")
    print("-"*100)

    return accuracy_by_size

# Load bon files
base_path = "result_dirs_parsed/zebra-grid/bon_all"
modes = ["best_of_n", "most_common_of_n", "rm_bon"]
models = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
# Update the K_values and modes definitions
K_values_by_mode = {
    "best_of_n": [1, 4, 8, 16, 32, 64, 128],
    "most_common_of_n": [1, 4, 8, 16, 32, 64, 128],
    "rm_bon": [1, 4, 8, 16, 32]
}
# Add this mapping dictionary after the existing model definitions
model_name_mapping = {
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "o1-preview-2024-09-12-v2": "o1-preview",
    "o1-mini-2024-09-12-v3": "o1-mini"
}

# Initialize a dictionary to store accuracies
accuracies = {size: {model_name_mapping[model]: {mode: [] for mode in modes} for model in models} for size in ["Small", "Medium", "Large", "X-Large"]}

for model in models:
    for mode in modes:
        for K in K_values_by_mode[mode]:
            if mode in ["best_of_n", "most_common_of_n"]:
                file_name = f"{model}.{mode}.K={K}.json"
                file_path = os.path.join(base_path, file_name)
                if os.path.exists(file_path):
                    accuracy_by_size = compute_puzzle_accuracy_by_size([file_path], K, mode)
                    for size_label in accuracies:
                        accuracies[size_label][model_name_mapping[model]][mode].append(accuracy_by_size[size_label]["correct"] / accuracy_by_size[size_label]["total"] * 100 if accuracy_by_size[size_label]["total"] > 0 else 0)
            else:
                file_name = f"{model}_rm_scores.{mode}.K={K}.json"
                file_path = os.path.join("result_dirs_parsed/zebra-grid/rm_32", file_name)
                if os.path.exists(file_path):
                    accuracy_by_size = compute_puzzle_accuracy_by_size([file_path], K, mode)
                    for size_label in accuracies:
                        accuracies[size_label][model_name_mapping[model]][mode].append(accuracy_by_size[size_label]["correct"] / accuracy_by_size[size_label]["total"] * 100 if accuracy_by_size[size_label]["total"] > 0 else 0)






# Add new models
additional_models = ["o1-preview-2024-09-12-v2", "o1-mini-2024-09-12-v3"]
additional_model_accuracies = {model_name_mapping[model]: {size: 0 for size in ["Small", "Medium", "Large", "X-Large"]} for model in additional_models}

# Load additional models' data
for model in additional_models:
    file_path = f"result_dirs_parsed/zebra-grid/{model}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            additional_model_accuracies[model_name_mapping[model]] = {size: {"correct": 0, "total": 0} for size in size_ranges}
            for entry in data:
                search_space_size_value = search_space_size(entry.get("size", "1*1"))
                solved = entry.get("solved", False)
                
                for size_label, (min_size, max_size) in size_ranges.items():
                    if min_size <= search_space_size_value < max_size:
                        additional_model_accuracies[model_name_mapping[model]][size_label]["total"] += 1
                        if solved:
                            additional_model_accuracies[model_name_mapping[model]][size_label]["correct"] += 1
                        break

# Calculate accuracies for additional models
for model in additional_models:
    for size_label, counts in additional_model_accuracies[model_name_mapping[model]].items():
        total = counts["total"]
        correct = counts["correct"]
        additional_model_accuracies[model_name_mapping[model]][size_label] = (correct / total * 100) if total > 0 else 0

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(26, 14), gridspec_kw={'hspace': 0.2, 'wspace': 0.13})

# Increase font size
plt.rcParams.update({'font.size': 12})  # Increase base font size

size_labels = ["Small", "Medium", "Large", "X-Large"]
colors = {
    "gpt-4o": "blue",
    "gpt-4o-mini": "#FFA500",  # More orange
    "o1-preview": "#9370DB",  # Brighter purple
    "o1-mini": "#A0522D"       # Lighter shade of brown
}
markers = {'gpt-4o': 'o', 'gpt-4o-mini': '<'}
linestyles = {"most_common_of_n": "-", "best_of_n": "dotted", "rm_bon": "dashdot"}

legend_handles = []
legend_labels = []

# Define custom y-axis limits for each subplot
y_limits = [(0, 100), (0, 100), (0, 100), (0, 40)]

 
for ax, size_label, ylim in zip(axes.flatten(), size_labels, y_limits):
    for model in models:
        mapped_model = model_name_mapping[model]
        for mode in modes:
            if len(accuracies[size_label][mapped_model][mode]) > 0:
                line, = ax.plot(K_values_by_mode[mode], 
                        accuracies[size_label][mapped_model][mode], 
                        color=colors[mapped_model], 
                        marker=markers[mapped_model],
                        linestyle=linestyles[mode],
                        linewidth=2
                )
            
            if ax == axes[0, 0]:
                legend_handles.append(line)
                mode_str = mode.replace("best_of_n", "BoN-Oracle")
                mode_str = mode_str.replace("rm_bon", "BoN-RM")
                mode_str = mode_str.replace("most_common_of_n", "Majority")
                legend_labels.append(f"{mapped_model} {mode_str}")


    for i, model in enumerate(additional_models):
        mapped_model = model_name_mapping[model]
        line = ax.axhline(y=additional_model_accuracies[mapped_model][size_label], 
                   color=colors[mapped_model], 
                   linestyle='--', 
                   linewidth=3)
        
        if ax == axes[0, 0]:
            legend_handles.append(line)
            legend_labels.append(f"{mapped_model} (ref)")
    
    ax.set_title(f"{size_label} Search Space", fontsize=25, fontweight='bold')
    
    
    if ax not in [axes[0, 0], axes[0, 1]]:  # Skip xlabel for the first two subplots
        ax.set_xlabel(r"$\mathbf{Number\ of\ Samples}$", fontsize=25)
    if ax not in [axes[0, 1], axes[1, 1]]:  # Skip ylabel for the second and fourth subplots
        ax.set_ylabel("Accuracy (%)", fontsize=25)
    ax.set_xscale('log', base=2)
    ax.set_xticks(K_values_by_mode["best_of_n"])
    ax.set_xticklabels(K_values_by_mode["best_of_n"])
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.grid(True)
    ax.set_ylim(ylim[0], ylim[1])  # Set custom y-axis limits for each subplot

    # Make the first and last y-axis labels bold
    yticks = ax.get_yticks()
    yticklabels = [f'${y:.0f}$' for y in yticks]
    yticklabels[0] = r'$\mathbf{' + yticklabels[0][1:-1] + '}$'
    yticklabels[-1] = r'$\mathbf{' + yticklabels[-1][1:-1] + '}$'
    ax.set_yticklabels(yticklabels)

# After the plotting loop, modify the legend order
new_order = [
    "o1-preview (ref)",  "gpt-4o BoN-Oracle", "gpt-4o BoN-RM",  "gpt-4o Majority",
    "o1-mini (ref)",   "gpt-4o-mini BoN-Oracle", "gpt-4o-mini BoN-RM", "gpt-4o-mini Majority",
]
# Reorder the legend_handles and legend_labels
reordered_handles = []
reordered_labels = []
for item in new_order:
    index = legend_labels.index(item)
    reordered_handles.append(legend_handles[index])
    reordered_labels.append(legend_labels[index])

# Replace the original lists with the reordered ones
legend_handles = reordered_handles
legend_labels = reordered_labels

# Update the legend creation for each subplot
axes[0, 0].legend(legend_handles, legend_labels, loc='lower right', ncol=2, fontsize=22.5, bbox_to_anchor=(1, 0))
# axes[0, 1].legend(legend_handles, legend_labels, loc='upper left', ncol=1, fontsize=18, bbox_to_anchor=(0, 1))
axes[1, 0].legend(legend_handles, legend_labels, loc='upper center', ncol=2, fontsize=22.5, bbox_to_anchor=(0.5, 1))
axes[1, 1].legend(legend_handles, legend_labels, loc='upper center', ncol=2, fontsize=22.5, bbox_to_anchor=(0.5, 1))

plt.tight_layout()

plt.savefig("zebra_logic_analysis/sampling.png", dpi=300, bbox_inches='tight')