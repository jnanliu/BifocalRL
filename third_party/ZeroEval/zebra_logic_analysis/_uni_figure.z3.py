import json

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

z3_stat = {}
with open("zebra_logic_analysis/z3_data_dict.json") as f:
    z3_stat = json.load(f) # key is th id and the value is a dict of z3 data (like conflicts, time, etc)


def search_space_size(size):
    N, M = map(int, size.split("*"))
    return (math.factorial(N))**M


def clean_model_name(model_name):
    if "bon_all/gpt-4o-mini-2024-07-18" in model_name:
        # Extract K value and return simplified name
        k_value = model_name.split("K=")[1]
        return f"gpt-4o-mini (pass@{k_value})"
    elif "bon_all/gpt-4o-2024-08-06" in model_name:
        # Extract K value and return simplified name
        k_value = model_name.split("K=")[1]
        return f"gpt-4o (pass@{k_value})"
    suffix = ""
    model_name = model_name.replace("o1-mini-2024-09-12-v3", f"o1-mini{suffix}")
    model_name = model_name.replace("o1-preview-2024-09-12-v2", f"o1-preview{suffix}")
    model_name = model_name.replace("o1-2024-12-17", f"o1{suffix}")
    model_name = model_name.replace("deepseek-r1", f"deepseek-r1{suffix}")
    model_name = model_name.replace("Meta-Llama", "Llama")
    model_name = model_name.replace("@together", "")
    model_name = model_name.replace("-Turbo", "")

    model_name = model_name.replace("-2024-08-06", "")
    model_name = model_name.replace("-2024-07-18", "")
    return model_name
    
def load_data(model_list, base_path):
    data_by_model = {}
    for model in model_list:
        with open(f"{base_path}/{model}.json") as f:
            data = json.load(f)
        data_by_model[model] = data
    return data_by_model

def plot_hidden_reasoning_vs_z3(data_by_model, model_list, output_file_name, z3_key="conflicts"):
    plt.figure(figsize=(10, 6))
    
    for model in model_list:
        data = data_by_model[model]
        hidden_reasoning_token = [d["hidden_reasoning_token"] for d in data]
        size = [d["size"] for d in data]
        search_space_sizes = [search_space_size(s) for s in size]
        solved_status = [d["solved"] for d in data]
        z3_conflicts = [z3_stat[d["id"]]["conflicts"] for d in data]

        df = pd.DataFrame({
            'hidden_reasoning_token': hidden_reasoning_token,
            'search_space_size': search_space_sizes,
            'z3_conflicts': z3_conflicts,
            'solved': solved_status
        })
        # Bin the z3_conflicts and calculate mean hidden reasoning tokens for each bin
        df["z3_conflicts_bin"] = pd.qcut(df["z3_conflicts"], q=15, duplicates='drop')
        tokens_by_conflicts = df.groupby("z3_conflicts_bin")["hidden_reasoning_token"].mean()
        bin_centers = df.groupby("z3_conflicts_bin")["z3_conflicts"].mean()

        # Plot average hidden reasoning tokens vs z3_conflicts
        plt.plot(bin_centers, tokens_by_conflicts, marker='o', linewidth=2, label=clean_model_name(model))

    plt.xlabel("Z3 Conflicts")
    plt.ylabel("Average Hidden CoT Tokens")
    plt.title("Hidden CoT Tokens vs. Z3 Conflicts")
    plt.grid(True)
    plt.legend(title="Model")

    plt.savefig(output_file_name, dpi=300)
    print(f"Saved the plot to {output_file_name}")


def plot_hidden_reasoning_vs_z3_v2(data, output_file_name, z3_key="conflicts", max_space_size=80):
    hidden_reasoning_token = [d["hidden_reasoning_token"] for d in data]
    size = [d["size"] for d in data]
    search_space_sizes = [search_space_size(s) for s in size]
    solved_status = [d["solved"] for d in data]
    z3_conflicts = [z3_stat[d["id"]]["conflicts"] for d in data]

    df = pd.DataFrame({
        'hidden_reasoning_token': hidden_reasoning_token,
        'search_space_size': search_space_sizes,
        'z3_conflicts': z3_conflicts,
        'solved': solved_status
    })

    plt.figure(figsize=(9, 6))
    
    # First plot the individual points
    sns.scatterplot(data=df, x='z3_conflicts', y='hidden_reasoning_token', 
                    hue='solved', style='solved', 
                    palette={True: '#4a90e2', False: '#d9534f'},
                    markers={True: 'o', False: 'X'}, 
                    hue_order=[True, False], 
                    legend='full', 
                    alpha=0.5,  # Make points semi-transparent
                    s=100)  # Point size
    
    # Then plot the average line
    df["z3_conflicts_bin"] = pd.qcut(df["z3_conflicts"], q=15, duplicates='drop')
    tokens_by_conflicts = df.groupby("z3_conflicts_bin")["hidden_reasoning_token"].mean()
    bin_centers = df.groupby("z3_conflicts_bin")["z3_conflicts"].mean()
    plt.plot(bin_centers, tokens_by_conflicts, 
             color='green', linewidth=2, label='Average',
             zorder=5)  # Put line on top of points
    # set x-axis limit to 80 
    plt.xlim(0, max_space_size)
    if "o1.hidden" in output_file_name:
        plt.ylim(0, 20000)
    else:
        plt.ylim(0, 15000)
    plt.xlabel("# Z3 Conflicts")
    plt.ylabel("Hidden CoT Tokens")
    # plt.title(f"Hidden CoT Tokens vs. Z3 Conflicts")
    plt.grid(True)
    
    # Update legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=['Correct', 'Incorrect', 'Average'],
              title='Solution Status')

    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    print(f"Saved the plot to {output_file_name}")


# def plot_hidden_reasoning_vs_search_space(data, output_file_name):
#     # visible_reasoning_token = [d["visible_reasoning_token"] for d in data]
#     # define visible reasoning token as the sum of the number of tokens in the output 
    

#     size = [d["size"] for d in data]
#     search_space_sizes = [search_space_size(s) for s in size]
#     solved_status = [d["solved"] for d in data]

#     df = pd.DataFrame({
#         'visible_reasoning_token': visible_reasoning_token,
#         'search_space_size': search_space_sizes,
#         'solved': solved_status
#     })

#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(data=df, x='search_space_size', y='visible_reasoning_token', 
#                     hue='solved', style='solved', 
#                     palette={True: '#4a90e2', False: '#d9534f'},
#                     markers={True: 'o', False: 'X'}, hue_order=[True, False], legend='full', s=100)
#     plt.legend(title='Solution Status', labels=['Incorrect', 'Correct'])
#     plt.xscale('log')
#     plt.xlabel('Search Space Size (log scale)')
#     plt.ylabel('# Visible Reasoning Tokens')
#     plt.ylim(0, 20000)
#     plt.grid(True)

#     log_x = np.log10(df['search_space_size'])
#     coeffs = np.polyfit(log_x, df['visible_reasoning_token'], deg=2)
#     poly = np.poly1d(coeffs)
#     x_fit = np.linspace(log_x.min(), log_x.max(), 500)
#     y_fit = poly(x_fit)
#     plt.plot(10**x_fit, y_fit, color='green', label='Polynomial Fit (degree 2)', linewidth=2)

#     plt.savefig(output_file_name, dpi=300)
#     print(f"Saved the plot to {output_file_name}")

def plot_accuracy_vs_z3(data_by_model, model_list, output_file_name, max_space_size, z3_key="conflicts"):
    plt.figure(figsize=(6, 6))
    for model in model_list:
        model_data = data_by_model[model]
        df = pd.DataFrame(model_data)
        df["z3_conflicts"] = df["id"].apply(lambda x: z3_stat[x]["conflicts"])
        # Bin the z3_conflicts and calculate accuracy for each bin
        df["z3_conflicts_bin"] = pd.qcut(df["z3_conflicts"], q=10, duplicates='drop')
        # also for the z3_propagations
        df["z3_propagations"] = df["id"].apply(lambda x: z3_stat[x]["propagations"])
        df["z3_propagations_bin"] = pd.qcut(df["z3_propagations"], q=10, duplicates='drop')
        if z3_key == "conflicts":
            accuracy_by_conflicts = df.groupby("z3_conflicts_bin")["solved"].mean() * 100
            bin_centers = df.groupby("z3_conflicts_bin")["z3_conflicts"].mean()
        elif z3_key == "propagations":
            accuracy_by_conflicts = df.groupby("z3_propagations_bin")["solved"].mean() * 100
            bin_centers = df.groupby("z3_propagations_bin")["z3_propagations"].mean()
        
        # Plot accuracy vs z3_conflicts
        plt.plot(bin_centers, accuracy_by_conflicts, marker='o', label=clean_model_name(model))

    # set the x-axis limit with max_space_size
    plt.xlim(0, max_space_size)
    plt.ylim(None, 102)

    # plt.xscale("log")  # Use log scale for conflicts
    if z3_key == "conflicts":
        # plt.xlim(1, 10**max_space_size)
        plt.xlabel("Z3 Conflicts")
        # plt.title("Accuracy vs. Z3 Conflicts")
    elif z3_key == "propagations":
        plt.xlabel("Z3 Propagations")
        # plt.title("Accuracy vs. Z3 Propagations")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend(title="Model")
    output_file_name = output_file_name.replace(".z3.", f".z3_{z3_key}.")
    plt.savefig(output_file_name)
    print(f"Saved the plot to {output_file_name}")

def plot_reasoning_length_vs_search_space(data_by_model, model_list, output_file_name, max_space_size):
    plt.figure(figsize=(20, 5))
    for model in model_list:
        model_data = data_by_model[model]
        df = pd.DataFrame(model_data)
        df["search_space_size"] = df["size"].apply(search_space_size)
        reasoning_length_data = df.groupby("search_space_size", as_index=False).apply(
            lambda group: pd.Series({
                # "average_reasoning_length": group["output"].apply(lambda x: len(x[0])).mean()
                # "average_reasoning_length": group["output"].apply(lambda x: len(x[0].split('solution\":')[0])).mean()
                "average_reasoning_length": group["output"].apply(lambda x: len(x[0].split('solution\":')[0])).mean()/4
                # "average_reasoning_length": group["reasoning"].apply(lambda x: len(x)).mean()
            })
        ).reset_index(drop=True)
        sns.lineplot(data=reasoning_length_data, x="search_space_size", y="average_reasoning_length", marker="o", label=model)

    plt.xscale("log")
    plt.xlim(1, 10**max_space_size)
    plt.xlabel("Search Space Size (log scale)")
    plt.ylabel("# Visible Reasoning Tokens")
    plt.title("# Visible Reasoning Tokens vs. Search Space Size for Multiple Models")
    plt.grid(True)
    plt.legend(title="Model")
    plt.savefig(output_file_name)
    print(f"Saved the plot to {output_file_name}")

def main():
    parser = argparse.ArgumentParser(description="Unified analysis script for zebra logic puzzles.")
    parser.add_argument('--analysis', type=str, choices=['hidden_reasoning', 'accuracy', 'reasoning_length', "hidden_reasoning_v1"], required=True, help="Type of analysis to perform.")
    parser.add_argument('--model_list', nargs='+', required=True, help="List of models to analyze.")
    parser.add_argument('--output_file', type=str, required=True, help="Output file name for the plot.")
    parser.add_argument('--max_space_size', type=float, default=17.5, help="Maximum search space size for x-axis limit.")
    parser.add_argument('--base_path', type=str, default="result_dirs_parsed/zebra-grid", help="Base path for model data files.")
    args = parser.parse_args()

    data_by_model = load_data(args.model_list, args.base_path)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 16
    })

    if args.analysis == 'hidden_reasoning':
        plot_hidden_reasoning_vs_z3_v2(data_by_model[args.model_list[0]], args.output_file, args.max_space_size)
    elif args.analysis == 'hidden_reasoning_v1':
        plot_hidden_reasoning_vs_z3(data_by_model, args.model_list, args.output_file, args.max_space_size)
    elif args.analysis == 'accuracy':
        plot_accuracy_vs_z3(data_by_model, args.model_list, args.output_file, args.max_space_size)
    elif args.analysis == 'reasoning_length':
        plot_reasoning_length_vs_search_space(data_by_model, args.model_list, args.output_file, args.max_space_size)

if __name__ == "__main__":
    main()


"""

# For showing hidden reasoning token vs search space size

python zebra_logic_analysis/_uni_figure.z3.py --analysis hidden_reasoning_v1 --model_list o1-2024-12-17 --output_file zebra_logic_analysis/o1.hidden_cot.z3_v1.png

python zebra_logic_analysis/_uni_figure.z3.py --analysis hidden_reasoning_v1 --model_list o1-2024-12-17 o1-preview-2024-09-12-v2 o1-mini-2024-09-12-v3 --output_file zebra_logic_analysis/o1.more.hidden_cot.z3_v1.png

python zebra_logic_analysis/_uni_figure.z3.py --analysis hidden_reasoning --model_list o1-2024-12-17 --output_file zebra_logic_analysis/o1.hidden_cot.z3_v2.png
python zebra_logic_analysis/_uni_figure.z3.py --analysis hidden_reasoning --model_list o1-preview-2024-09-12-v2 --output_file zebra_logic_analysis/o1_preview.hidden_cot.z3_v2.pdf
python zebra_logic_analysis/_uni_figure.z3.py --analysis hidden_reasoning --model_list o1-mini-2024-09-12-v3 --output_file zebra_logic_analysis/o1_mini.hidden_cot.z3_v2.png



# For showing accuracy histograms
python zebra_logic_analysis/_uni_figure.z3.py --analysis accuracy \
    --model_list o1-preview-2024-09-12-v2 o1-mini-2024-09-12-v3 gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18 \
    --output_file zebra_logic_analysis/openai.accuracy_hists.z3.png --max_space_size 80

    
python zebra_logic_analysis/_uni_figure.z3.py --analysis accuracy \
    --model_list o1-2024-12-17 deepseek-r1 o1-preview-2024-09-12-v2 o1-mini-2024-09-12-v3 gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18 \
    --output_file zebra_logic_analysis/openai.new.accuracy_hists.z3.png --max_space_size 82

python zebra_logic_analysis/_uni_figure.z3.py --analysis accuracy \
    --model_list Llama-3.2-3B-Instruct@together Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-70B-Instruct Llama-3.1-405B-Instruct-Turbo \
    --output_file zebra_logic_analysis/llama.accuracy_hists.z3.png --max_space_size 40

python zebra_logic_analysis/_uni_figure.z3.py --analysis accuracy \
    --model_list "Qwen2.5-3B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-72B-Instruct" \
    --output_file zebra_logic_analysis/qwen.accuracy_hists.z3.png --max_space_size 40


# For showing accuracy histograms 
python zebra_logic_analysis/_uni_figure.z3.py --analysis accuracy \
    --model_list "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=1" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=4" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=8" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=16" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=32" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=64" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=128" "o1-mini-2024-09-12-v3"  \
    --output_file zebra_logic_analysis/bon_4o_mini.accuracy_hists.z3.png --max_space_size 80



# For showing accuracy histograms
python zebra_logic_analysis/_uni_figure.z3.py --analysis accuracy \
    --model_list "bon_all/gpt-4o-2024-08-06.best_of_n.K=1" "bon_all/gpt-4o-2024-08-06.best_of_n.K=4" "bon_all/gpt-4o-2024-08-06.best_of_n.K=8" "bon_all/gpt-4o-2024-08-06.best_of_n.K=16" "bon_all/gpt-4o-2024-08-06.best_of_n.K=32" "bon_all/gpt-4o-2024-08-06.best_of_n.K=64" "bon_all/gpt-4o-2024-08-06.best_of_n.K=128" "o1-preview-2024-09-12-v2"  \
    --output_file zebra_logic_analysis/bon_4o.accuracy_hists.z3.png --max_space_size 18


python zebra_logic_analysis/_uni_figure.py --analysis accuracy \
    --model_list Llama-3.2-3B-Instruct@together Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-70B-Instruct Llama-3.1-405B-Instruct-Turbo gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18 \
    --output_file zebra_logic_analysis/llama_gpt.accuracy_hists.png --max_space_size 18


python zebra_logic_analysis/_uni_figure.py --analysis accuracy \
    --model_list "Qwen2.5-3B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-72B-Instruct" \
    --output_file zebra_logic_analysis/qwen.accuracy_hists.png --max_space_size 10


python zebra_logic_analysis/_uni_figure.py --analysis accuracy \
    --model_list "Qwen2.5-3B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-72B-Instruct" \
    --output_file zebra_logic_analysis/gpt.accuracy_hists.png --max_space_size 10




python zebra_logic_analysis/_uni_figure.py --analysis accuracy \
    --model_list Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct \
    --output_file zebra_logic_analysis/qwen.accuracy_hists.png --max_space_size 7 

# For showing explicit reasoning length

python zebra_logic_analysis/_uni_figure.py --analysis reasoning_length \
    --model_list o1-preview-2024-09-12-v2 o1-mini-2024-09-12-v3 gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18 \
    --output_file zebra_logic_analysis/openai.reason_len.png --max_space_size 18

python zebra_logic_analysis/_uni_figure.py --analysis reasoning_length \
    --model_list Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-70B-Instruct Llama-3.1-405B-Instruct-Turbo \
    --output_file zebra_logic_analysis/llama.reason_len.png --max_space_size 18

python zebra_logic_analysis/_uni_figure.py --analysis reasoning_length \
    --model_list Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct \
    --output_file zebra_logic_analysis/qwen.reason_len.png --max_space_size 18  

"""