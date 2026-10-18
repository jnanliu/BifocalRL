import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

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
    model_name = model_name.replace("o1-mini-2024-09-12-v3", "o1-mini (pass@1)")
    model_name = model_name.replace("o1-preview-2024-09-12-v2", "o1-preview (pass@1)")
    model_name = model_name.replace("Meta-Llama", "Llama")
    model_name = model_name.replace("@together", "")
    model_name = model_name.replace("-Turbo", "")
    return model_name
    
def load_data(model_list, base_path):
    data_by_model = {}
    for model in model_list:
        with open(f"{base_path}/{model}.json") as f:
            data = json.load(f)
        data_by_model[model] = data
    return data_by_model

def plot_hidden_reasoning_vs_search_space(data, output_file_name):
    hidden_reasoning_token = [d["hidden_reasoning_token"] for d in data]
    size = [d["size"] for d in data]
    search_space_sizes = [search_space_size(s) for s in size]
    solved_status = [d["solved"] for d in data]

    df = pd.DataFrame({
        'hidden_reasoning_token': hidden_reasoning_token,
        'search_space_size': search_space_sizes,
        'solved': solved_status
    })

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='search_space_size', y='hidden_reasoning_token', 
                    hue='solved', style='solved', 
                    palette={True: '#4a90e2', False: '#d9534f'},
                    markers={True: 'o', False: 'X'}, hue_order=[True, False], legend='full', s=100)
    plt.legend(title='Solution Status', labels=['Incorrect', 'Correct'])
    plt.xscale('log')
    plt.xlabel('Search Space Size (log scale)')
    plt.ylabel('# Hidden CoT Tokens')
    plt.ylim(0, 20000)
    plt.grid(True)

    log_x = np.log10(df['search_space_size'])
    coeffs = np.polyfit(log_x, df['hidden_reasoning_token'], deg=2)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(log_x.min(), log_x.max(), 500)
    y_fit = poly(x_fit)
    plt.plot(10**x_fit, y_fit, color='green', label='Polynomial Fit (degree 2)', linewidth=2)

    # plt.savefig(output_file_name, dpi=300)
    # print(f"Saved the plot to {output_file_name}")
    
    # print the total and avg number of hidden reasoning tokens
    total_tokens = df['hidden_reasoning_token'].sum()
    avg_tokens = df['hidden_reasoning_token'].mean()
    print(f"Total hidden reasoning tokens: {total_tokens:,.0f}")
    print(f"Average hidden reasoning tokens per puzzle: {avg_tokens:,.1f}")

    




def plot_hidden_reasoning_vs_search_space_v2(data, output_file_name):
    # visible_reasoning_token = [d["visible_reasoning_token"] for d in data]
    # define visible reasoning token as the sum of the number of tokens in the output 

    size = [d["size"] for d in data]
    search_space_sizes = [search_space_size(s) for s in size]
    solved_status = [d["solved"] for d in data]

    df = pd.DataFrame({
        'visible_reasoning_token': visible_reasoning_token,
        'search_space_size': search_space_sizes,
        'solved': solved_status
    })

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='search_space_size', y='visible_reasoning_token', 
                    hue='solved', style='solved', 
                    palette={True: '#4a90e2', False: '#d9534f'},
                    markers={True: 'o', False: 'X'}, hue_order=[True, False], legend='full', s=100)
    plt.legend(title='Solution Status', labels=['Incorrect', 'Correct'])
    plt.xscale('log')
    plt.xlabel('Search Space Size (log scale)')
    plt.ylabel('# Visible Reasoning Tokens')
    plt.ylim(0, 20000)
    plt.grid(True)

    log_x = np.log10(df['search_space_size'])
    coeffs = np.polyfit(log_x, df['visible_reasoning_token'], deg=2)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(log_x.min(), log_x.max(), 500)
    y_fit = poly(x_fit)
    plt.plot(10**x_fit, y_fit, color='green', label='Polynomial Fit (degree 2)', linewidth=2)

    # print the total and avg number of hidden reasoning tokens
    total_tokens = df['visible_reasoning_token'].sum()
    avg_tokens = df['visible_reasoning_token'].mean()
    print(f"Total visible reasoning tokens: {total_tokens:,.0f}")
    print(f"Average visible reasoning tokens per puzzle: {avg_tokens:,.1f}")

    # plt.savefig(output_file_name, dpi=300)
    # print(f"Saved the plot to {output_file_name}")

def plot_accuracy_vs_search_space_v1(data_by_model, model_list, output_file_name, max_space_size):
    plt.figure(figsize=(10, 6))
    for model in model_list:
        model_data = data_by_model[model]
        df = pd.DataFrame(model_data)
        df["search_space_size"] = df["size"].apply(search_space_size)
        accuracy_data = df.groupby("search_space_size", as_index=False).apply(
            lambda group: pd.Series({
                "accuracy": group["solved"].sum() / 40 * 100
            })
        ).reset_index(drop=True)
        clean_name = clean_model_name(model)
        sns.lineplot(data=accuracy_data, x="search_space_size", y="accuracy", marker="o", label=clean_name)

    plt.xscale("log")
    plt.xlim(1, 10**max_space_size)
    plt.xlabel("Search Space Size (log scale)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Search Space Size for Multiple Models")
    plt.grid(True)
    plt.legend(title="Model")
    plt.savefig(output_file_name)
    print(f"Saved the plot to {output_file_name}")

def plot_accuracy_vs_search_space(data_by_model, model_list, output_file_name, max_space_size):
    plt.figure(figsize=(10, 6))
    
    # Define bins in log space
    bin_edges = np.logspace(0, max_space_size, num=15)  # 20 bins from 10^0 to 10^max_space_size
    
    for model in model_list:
        model_data = data_by_model[model]
        df = pd.DataFrame(model_data)
        df["search_space_size"] = df["size"].apply(search_space_size)
        
        # Bin the data
        df['space_size_bin'] = pd.cut(df['search_space_size'], bins=bin_edges, labels=bin_edges[:-1])
        
        # Calculate accuracy for each bin - modified to handle empty groups
        accuracy_data = []
        for name, group in df.groupby("space_size_bin"):
            if len(group) > 0:  # Only process non-empty groups
                accuracy_data.append({
                    "search_space_size": name,
                    "accuracy": group["solved"].sum() / len(group) * 100
                })
        
        accuracy_data = pd.DataFrame(accuracy_data)
        if not accuracy_data.empty:  # Only plot if we have data
            clean_name = clean_model_name(model)
            sns.lineplot(data=accuracy_data, x="search_space_size", y="accuracy", 
                        marker="o", label=clean_name)

    plt.xscale("log")
    plt.xlim(1, 10**max_space_size)
    plt.xlabel("Search Space Size (log scale)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Search Space Size")
    plt.grid(True)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_file_name)
    print(f"Saved the plot to {output_file_name}")

# def plot_reasoning_length_vs_search_space(data_by_model, model_list, output_file_name, max_space_size):
#     plt.figure(figsize=(20, 5))
#     for model in model_list:
#         model_data = data_by_model[model]
#         df = pd.DataFrame(model_data)
#         df["search_space_size"] = df["size"].apply(search_space_size)
#         reasoning_length_data = df.groupby("search_space_size", as_index=False).apply(
#             lambda group: pd.Series({
#                 # "average_reasoning_length": group["output"].apply(lambda x: len(x[0])).mean()
#                 # "average_reasoning_length": group["output"].apply(lambda x: len(x[0].split('solution\":')[0])).mean()
#                 "average_reasoning_length": group["output"].apply(lambda x: len(x[0].split('solution\":')[0])).mean()/4
#                 # "average_reasoning_length": group["reasoning"].apply(lambda x: len(x)).mean()
#             })
#         ).reset_index(drop=True)
#         sns.lineplot(data=reasoning_length_data, x="search_space_size", y="average_reasoning_length", marker="o", label=model)

#     plt.xscale("log")
#     plt.xlim(1, 10**max_space_size)
#     plt.xlabel("Search Space Size (log scale)")
#     plt.ylabel("# Visible Reasoning Tokens")
#     plt.title("# Visible Reasoning Tokens vs. Search Space Size for Multiple Models")
#     plt.grid(True)
#     plt.legend(title="Model")
#     plt.savefig(output_file_name)
#     print(f"Saved the plot to {output_file_name}")

def plot_reasoning_length_vs_search_space(data_by_model, model_list, output_file_name, max_space_size):
    plt.figure(figsize=(20, 5))
    
    # Define bins in log space
    bin_edges = np.logspace(0, max_space_size, num=20)  # 15 bins from 10^0 to 10^max_space_size
    
    for model in model_list:
        model_data = data_by_model[model]
        df = pd.DataFrame(model_data)
        df["search_space_size"] = df["size"].apply(search_space_size)
        
        # Bin the data
        df['space_size_bin'] = pd.cut(df['search_space_size'], bins=bin_edges, labels=bin_edges[:-1])
        
        # Calculate average reasoning length for each bin
        reasoning_length_data = []
        for name, group in df.groupby("space_size_bin"):
            if len(group) > 0:  # Only process non-empty groups
                reasoning_length_data.append({
                    "search_space_size": name,
                    "average_reasoning_length": group["output"].apply(
                        lambda x: len(x[0].split('solution\":')[0])).mean()/4
                })
        
        reasoning_length_data = pd.DataFrame(reasoning_length_data)
        if not reasoning_length_data.empty:  # Only plot if we have data
            clean_name = clean_model_name(model)
            sns.lineplot(data=reasoning_length_data, x="search_space_size", 
                        y="average_reasoning_length", marker="o", label=clean_name)
        
    # Print total reasoning tokens for each model
    for model in model_list:
        model_data = data_by_model[model]
        df = pd.DataFrame(model_data)
        total_tokens = df["output"].apply(lambda x: len(x[0].split('solution\":')[0])).sum()/4
        clean_name = clean_model_name(model)
        print(f"Total reasoning tokens for {clean_name}: {total_tokens:.0f} (average: {total_tokens/len(df):.2f})")

    plt.xscale("log")
    plt.xlim(1, 10**max_space_size)
    plt.xlabel("Search Space Size (log scale)")
    plt.ylabel("# Visible Reasoning Tokens")
    plt.title("# Visible Reasoning Tokens vs. Search Space Size")
    plt.grid(True)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_file_name)
    print(f"Saved the plot to {output_file_name}")

def main():
    parser = argparse.ArgumentParser(description="Unified analysis script for zebra logic puzzles.")
    parser.add_argument('--analysis', type=str, choices=['hidden_reasoning', 'accuracy', 'reasoning_length'], required=True, help="Type of analysis to perform.")
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
        plot_hidden_reasoning_vs_search_space(data_by_model[args.model_list[0]], args.output_file)
    elif args.analysis == 'accuracy':
        plot_accuracy_vs_search_space(data_by_model, args.model_list, args.output_file, args.max_space_size)
    elif args.analysis == 'reasoning_length':
        plot_reasoning_length_vs_search_space(data_by_model, args.model_list, args.output_file, args.max_space_size)

if __name__ == "__main__":
    main()


"""

# For showing hidden reasoning token vs search space size
python zebra_logic_analysis/_uni_figure.py --analysis hidden_reasoning --model_list o1-preview-2024-09-12-v2 --output_file zebra_logic_analysis/o1_preview.hidden_cot.png
python zebra_logic_analysis/_uni_figure.py --analysis hidden_reasoning --model_list o1-mini-2024-09-12-v3 --output_file zebra_logic_analysis/o1_mini.hidden_cot.png



# For showing accuracy histograms
python zebra_logic_analysis/_uni_figure.py --analysis accuracy \
    --model_list o1-preview-2024-09-12-v2 o1-mini-2024-09-12-v3 gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18 \
    --output_file zebra_logic_analysis/openai.accuracy_hists.png --max_space_size 18


python zebra_logic_analysis/_uni_figure.py --analysis accuracy \
    --model_list Llama-3.2-3B-Instruct@together Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-70B-Instruct Llama-3.1-405B-Instruct-Turbo \
    --output_file zebra_logic_analysis/llama.accuracy_hists.png --max_space_size 18


# For showing accuracy histograms 
python zebra_logic_analysis/_uni_figure.py --analysis accuracy \
    --model_list "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=1" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=4" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=8" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=16" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=32" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=64" "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=128" "o1-mini-2024-09-12-v3"  \
    --output_file zebra_logic_analysis/bon_4o_mini.accuracy_hists.png --max_space_size 18
# "bon_all/gpt-4o-mini-2024-07-18.best_of_n.K=256" \


# For showing accuracy histograms
python zebra_logic_analysis/_uni_figure.py --analysis accuracy \
    --model_list "bon_all/gpt-4o-2024-08-06.best_of_n.K=1" "bon_all/gpt-4o-2024-08-06.best_of_n.K=4" "bon_all/gpt-4o-2024-08-06.best_of_n.K=8" "bon_all/gpt-4o-2024-08-06.best_of_n.K=16" "bon_all/gpt-4o-2024-08-06.best_of_n.K=32" "bon_all/gpt-4o-2024-08-06.best_of_n.K=64" "bon_all/gpt-4o-2024-08-06.best_of_n.K=128" "o1-preview-2024-09-12-v2"  \
    --output_file zebra_logic_analysis/bon_4o.accuracy_hists.png --max_space_size 18


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