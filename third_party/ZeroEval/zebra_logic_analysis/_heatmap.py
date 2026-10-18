import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def categorize_search_space(N, M):
    # Calculate the search space size: (N!)^M
    factorial_N = math.factorial(N)
    search_space_size = factorial_N ** M

    # Determine the category based on search space size
    if search_space_size < 10**3:
        category = "Small"
    elif 10**3 <= search_space_size < 10**6:
        category = "Medium"
    elif 10**6 <= search_space_size < 10**10:
        category = "Large"
    else:
        category = "X-Large"

    return search_space_size, category

# Create a grid with the actual search space sizes for 2x2 to 6x6
heatmap_data_search_space = np.zeros((5, 5), dtype=float)

for N in range(2, 7):
    for M in range(2, 7):
        search_space_size, _ = categorize_search_space(N, M)
        heatmap_data_search_space[N-2, M-2] = search_space_size  # Offset by 2 for 2x2 to 6x6

# Apply log10 to the search space size for better visualization
heatmap_data_log_search_space = np.log10(heatmap_data_search_space)
 
# Create the heatmap
plt.figure(figsize=(12, 10))  # Slightly larger figure
plt.rcParams.update({'font.size': 24})  # Increase base font size

# Create size labels
size_labels = ['Small', 'Medium', 'Large', 'X-Large']
annotations = []
for i in range(heatmap_data_log_search_space.shape[0]):
    row = []
    for j in range(heatmap_data_log_search_space.shape[1]):
        value = heatmap_data_log_search_space[i, j]
        _, category = categorize_search_space(i+2, j+2)  # Add 2 to get actual N,M values
        row.append(f"{value:.1f}\n{category}")
    annotations.append(row)

sns.heatmap(
    heatmap_data_log_search_space,
    annot=annotations,
    fmt='',
    cbar_kws={'label': 'Log10 of Search Space Size'},
    cmap='YlGnBu',
    linewidths=0.5,
    xticklabels=[f"{i}" for i in range(2, 7)],
    yticklabels=[f"{i}" for i in range(2, 7)],
    annot_kws={"size": 18}
)


# Set labels and title with larger fonts
plt.xlabel('Number of Attributes (M)', fontsize=22)
plt.ylabel('Number of Houses (N)', fontsize=22)
plt.title('Log-Scaled Grid Size Search Space Visualization (2x2 to 6x6)', fontsize=22, pad=15)

# Adjust layout to prevent label cutoff
plt.tight_layout()


# save the figure to zebra_logic_analysis/heatmap_size.png
plt.savefig("zebra_logic_analysis/heatmap_size.pdf", dpi=300)
