import json 
import matplotlib.pyplot as plt
import numpy as np
import math 
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

z3_stat = {}
with open("zebra_logic_analysis/z3_data_dict.json") as f:
    z3_stat = json.load(f) # key is th id and the value is a dict of z3 data (like conflicts, time, etc)

z3_conflicts = []
search_space_sizes = []
for k, v in z3_stat.items():
    z3_conflicts.append(v["conflicts"])
    #  "lgp-test-6x4-38" --> N= 6, M = 4
    N, M = int(k.split("-")[2][0]), int(k.split("-")[2][2])
    factorial_N = math.factorial(N)
    search_space_size = factorial_N ** M
    search_space_sizes.append(search_space_size)

# draw the figure where x is the search space size and y is the number of conflicts

# Create figure
plt.figure(figsize=(12, 10))
plt.rcParams.update({'font.size': 24})

# Create binned data with mean and std for error bars
df = pd.DataFrame({
    'search_space_sizes': search_space_sizes,
    'z3_conflicts': z3_conflicts
})
df["search_space_bin"] = pd.qcut(df["search_space_sizes"], q=15, duplicates='drop')
bin_stats = df.groupby("search_space_bin").agg({
    'z3_conflicts': ['mean', 'std'],
    'search_space_sizes': 'mean'
}).reset_index()

# Extract values
bin_centers = bin_stats['search_space_sizes']['mean']
conflicts_mean = bin_stats['z3_conflicts']['mean']
conflicts_std = bin_stats['z3_conflicts']['std']


show_error_bars = False 

# Plot line with or without error bars based on parameter
if show_error_bars:
    plt.errorbar(bin_centers, conflicts_mean, yerr=conflicts_std, 
                fmt='o-', capsize=5, capthick=1, elinewidth=1)
else:
    plt.plot(bin_centers, conflicts_mean, 'o-')

# Use log scale for x-axis due to potentially large search space sizes
plt.xscale('log')
# Add grid lines
plt.grid(True)

# Add labels and title with larger font
plt.xlabel('Search Space Size', fontsize=18)
plt.ylabel('Number of Z3 Conflicts', fontsize=18)
plt.title('Z3 Conflicts vs Search Space Size', fontsize=18)

# Save the figure
plt.savefig('zebra_logic_analysis/z3_conflicts_vs_search_space.png')
plt.close()

# Create a heatmap of average Z3 conflicts by problem size

def categorize_conflicts(search_space_size):
    if search_space_size < 10**3:
        category = "Small"
    elif 10**3 <= search_space_size < 10**6:
        category = "Medium"
    elif 10**6 <= search_space_size < 10**10:
        category = "Large"
    else:
        category = "X-Large"
    return category

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

# Create a grid to store average conflicts for each N,M combination
heatmap_data_conflicts = np.zeros((5, 5))  # For 2x2 to 6x6
conflict_counts = np.zeros((5, 5))  # To count number of instances for averaging

# Populate the grid with conflict data
for k, v in z3_stat.items():
    # Parse N and M from the key (e.g., "lgp-test-6x4-38")
    N, M = int(k.split("-")[2][0]), int(k.split("-")[2][2])
    if 2 <= N <= 6 and 2 <= M <= 6:  # Only consider 2x2 to 6x6
        heatmap_data_conflicts[N-2, M-2] += v["conflicts"]
        conflict_counts[N-2, M-2] += 1

# Calculate averages, avoiding division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    heatmap_data_conflicts = np.divide(heatmap_data_conflicts, conflict_counts)
    heatmap_data_conflicts = np.nan_to_num(heatmap_data_conflicts)  # Replace NaN with 0

# Create the heatmap
plt.figure(figsize=(12, 10))
plt.rcParams.update({'font.size': 24})

# Create annotations with both value and category
annotations = []
for i in range(heatmap_data_conflicts.shape[0]):
    row = []
    for j in range(heatmap_data_conflicts.shape[1]):
        value = heatmap_data_conflicts[i, j]
        search_space_size, category = categorize_search_space(i+2, j+2)  # Add 2 to get actual N,M values
        row.append(f"{value:.2f}\n{category}")  # Changed from :.0f to :.2f
    annotations.append(row)

sns.heatmap(
    heatmap_data_conflicts,
    annot=annotations,
    fmt='',
    cbar_kws={'label': 'Average Number of Z3 Conflicts'},
    cmap='YlOrRd',
    linewidths=0.5,
    xticklabels=[f"{i}" for i in range(2, 7)],
    yticklabels=[f"{i}" for i in range(2, 7)],
    annot_kws={"size": 18}
)

plt.xlabel('Number of Attributes (M)', fontsize=22)
plt.ylabel('Number of Houses (N)', fontsize=22)
plt.title('Average Z3 Conflicts by Problem Size (2x2 to 6x6)', fontsize=22, pad=15)

plt.tight_layout()
plt.savefig('zebra_logic_analysis/z3_conflicts_heatmap.pdf', dpi=300)
plt.close()
