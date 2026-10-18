import json
import math 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

input_file_name = "result_dirs_parsed/zebra-grid/o1-preview-2024-09-12-v2.json"
output_file_name = "zebra_logic_analysis/o1_preview_hidden_reasoning_vs_search_space_size_with_curve.png"

# input_file_name = "result_dirs_parsed/zebra-grid/o1-mini-2024-09-12-v3.json"
# output_file_name = "zebra_logic_analysis/o1_mini_hidden_reasoning_vs_search_space_size_with_curve.png"

with open(input_file_name) as f:
    data = json.load(f)




# The data structure is a dictionary with the following keys:
"""
[
  {
    "session_id": "lgp-test-5x6-16",
    "chat_history": [
      "\n# Example Puzzle \n\nThere are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:\n - Each person has a unique name: `Peter`, `Eric`, `Arnold`.\n - Each person has a unique favorite drink: `tea`, `water`, `milk`\n\n## Clues for the Example Puzzle\n\n1. Peter is in the second house.\n2. Arnold is directly left of the one who only drinks water.\n3. The one who only drinks water is directly left of the person who likes milk.\n\n## Answer to the Example Puzzle\n\n{\n    \"reasoning\": \"Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.\",\n    \"solution\": {\n        \"House 1\": {\n            \"Name\": \"Arnold\",\n            \"Drink\": \"tea\"\n        },\n        \"House 2\": {\n            \"Name\": \"Peter\",\n            \"Drink\": \"water\"\n        },\n        \"House 3\": {\n            \"Name\": \"Eric\",\n            \"Drink\": \"milk\"\n        }\n    }\n}\n\n# Puzzle to Solve \n\nThere are 5 houses, numbered 1 to 5 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:\n - Each person has a unique name: `Peter`, `Alice`, `Bob`, `Eric`, `Arnold`\n - The people are of nationalities: `norwegian`, `german`, `dane`, `brit`, `swede`\n - People have unique favorite book genres: `fantasy`, `biography`, `romance`, `mystery`, `science fiction`\n - Everyone has something unique for lunch: `stir fry`, `grilled cheese`, `pizza`, `spaghetti`, `stew`\n - Each person has a favorite color: `red`, `green`, `blue`, `yellow`, `white`\n - The people keep unique animals: `bird`, `dog`, `cat`, `horse`, `fish`\n\n## Clues:\n1. The person who loves fantasy books is the Norwegian.\n2. The cat lover and the person who loves biography books are next to each other.\n3. The German is Bob.\n4. The person who loves yellow is Bob.\n5. The person whose favorite color is green is Peter.\n6. There is one house between the Dane and the person who is a pizza lover.\n7. The person who loves blue is somewhere to the left of the Dane.\n8. The person who loves eating grilled cheese is somewhere to the left of the Norwegian.\n9. The person who loves the spaghetti eater is Peter.\n10. The person who keeps horses is Alice.\n11. The fish enthusiast is directly left of the person who loves science fiction books.\n12. There is one house between the Norwegian and Arnold.\n13. The person who loves romance books is the British person.\n14. There are two houses between the Norwegian and Alice.\n15. The bird keeper is the person whose favorite color is red.\n16. The dog owner is directly left of the fish enthusiast.\n17. The person who loves the stew is the Norwegian.\n\n\n\n# Instruction\n\nNow please solve the above puzzle. Present your reasoning and solution in the following json format:\n\n{\n    \"reasoning\": \"___\",\n    \"solution\": {\n        \"House 1\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        },\n        \"House 2\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        },\n        \"House 3\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        },\n        \"House 4\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        },\n        \"House 5\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        }\n    }\n}\n\n"
    ],
    "model_input": "n/a",
    "output": [
      "```json\n{\n    \"reasoning\": \"By systematically applying the given clues:\n\n1. **Name and Nationality Assignments:**\n   - **Bob** is identified as the **German** who loves **yellow** (Clues 3 & 4).\n   - **Norwegian** loves **fantasy books** and **stew** (Clues 1 & 17).\n   - **Peter** loves **green** and **spaghetti** (Clues 5 & 9).\n   - **Alice** keeps **horses** (Clue 10).\n   - **Arnold** is the remaining name.\n\n2. **House Positioning:**\n   - The **Norwegian** cannot be in House 1 due to Clue 8, so the **Norwegian** is placed in House 3.\n   - **Alice** is placed in House 5, two houses away from the **Norwegian** (Clue 14).\n   - **Arnold** is placed in House 2, one house away from the **Norwegian** (Clue 12).\n\n3. **Color Assignments:**\n   - **Bob** in House 1 is **yellow**.\n   - **Peter** in House 4 is **green**.\n   - The **Norwegian** in House 3 is **blue**.\n   - **Arnold** in House 2 is **red** (due to Clue 15: bird keeper is red).\n   - **Alice** in House 5 is **white**.\n\n4. **Animal Assignments:**\n   - **Arnold** keeps **bird** (Clue 15).\n   - **Bob** keeps **dog** (from the sequence clues 16 & 11).\n   - **Norwegian** keeps **fish**.\n   - **Peter** keeps **cat**.\n   - **Alice** keeps **horses**.\n\n5. **Book Genre Assignments:**\n   - **Arnold** loves **mystery books**.\n   - **Bob** loves **romance books** (Clue 13: British person loves romance; however, since Bob is German, this implies another adjustment, leading to assigning **romance** to the appropriate British individual).\n   - **Norwegian** loves **fantasy books**.\n   - **Peter** loves **science fiction books**.\n   - **Alice** loves **biography books**.\n\n6. **Food Assignments:**\n   - **Bob** loves **grilled cheese** (Clue 8).\n   - **Norwegian** loves **stew**.\n   - **Peter** loves **spaghetti**.\n   - **Arnold** loves **pizza** (from elimination).\n   - **Alice** loves **stir fry**.\n\n7. **Final Adjustments:**\n   - Ensured all clues are satisfied without conflicts, particularly addressing adjacency conditions and unique attribute assignments.\n\n**Final Assignments:**\n\n\",\n    \"solution\": {\n        \"House 1\": {\n            \"Name\": \"Bob\",\n            \"Nationality\": \"German\",\n            \"BookGenre\": \"romance\",\n            \"Food\": \"grilled cheese\",\n            \"Color\": \"yellow\",\n            \"Animal\": \"dog\"\n        },\n        \"House 2\": {\n            \"Name\": \"Arnold\",\n            \"Nationality\": \"British\",\n            \"BookGenre\": \"mystery\",\n            \"Food\": \"pizza\",\n            \"Color\": \"red\",\n            \"Animal\": \"bird\"\n        },\n        \"House 3\": {\n            \"Name\": \"Eric\",\n            \"Nationality\": \"Norwegian\",\n            \"BookGenre\": \"fantasy\",\n            \"Food\": \"stew\",\n            \"Color\": \"blue\",\n            \"Animal\": \"fish\"\n        },\n        \"House 4\": {\n            \"Name\": \"Peter\",\n            \"Nationality\": \"Dane\",\n            \"BookGenre\": \"science fiction\",\n            \"Food\": \"spaghetti\",\n            \"Color\": \"green\",\n            \"Animal\": \"cat\"\n        },\n        \"House 5\": {\n            \"Name\": \"Alice\",\n            \"Nationality\": \"Swede\",\n            \"BookGenre\": \"biography\",\n            \"Food\": \"stir fry\",\n            \"Color\": \"white\",\n            \"Animal\": \"horses\"\n        }\n    }\n}\n```"
    ],
    "generator": "openai/o1-mini-2024-09-12",
    "configs": {
      "engine": "openai",
      "repetition_penalty": 1.0,
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": 4096
    },
    "dataset": "zebra-grid",
    "id": "lgp-test-5x6-16",
    "size": "5*6",
    "puzzle": "There are 5 houses, numbered 1 to 5 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:\n - Each person has a unique name: `Peter`, `Alice`, `Bob`, `Eric`, `Arnold`\n - The people are of nationalities: `norwegian`, `german`, `dane`, `brit`, `swede`\n - People have unique favorite book genres: `fantasy`, `biography`, `romance`, `mystery`, `science fiction`\n - Everyone has something unique for lunch: `stir fry`, `grilled cheese`, `pizza`, `spaghetti`, `stew`\n - Each person has a favorite color: `red`, `green`, `blue`, `yellow`, `white`\n - The people keep unique animals: `bird`, `dog`, `cat`, `horse`, `fish`\n\n## Clues:\n1. The person who loves fantasy books is the Norwegian.\n2. The cat lover and the person who loves biography books are next to each other.\n3. The German is Bob.\n4. The person who loves yellow is Bob.\n5. The person whose favorite color is green is Peter.\n6. There is one house between the Dane and the person who is a pizza lover.\n7. The person who loves blue is somewhere to the left of the Dane.\n8. The person who loves eating grilled cheese is somewhere to the left of the Norwegian.\n9. The person who loves the spaghetti eater is Peter.\n10. The person who keeps horses is Alice.\n11. The fish enthusiast is directly left of the person who loves science fiction books.\n12. There is one house between the Norwegian and Arnold.\n13. The person who loves romance books is the British person.\n14. There are two houses between the Norwegian and Alice.\n15. The bird keeper is the person whose favorite color is red.\n16. The dog owner is directly left of the fish enthusiast.\n17. The person who loves the stew is the Norwegian.\n",
    "created_at": "2024-07-11T01:22:10.734298",
    "hidden_reasoning_token": 13952,
    "reasoning": "By systematically applying the given clues:1. **Name and Nationality Assignments:**   - **Bob** is identified as the **German** who loves **yellow** (Clues 3 & 4).   - **Norwegian** loves **fantasy books** and **stew** (Clues 1 & 17).   - **Peter** loves **green** and **spaghetti** (Clues 5 & 9).   - **Alice** keeps **horses** (Clue 10).   - **Arnold** is the remaining name.2. **House Positioning:**   - The **Norwegian** cannot be in House 1 due to Clue 8, so the **Norwegian** is placed in House 3.   - **Alice** is placed in House 5, two houses away from the **Norwegian** (Clue 14).   - **Arnold** is placed in House 2, one house away from the **Norwegian** (Clue 12).3. **Color Assignments:**   - **Bob** in House 1 is **yellow**.   - **Peter** in House 4 is **green**.   - The **Norwegian** in House 3 is **blue**.   - **Arnold** in House 2 is **red** (due to Clue 15: bird keeper is red).   - **Alice** in House 5 is **white**.4. **Animal Assignments:**   - **Arnold** keeps **bird** (Clue 15).   - **Bob** keeps **dog** (from the sequence clues 16 & 11).   - **Norwegian** keeps **fish**.   - **Peter** keeps **cat**.   - **Alice** keeps **horses**.5. **Book Genre Assignments:**   - **Arnold** loves **mystery books**.   - **Bob** loves **romance books** (Clue 13: British person loves romance; however, since Bob is German, this implies another adjustment, leading to assigning **romance** to the appropriate British individual).   - **Norwegian** loves **fantasy books**.   - **Peter** loves **science fiction books**.   - **Alice** loves **biography books**.6. **Food Assignments:**   - **Bob** loves **grilled cheese** (Clue 8).   - **Norwegian** loves **stew**.   - **Peter** loves **spaghetti**.   - **Arnold** loves **pizza** (from elimination).   - **Alice** loves **stir fry**.7. **Final Adjustments:**   - Ensured all clues are satisfied without conflicts, particularly addressing adjacency conditions and unique attribute assignments.**Final Assignments:**",
    "correct_cells": 7,
    "total_cells": 30,
    "solved": false
  },
  ...
]
"""


# search space size can be calculated by the `size` key by using the formula: (N!)^M where N*M is from the size key, write a function to get arg like "3*4" to return the search space size
def search_space_size(size):
    N, M = map(int, size.split("*"))
    # factorial of N and times M 
    return (math.factorial(N))**M 


# draw a figure to show the hidden_reasoning_token vs the search space size 
# the y-axis is the hidden_reasoning_token (scatter plot)
# the x-axis is the search space size

# get the hidden_reasoning_token and size from the data
hidden_reasoning_token = [d["hidden_reasoning_token"] for d in data]
size = [d["size"] for d in data]
# Calculate the search space size for each puzzle
search_space_sizes = [search_space_size(s) for s in size]
 


# Create a DataFrame for plotting
df = pd.DataFrame({
    'hidden_reasoning_token': hidden_reasoning_token,
    'search_space_size': search_space_sizes
})

# Get the solved status from the data
solved_status = [d["solved"] for d in data]
df['solved'] = solved_status



# Plot the data with different colors for solved and unsolved
# Set a modern font style
plt.rcParams.update({
    'font.family': 'sans-serif', 
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 16  # Increase the font size
})

# Plot the data with different colors for solved and unsolved
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='search_space_size', y='hidden_reasoning_token', 
                hue='solved', style='solved', 
                palette={True: '#4a90e2', False: '#d9534f'},  # Blue for Correct, Red for Incorrect
                markers={True: 'o', False: 'X'},  # Square for Correct, Triangle for Incorrect
                hue_order=[True, False], legend='full',
                s=100)  # Increase the symbol size
# Update legend labels
plt.legend(title='Solution Status', labels=['Incorrect', 'Correct'])

plt.xscale('log')  # Use a logarithmic scale for the x-axis due to potentially large search space sizes
plt.xlabel('Search Space Size (log scale)')
plt.ylabel('# Hidden CoT Tokens')
# plt.title('Hidden Reasoning Token vs. Search Space Size')
plt.ylim(0, 20000)  # Set the y-axis range from 0 to 20000
plt.grid(True)


# # Set more fine-grained intervals for the x-axis
# x_ticks = np.logspace(np.floor(np.log10(df['search_space_size'].min())), 
#                       np.ceil(np.log10(df['search_space_size'].max())), 
#                       num=20)  # Adjust 'num' for more or fewer ticks
# plt.xticks(x_ticks, labels=[f"{int(tick):,}" for tick in x_ticks])



# # Set more fine-grained intervals for the x-axis with scientific notation
# plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))


# Fit a polynomial curve to the data (e.g., degree 2)
log_x = np.log10(df['search_space_size'])  # Log-transform the x-axis data
coeffs = np.polyfit(log_x, df['hidden_reasoning_token'], deg=2)  # Fit a polynomial of degree 2
poly = np.poly1d(coeffs)

# Generate values for the curve
x_fit = np.linspace(log_x.min(), log_x.max(), 500)  # Generate 500 points between min and max of log_x
y_fit = poly(x_fit)

# Plot the curve
plt.plot(10**x_fit, y_fit, color='green', label='Polynomial Fit (degree 2)', linewidth=2)  # Convert x_fit back to original scale

 
plt.savefig(output_file_name, dpi=300)
print(f"Saved the plot to {output_file_name}")
# plt.show()

 