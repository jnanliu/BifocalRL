import json 

# example format 
"""
[
  {
    "session_id": "lgp-test-5x6-16",
    "chat_history": [
      "\n# Example Puzzle \n\nThere are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:\n - Each person has a unique name: `Peter`, `Eric`, `Arnold`.\n - Each person has a unique favorite drink: `tea`, `water`, `milk`\n\n## Clues for the Example Puzzle\n\n1. Peter is in the second house.\n2. Arnold is directly left of the one who only drinks water.\n3. The one who only drinks water is directly left of the person who likes milk.\n\n## Answer to the Example Puzzle\n\n{\n    \"reasoning\": \"Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.\",\n    \"solution\": {\n        \"House 1\": {\n            \"Name\": \"Arnold\",\n            \"Drink\": \"tea\"\n        },\n        \"House 2\": {\n            \"Name\": \"Peter\",\n            \"Drink\": \"water\"\n        },\n        \"House 3\": {\n            \"Name\": \"Eric\",\n            \"Drink\": \"milk\"\n        }\n    }\n}\n\n# Puzzle to Solve \n\nThere are 5 houses, numbered 1 to 5 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:\n - Each person has a unique name: `Peter`, `Alice`, `Bob`, `Eric`, `Arnold`\n - The people are of nationalities: `norwegian`, `german`, `dane`, `brit`, `swede`\n - People have unique favorite book genres: `fantasy`, `biography`, `romance`, `mystery`, `science fiction`\n - Everyone has something unique for lunch: `stir fry`, `grilled cheese`, `pizza`, `spaghetti`, `stew`\n - Each person has a favorite color: `red`, `green`, `blue`, `yellow`, `white`\n - The people keep unique animals: `bird`, `dog`, `cat`, `horse`, `fish`\n\n## Clues:\n1. The person who loves fantasy books is the Norwegian.\n2. The cat lover and the person who loves biography books are next to each other.\n3. The German is Bob.\n4. The person who loves yellow is Bob.\n5. The person whose favorite color is green is Peter.\n6. There is one house between the Dane and the person who is a pizza lover.\n7. The person who loves blue is somewhere to the left of the Dane.\n8. The person who loves eating grilled cheese is somewhere to the left of the Norwegian.\n9. The person who loves the spaghetti eater is Peter.\n10. The person who keeps horses is Alice.\n11. The fish enthusiast is directly left of the person who loves science fiction books.\n12. There is one house between the Norwegian and Arnold.\n13. The person who loves romance books is the British person.\n14. There are two houses between the Norwegian and Alice.\n15. The bird keeper is the person whose favorite color is red.\n16. The dog owner is directly left of the fish enthusiast.\n17. The person who loves the stew is the Norwegian.\n\n\n\n# Instruction\n\nNow please solve the above puzzle. Present your reasoning and solution in the following json format:\n\n{\n    \"reasoning\": \"___\",\n    \"solution\": {\n        \"House 1\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        },\n        \"House 2\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        },\n        \"House 3\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        },\n        \"House 4\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        },\n        \"House 5\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        }\n    }\n}\n\n"
    ],
    "model_input": "n/a",
    "output": [
        ...
"""

# merge the output filed by the sessoin_id from multiple json files into a single json file
# source_files = [
#     "result_dirs/zebra-grid/bon_32/gpt-4o-mini-2024-07-18.json",
#     "result_dirs/zebra-grid/bon_32_v2/gpt-4o-mini-2024-07-18.json",
#     "result_dirs/zebra-grid/bon_64/gpt-4o-mini-2024-07-18.json",
#     "result_dirs/zebra-grid/bon_64_v2/gpt-4o-mini-2024-07-18.json",
#     "result_dirs/zebra-grid/bon_64_v3/gpt-4o-mini-2024-07-18.json",
# ]
# target_file = "result_dirs/zebra-grid/bon_all/gpt-4o-mini-2024-07-18.json"

#

source_files = [
    "result_dirs/zebra-grid/bon_32/gpt-4o-2024-08-06.json", 
    "result_dirs/zebra-grid/bon_32_v2/gpt-4o-2024-08-06.json", 
    "result_dirs/zebra-grid/bon_64_v2/gpt-4o-2024-08-06.json",  # mistake ... it is actually only 32 here 
    "result_dirs/zebra-grid/bon_32_v3/gpt-4o-2024-08-06.json",  
]
target_file = "result_dirs/zebra-grid/bon_all/gpt-4o-2024-08-06.json"

merged_data = {}
for file_path in source_files:
    with open(file_path, 'r') as file:
        data = json.load(file)
        for entry in data:
            session_id = entry['session_id']
            if session_id not in merged_data:
                merged_data[session_id] = entry
            else:
                # If session_id already exists, merge the output
                merged_data[session_id]['output'].extend(entry['output'])
# print the size of 'output'
for session_id, entry in merged_data.items():
    print(f"Output Size: {len(entry['output'])}")
    break 
# Write the merged data to the target file
with open(target_file, 'w') as file:
    json.dump(list(merged_data.values()), file, indent=2)

