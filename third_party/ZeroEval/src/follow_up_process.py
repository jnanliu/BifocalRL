"""
This script is to add a follow-up instruction to the existing result file, 
where there is already a chat-history of the previous conversation, and an existing output (at least one) from the model.
"""
import json, os
from templates import FOLLOW_UP

def add_follow_up_instruction(file_path, output_path, follow_up_mode="self_verification"):
    # Load the existing data
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for item in data:
        # Add the follow-up instruction to each item
        chat_history = item["chat_history"]
        current_output = item["output"][0]
        if follow_up_mode == "self_verification":
            follow_up_prompt = FOLLOW_UP.SELF_VERIFICATION
        elif follow_up_mode == "neg_feedback":
            follow_up_prompt = FOLLOW_UP.NEGATIVE_FEEDBACK
        elif follow_up_mode == "zebra_oracle":
            if item["solved"]:
                follow_up_prompt = "Your answer is correct. Please repeat the json-formatted output again."
            else:
                follow_up_prompt = FOLLOW_UP.NEGATIVE_FEEDBACK_ORACLE_ZEBRA
        else:
            raise ValueError(f"Unknown follow_up_mode: {follow_up_mode}")
        new_chat_history = chat_history + [current_output] + [follow_up_prompt]
        item["chat_history"] = new_chat_history
        item["output"] = []
    
    # save the modified data back to the file
    # create the output directory (and the parents) if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=2)

if __name__ == "__main__":


    # Negative feedback T=2

    file_path = "result_dirs/zebra-grid/neg_feedback_v2/gpt-4o-mini-2024-07-18.neg_feedback_v2.T=1.json"
    follow_up_mode = "neg_feedback"
    output_file = "result_dirs_follow_up/zebra-grid/gpt-4o-mini-2024-07-18.neg_feedback_v2.T=2.json"
    add_follow_up_instruction(file_path, output_file, follow_up_mode)  # Call the function to add follow-up instruction

    file_path = "result_dirs/zebra-grid/neg_feedback_v2/gpt-4o-2024-08-06.neg_feedback_v2.T=1.json"
    output_file = "result_dirs_follow_up/zebra-grid/gpt-4o-2024-08-06.neg_feedback_v2.T=2.json"
    add_follow_up_instruction(file_path, output_file, follow_up_mode)  # Call the function to add follow-up instruction

    exit()

    
    # Example usage
    file_path = "result_dirs/zebra-grid/gpt-4o-mini-2024-07-18.json"
    follow_up_mode = "self_verification"
    output_file = "result_dirs_follow_up/zebra-grid/gpt-4o-mini-2024-07-18.self_verification.T=1.json"
    add_follow_up_instruction(file_path, output_file, follow_up_mode)  # Call the function to add follow-up instruction

    file_path = "result_dirs/zebra-grid/gpt-4o-2024-08-06.json"
    output_file = "result_dirs_follow_up/zebra-grid/gpt-4o-2024-08-06.self_verification.T=1.json"
    add_follow_up_instruction(file_path, output_file, follow_up_mode)  # Call the function to add follow-up instruction
    

    # Negative feedback 

    file_path = "result_dirs/zebra-grid/gpt-4o-mini-2024-07-18.json"
    follow_up_mode = "neg_feedback"
    output_file = "result_dirs_follow_up/zebra-grid/gpt-4o-mini-2024-07-18.neg_feedback_v2.T=1.json"
    add_follow_up_instruction(file_path, output_file, follow_up_mode)  # Call the function to add follow-up instruction

    file_path = "result_dirs/zebra-grid/gpt-4o-2024-08-06.json"
    output_file = "result_dirs_follow_up/zebra-grid/gpt-4o-2024-08-06.neg_feedback_v2.T=1.json"
    add_follow_up_instruction(file_path, output_file, follow_up_mode)  # Call the function to add follow-up instruction
    


    


    # Zebra Oracle Feedback 
    file_path = "result_dirs_parsed/zebra-grid/gpt-4o-mini-2024-07-18.json"
    follow_up_mode = "zebra_oracle"
    output_file = "result_dirs_follow_up/zebra-grid/gpt-4o-mini-2024-07-18.zebra_oracle.T=1.json"
    add_follow_up_instruction(file_path, output_file, follow_up_mode)  # Call the function to add follow-up instruction

    file_path = "result_dirs_parsed/zebra-grid/gpt-4o-2024-08-06.json"
    output_file = "result_dirs_follow_up/zebra-grid/gpt-4o-2024-08-06.zebra_oracle.T=1.json"
    add_follow_up_instruction(file_path, output_file, follow_up_mode)  # Call the function to add follow-up instruction
    
