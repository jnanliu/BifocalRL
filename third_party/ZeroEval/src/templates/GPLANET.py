# GPLANET Template

GPLANET = """

# Instructions

You are an intelligent robot that can move around a room and interact with objects.
We will provide you with a list of objects in the room, and a task description for you to complete.
The object information is given in a json format and there are 3D coordinates for each object, and their rotations as well as their receptacle (parent) information.
The task description will be given in a natural language description. You are given a shuffled list of candidate actions that you can perform.
Your job is to complete the task by selecting the correct actions and ordering them in the proper sequence.

## Detailed Instructions

1. **Review room objects:** Examine the list of objects below, noting each object's attributes: id, object_type, position, rotation, and parent_receptacle.
2. **Understand the task:** Read the task description carefully to identify the goal.
3. **Evaluate candidate actions:** Review the provided actions. Exclude exactly **two** irrelevant actions that are not needed in the final sequence.
4. **Sequence the actions:** Select the necessary actions and order them correctly to complete the task.
5. **Submit your answer:** Include reasoning if desired, but conclude with the answer sequence, ending the response afterward.



# Example Task

## Objects in the example room
<objects>
```json
[
    {
        "id": "0",
        "object_type": "Agent",
        "position_xyz": "(1.25, 0.900, 0.5)",
        "rotation_xyz": "(0, 0, 0)",
        "parent_receptacle": "N/A"
    },
    {
        "id": "8",
        "object_type": "Watch",
        "position_xyz": "(-0.862, 0.552, -0.192)",
        "rotation_xyz": "(0, 270, 0)",
        "parent_receptacle": "30"
    },
    {
        "id": "30",
        "object_type": "CoffeeTable",
        "position_xyz": "(-0.711, 0.002, -0.364)",
        "rotation_xyz": "(0, 180, 0)",
        "parent_receptacle": "N/A"
    },
    {
        "id": "36",
        "object_type": "SideTable",
        "position_xyz": "(1.807, 0.000, 0.326)",
        "rotation_xyz": "(0, 270, 0)",
        "parent_receptacle": "N/A"
    },
    {
        "id": "14",
        "object_type": "Statue",
        "position_xyz": "(1.807, 0.880, 0.537)",
        "rotation_xyz": "(0, 270, 0)",
        "parent_receptacle": "36"
    },
    {
        "id": "15",
        "object_type": "Statue",
        "position_xyz": "(1.682, 0.880, 0.220)",
        "rotation_xyz": "(0, 90, 0)",
        "parent_receptacle": "36"
    }
]
```
</objects>
Note that there are more objects in the room, but we only show the relevant ones for the example task.

## Task Description

<task>
Move the wristwatch from the coffee table to the dresser.
</task>

## Candidate Actions

Here are the candidate actions that you can perform:

<candidate_actions>
- (A) Turn right and walk towards the bookshelf.
- (B) Put the wristwatch down on the dresser to the right of the grey statue.
- (C) Pick up a book from the coffee table.
- (D) Turn left and walk to the dresser with statues on it.
- (E) Pick up the wristwatch from the coffee table.
- (F) Turn left and walk to the coffee table.
</candidate_actions>

## Expected response from the robotic agent


To move the wristwatch from the coffee table to the dresser, a robot should perform the following actions in the following order:

1. **Turn left and walk to the coffee table.** (Action F)
2. **Pick up the wristwatch from the coffee table.** (Action E)
3. **Turn left and walk to the dresser with statues on it.** (Action D)
4. **Put the wristwatch down on the dresser to the right of the grey statue.** (Action B)


The irrelevant actions are:

- (A) Turn right and walk towards the bookshelf.
- (C) Pick up a book from the coffee table.

Therefore, the final output from the model should be:

<final_action_sequence>
(F) (E) (D) (B) END
</final_action_sequence>

# The Current Room, Task, and Candidate Actions


## Objects in the current room

<objects>
{objects_str}
</objects>

## Task Description to complete for the current room now
<task>
{task}
</task>

## Candidate Actions for the current room

<candidate_actions>
{actions}
</candidate_actions>

Note that there are TWO and ONLY TWO irrelevant actions that you should exclude from the final plan. Therefore, your final output should be a sequence of actions that is exactly two actions shorter than the total number of candidate actions. For example, if there are 10 candidate actions, your final output should be a sequence of 8 actions.


# Instruction Recap

## Detailed Instructions

1. **Review room objects:** Examine the list of objects below, noting each object's attributes: id, object_type, position, rotation, and parent_receptacle.
2. **Understand the task:** Read the task description carefully to identify the goal.
3. **Evaluate candidate actions:** Review the provided actions. Exclude exactly **two** irrelevant actions that are not needed in the final sequence.
4. **Sequence the actions:** Select the necessary actions and order them correctly to complete the task.
5. **Submit your answer:** Include reasoning if desired, but conclude with the answer sequence, ending the response afterward.

## Output format

You can think step by step and reason about the task and the candidate actions. Finally, you should output the final action sequence inside the <final_action_sequence> tag (e.g., <final_action_sequence> (A) (B) (C) END </final_action_sequence>).



"""
