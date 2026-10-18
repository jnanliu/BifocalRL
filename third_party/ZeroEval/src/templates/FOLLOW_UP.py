SELF_VERIFICATION = """
Please review the initial prompt, including the question and the constraints or requirements provided. 
Reassess your reasoning and the answer you provided to ensure they align with the given information. 
If any adjustments are needed, modify your reasoning and answer accordingly. 
Finally, present your response in the same JSON format mentioned in the initial prompt.
If the original answer was already correct, you can simply repeat it in the same JSON format.
"""

NEGATIVE_FEEDBACK = """
Your answer may be incorrect! 
Identify any mistakes in your reasoning and answer, if any.
Correct them to ensure they align with the given information. 
Present your updated response in the same JSON format mentioned in the initial prompt. 
"""

# for zebra logic 
NEGATIVE_FEEDBACK_ORACLE_ZEBRA = """
You are incorrect! Re-examine the clues, correct the mistakes, and then provide the revised solution in the original JSON format.
"""