"""
TTRL evaluation backend.

Provides TTRL-style three-tier math scoring (mathd -> sympy -> math_verify)
plus task-specific TTRL scorers such as GSM8K/GPQA, and the serial multi-model
evaluation runner with prompt mode presets.
"""
