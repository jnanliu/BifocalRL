"""
LightEval evaluation backend.

Wraps the external ``lighteval`` CLI with vLLM, custom tasks, and local
dataset support. Includes an idle watchdog for stuck vLLM subprocesses.
"""
