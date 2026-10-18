```bash
bash zero_eval_api.sh -f openai -d wildbench_v2-hard -m openai/gpt-4o-mini-2024-07-18 -p gpt-4o-mini-2024-07-18 -s 1

bash zero_eval_api.sh -f openai -d wildbench_v2-hard -m openai/o1-mini-2024-09-12 -p o1-mini-2024-09-12 -s 8

bash zero_eval_api.sh -f openai -d wildbench_v2-hard -m openai/o1-preview-2024-09-12 -p o1-preview-2024-09-12 -s 8


bash zero_eval_api.sh -f openai -d wildbench_v2-hard -m openai/gpt-4o-2024-08-06 -p gpt-4o-2024-08-06 -s 8

bash zero_eval_api.sh -f anthropic -d wildbench_v2-hard -m anthropic/claude-3-5-sonnet-20241022 -p claude-3-5-sonnet-20241022 -s 8

bash zero_eval_api.sh -d wildbench_v2-hard -f together -m meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo -p Llama-3.1-405B-Inst-together-fp8 -s 8 -x 2048
bash zero_eval_api.sh -d wildbench_v2-hard -f together -m meta-llama/Llama-3.2-3B-Instruct-Turbo -p Llama-3.2-3B-Inst-together -s 8 -x 4096

bash zero_eval_api.sh -f openai -d wildbench_v2-hard -m Meta-Llama-3.1-70B-Instruct@sambanova -p Meta-Llama-3.1-70B-Instruct@sambanova -s 4 -x 3072
bash zero_eval_api.sh -f openai -d wildbench_v2-hard -m Meta-Llama-3.1-8B-Instruct@sambanova -p Meta-Llama-3.1-8B-Instruct@sambanova -s 4 -x 3072
bash zero_eval_api.sh -f openai -d wildbench_v2-hard -m Meta-Llama-3.2-3B-Instruct@sambanova -p Meta-Llama-3.2-3B-Instruct@sambanova -s 4 -x 3072

bash zero_eval_api.sh -f openai -d wildbench_v2-hard -m Qwen/Qwen2.5-72B-Instruct@hyperbolic -p Qwen2.5-72B-Instruct@hyperbolic -s 4 -x 3072
# meta-llama/Llama-3.2-3B-Instruct
bash zero_eval_api.sh -f openai -d wildbench_v2-hard -m meta-llama/Llama-3.2-3B-Instruct@hyperbolic -p Llama-3.2-3B-Instruct@hyperbolic -s 4 -x 3072



# Qwen/Qwen2.5-72B-Instruct
````