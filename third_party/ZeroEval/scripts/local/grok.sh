# bash zero_eval_api.sh -f openai -d gsm -m grok-2-1212@xai -p grok-2-1212 -s 1
# wait
# bash zero_eval_api.sh -f openai -d crux -m grok-2-1212@xai -p grok-2-1212 -s 8
# wait
bash zero_eval_api.sh -f openai -d zebra-grid -m grok-2-1212@xai -p grok-2-1212 -s 8
wait
bash zero_eval_api.sh -f openai -d math-l5 -m grok-2-1212@xai -p grok-2-1212 -s 8
wait
bash zero_eval_api.sh -f openai -d mmlu-redux -m grok-2-1212@xai -p grok-2-1212 -s 8
wait


bash zero_eval_api.sh -f openai -d zebra-grid -m grok-3-mini-fast-beta-low@xai -p grok-3-mini-fast-beta-low -s 16
wait
bash zero_eval_api.sh -f openai -d zebra-grid -m grok-3-mini-fast-beta-high@xai -p grok-3-mini-fast-beta-high -s 16
wait


bash zero_eval_api.sh -f openai -d zebra-grid -m grok-3-fast-beta@xai -p grok-3-fast-beta -s 16
wait