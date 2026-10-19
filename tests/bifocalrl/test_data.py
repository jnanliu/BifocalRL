import importlib.util
import json
from pathlib import Path


def test_conversion_keeps_training_unlabelled_and_supports_base_models(tmp_path):
    path=Path(__file__).resolve().parents[2]/"examples/bifocalrl/prepare_data.py"
    spec=importlib.util.spec_from_file_location("bifocal_prepare_data",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data=[{"question":"q","answer":"reasoning\n#### 42"}]
    output=tmp_path/"data"/"train.jsonl"
    module.convert(data,output,prompt_key="question")
    assert json.loads(output.read_text())=={"prompt":[{"role":"user","content":"q"}]}
    module.convert(data,output,prompt_key="question",plain_prompts=True)
    assert json.loads(output.read_text())=={"prompt":"q"}
    module.convert(data,output,prompt_key="question",label_key="answer",label_format="gsm8k")
    assert json.loads(output.read_text())["label"]=="42"
