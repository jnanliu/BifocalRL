import importlib.util
import json
from pathlib import Path


def test_conversion_keeps_training_unlabelled_and_supports_base_models(tmp_path):
    path = Path(__file__).resolve().parents[2] / "examples/bifocalrl/prepare_data.py"
    spec = importlib.util.spec_from_file_location("bifocal_prepare_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = [{"question": "q", "answer": "reasoning\n#### 42"}]
    output = tmp_path / "data" / "train.jsonl"
    module.convert(data, output, prompt_key="question")
    assert json.loads(output.read_text()) == {"prompt": [{"role": "user", "content": "q"}], "datasource": "default"}
    module.convert(data, output, prompt_key="question", plain_prompts=True)
    assert json.loads(output.read_text()) == {"prompt": "q", "datasource": "default"}
    module.convert(data, output, prompt_key="question", label_key="answer", label_format="gsm8k")
    assert json.loads(output.read_text())["label"] == "42"
    data[0]["datasource"] = "original"
    module.convert(data, output, prompt_key="question")
    assert json.loads(output.read_text())["datasource"] == "original"
    module.convert(data, output, prompt_key="question", datasource="math500")
    assert json.loads(output.read_text())["datasource"] == "math500"
    data[0]["datasource"] = None
    module.convert(data, output, prompt_key="question")
    assert json.loads(output.read_text())["datasource"] == "default"
    for cases in ({"inputs": ["1"], "outputs": ["2"]}, '{"inputs": ["1"], "outputs": ["2"]}'):
        module.convert(
            [{
                "question": "code",
                "tests": cases
            }],
            output,
            prompt_key="question",
            label_key="tests",
            label_format="sandbox_fusion",
            datasource="livecodebench",
        )
        record = json.loads(output.read_text())
        assert record["datasource"] == "livecodebench"
        assert json.loads(record["label"]) == {
            "backend": "sandbox_fusion",
            "test_cases": {
                "inputs": ["1"],
                "outputs": ["2"]
            },
        }
