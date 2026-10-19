import asyncio
import math
from types import SimpleNamespace as NS

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from openrlhf.algorithms.bifocalrl import rollout
from openrlhf.algorithms.bifocalrl.runtime import DEFAULTS


def test_genprm_http_scores_yes_even_when_generated_judgement_is_no(monkeypatch):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode())

        def apply_chat_template(self, messages, **kwargs):
            assert "ground truth" not in str(messages)
            return "judge:"

    tokenizer = Tokenizer()
    monkeypatch.setattr(rollout, "tokenizer_for", lambda path: tokenizer)

    async def run():
        forced = []

        async def generate(request):
            payload = await request.json()
            if payload["sampling_params"]["max_new_tokens"]:
                return web.json_response({"text": "No}$\nCompletion **Judgement**: $\\boxed{No}$"})
            yes = tokenizer.encode("Yes")
            assert payload["input_ids"][-len(yes) :] == yes
            assert payload["input_ids"][-len(yes) - 1] == ord("{")
            forced.append(payload)
            rows = [[math.log(0.2) / len(yes), token, None] for token in yes]
            return web.json_response({"meta_info": {"input_token_logprobs": rows}})

        app = web.Application()
        app.router.add_post("/generate", generate)
        async with TestServer(app) as server:
            cfg = {**DEFAULTS, "method": "prism", "prm_url": str(server.make_url("/"))}
            sample = NS(prompt="question", response="solution", metadata={})
            reward = await rollout.score_prism(cfg, sample)
        assert len(forced) == 2
        assert reward == pytest.approx(0.2)
        assert sample.metadata["prism_step_probabilities"] == pytest.approx([0.2])

    asyncio.run(run())
