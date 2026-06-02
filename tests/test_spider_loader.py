"""Spider loader smoke tests.

The actual HF fetch is heavy and brittle in CI, so the import-time tests run
on the in-process helper. The full network test is opt-in via SPIDER_NETWORK=1.
"""
import os

import pytest

from u_jepa.data.spider import _build_prompt, load_spider_pairs


def test_prompt_template_includes_question_and_sql_anchor():
    prompt = _build_prompt("how many singers do we have?")
    assert "how many singers do we have?" in prompt
    assert prompt.rstrip().endswith("SQL:")


@pytest.mark.skipif(
    os.environ.get("SPIDER_NETWORK") != "1",
    reason="network test, opt in with SPIDER_NETWORK=1",
)
def test_load_returns_paired_view_dicts():
    items = load_spider_pairs(split="validation", n=4)
    assert len(items) > 0
    for ex in items:
        assert set(ex.keys()) >= {"prompt", "target", "view_a", "view_b"}
        assert ex["prompt"].endswith("SQL:")
        assert ex["target"] == ex["view_b"]
