"""format_chat_prompt: chat template when available, graceful fallback otherwise."""
from u_jepa.util.prompting import format_chat_prompt


class _NoTemplateTokenizer:
    """Stub without apply_chat_template, like the other test fakes."""
    pass


class _ThinkingTokenizer:
    """Stub whose chat template accepts enable_thinking."""
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=True):
        content = messages[0]["content"]
        think = "" if not enable_thinking else "<think>"
        gen = "<assistant>" if add_generation_prompt else ""
        return f"<user>{content}</user>{gen}{think}"


class _OldTokenizer:
    """Stub whose chat template predates the enable_thinking kwarg."""
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        content = messages[0]["content"]
        gen = "<assistant>" if add_generation_prompt else ""
        return f"<user>{content}</user>{gen}"


def test_no_template_returns_raw_and_false():
    text, used = format_chat_prompt(_NoTemplateTokenizer(), "hello")
    assert text == "hello"
    assert used is False


def test_thinking_tokenizer_disables_thinking():
    text, used = format_chat_prompt(_ThinkingTokenizer(), "Q?")
    assert used is True
    assert "<think>" not in text          # enable_thinking=False honored
    assert "<assistant>" in text          # generation prompt added
    assert "Q?" in text


def test_old_tokenizer_without_enable_thinking_still_works():
    text, used = format_chat_prompt(_OldTokenizer(), "Q?")
    assert used is True
    assert "<assistant>" in text
    assert "Q?" in text


def test_broken_template_falls_back_to_raw():
    class _Broken:
        def apply_chat_template(self, *a, **k):
            raise RuntimeError("boom")
    text, used = format_chat_prompt(_Broken(), "hello")
    assert text == "hello"
    assert used is False
