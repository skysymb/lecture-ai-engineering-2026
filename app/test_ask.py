"""app/ask.py のテスト（LLMもAPIキーも使わない。偽のクライアントで代用する）。"""

from types import SimpleNamespace

from google.genai import errors

from app import ask


class FakeModels:
    """generate_content に渡された contents を記録し、決まった返事を返す。"""

    def __init__(self, replies, fail_on=None):
        self.replies = iter(replies)
        self.calls = []
        self.fail_on = fail_on  # この回数目の呼び出しで APIError を投げる

    def generate_content(self, model, contents):
        self.calls.append(contents)
        if self.fail_on == len(self.calls):
            raise errors.APIError(429, {"error": {"message": "rate limit"}})
        usage = SimpleNamespace(prompt_token_count=10 * len(self.calls), candidates_token_count=5)
        return SimpleNamespace(text=next(self.replies), usage_metadata=usage)


def fake_client(replies, fail_on=None):
    return SimpleNamespace(models=FakeModels(replies, fail_on))


def texts(contents):
    return [(c.role, c.parts[0].text) for c in contents]


def test_format_usage():
    r = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34))
    assert ask.format_usage(r) == "[tokens] 入力 12 / 出力 34"
    assert "取得できません" in ask.format_usage(SimpleNamespace(usage_metadata=None))


def test_ask_prints_tokens(capsys):
    assert ask.ask("こんにちは", client=fake_client(["やあ"])) == "やあ"
    assert "[tokens] 入力 10 / 出力 5" in capsys.readouterr().out


def test_conversation_sends_history():
    client = fake_client(["私はアシスタントです", "さっきは自己紹介をしました"])
    conv = ask.Conversation(client=client)
    conv.send("あなたは誰？")
    conv.send("さっき何て言った？")

    second_call = client.models.calls[1]
    assert texts(second_call) == [
        ("user", "あなたは誰？"),
        ("model", "私はアシスタントです"),
        ("user", "さっき何て言った？"),
    ]
    assert conv.turns == 2


def test_failed_turn_is_not_kept_in_history(capsys):
    client = fake_client(["一つ目の答え", "三つ目の答え"], fail_on=2)
    inputs = iter(["一つ目", "二つ目", "三つ目", "exit"])
    conv = ask.Conversation(client=client)
    ask.chat_loop(conv, input_fn=lambda _: next(inputs))

    assert "レート制限" in capsys.readouterr().out
    assert texts(conv.history) == [
        ("user", "一つ目"),
        ("model", "一つ目の答え"),
        ("user", "三つ目"),
        ("model", "三つ目の答え"),
    ]


def test_chat_loop_ends_on_eof():
    conv = ask.Conversation(client=fake_client([]))

    def eof(_):
        raise EOFError

    ask.chat_loop(conv, input_fn=eof)
    assert conv.turns == 0
