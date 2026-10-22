"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保ったまま続けて質問する（宿題・実装）

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）

会話履歴について:
    LLM の API 自体は前の会話を覚えていない（ステートレス）。
    そこで、これまでのやり取り（ユーザーの発言とモデルの返答）をリストに貯め、
    毎回そのリスト全体を contents として送る。これで「続けて質問」ができる。
    そのぶん、ターンを重ねるほど入力トークン数が増えていく。
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# 会話モードを終了する入力
EXIT_WORDS = {"exit", "quit", "q", "終了"}


def format_usage(response) -> str:
    """response.usage_metadata から入力・出力トークン数を文字列にする（第1回 演習）。"""
    usage = response.usage_metadata
    if usage is None:
        return "[tokens] 取得できませんでした"
    prompt = usage.prompt_token_count or 0
    output = usage.candidates_token_count or 0
    return f"[tokens] 入力 {prompt} / 出力 {output}"


def ask(question: str, client=None) -> str:
    """1回だけ質問する（会話履歴なし）。"""
    client = client or genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    print(format_usage(response))
    return response.text


class Conversation:
    """会話履歴を保持して、続けて質問できるようにする（宿題・実装レベル）。"""

    def __init__(self, client=None):
        self.client = client or genai.Client(api_key=get_api_key())
        self.history: list[types.Content] = []

    def send(self, question: str) -> str:
        user_content = types.Content(role="user", parts=[types.Part.from_text(text=question)])
        response = self.client.models.generate_content(
            model=MODEL,
            contents=[*self.history, user_content],  # これまでの履歴 + 今回の質問
        )
        # API 呼び出しが成功したときだけ履歴に追加する（失敗したターンは残さない）
        self.history.append(user_content)
        self.history.append(
            types.Content(role="model", parts=[types.Part.from_text(text=response.text or "")])
        )
        print(format_usage(response))
        return response.text

    @property
    def turns(self) -> int:
        return len(self.history) // 2


def chat_loop(conversation: Conversation, input_fn=input) -> None:
    """標準入力から質問を読み、会話を続ける。exit / quit / 空行 / Ctrl-D で終了。"""
    print("会話モードです。終了するには exit と入力してください。\n")
    while True:
        try:
            question = input_fn(f"[{conversation.turns + 1}] あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in EXIT_WORDS:
            break
        try:
            answer = conversation.send(question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            if e.code in (400, 401, 403, 404):
                # キーやモデル名の問題は続けても直らないので終了する
                raise SystemExit(1)
            continue  # 429 / 503 などは少し待てば直るので、会話は続ける
        print(f"モデル> {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--chat", action="store_true", help="会話履歴を保ったまま続けて質問する"
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        chat_loop(Conversation())
        return

    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
