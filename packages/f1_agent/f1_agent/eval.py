import os

import mlflow
from mlflow.genai import scorer
from mlflow.genai.scorers import ToolCallCorrectness, ToolCallEfficiency

from f1_agent.agent import _enable_agent_tracking, build_agent

DATASET_NAME = "f1-agent-behavior"

# One raw case per question, single-worker or compound alike — "compound" is
# just whichever cases name more than one tool, not a separate category.
# A case can optionally carry:
#   - "followup": a second question asked right after, in the same thread,
#     that should still be answered from context (e.g. "Why?").
#   - "needs_reasoning": the answer must show its work (features/why), not
#     just a bare name — implied automatically for a "followup".
CASES_RAW = [
    # -- schedule only (info worker) --
    {
        "question": "When and where is the next race?",
        "tools": ["get_next_race_info"],
        "intent": "schedule",
    },
    {
        "question": "When and where is the next Grand Prix?",
        "tools": ["get_next_race_info"],
        "intent": "schedule",
    },
    {
        "question": "Which circuit will the next race be on?",
        "tools": ["get_next_race_info"],
        "intent": "schedule",
    },
    {"question": "What is the next race?", "tools": ["get_next_race_info"], "intent": "schedule"},
    # -- qualifying only (predictor worker) --
    {"question": "Who is on pole?", "tools": ["predict_next_qualifying"], "intent": "pole"},
    {"question": "Who's on pole?", "tools": ["predict_next_qualifying"], "intent": "pole"},
    {
        "question": "Who do you think will win qualifying?",
        "tools": ["predict_next_qualifying"],
        "intent": "pole",
    },
    {
        "question": "Who's on the pole position?",
        "tools": ["predict_next_qualifying"],
        "intent": "pole",
    },
    {
        "question": "What will the result of qualifying be?",
        "tools": ["predict_next_qualifying"],
        "intent": "pole",
    },
    # -- race winner only (predictor worker), each with a same-thread "why" follow-up --
    {
        "question": "Who will win the next race?",
        "tools": ["predict_next_race"],
        "intent": "win",
        "followup": "Why?",
    },
    {
        "question": "Who wins the next Grand Prix?",
        "tools": ["predict_next_race"],
        "intent": "win",
        "followup": "Why is this the case?",
    },
    {
        "question": "Predict the winner of the upcoming race",
        "tools": ["predict_next_race"],
        "intent": "win",
        "followup": "What are you basing your prediction on?",
    },
    {
        "question": "Who do you think takes the next race?",
        "tools": ["predict_next_race"],
        "intent": "win",
        "followup": "Why do you think this will happen?",
    },
    {
        "question": "What's your pick for the next race winner?",
        "tools": ["predict_next_race"],
        "intent": "win",
        "followup": "Is there a reason for this?",
    },
    # -- compound: multiple workers chained in a single turn --
    {
        "question": "What is the next race and who wins it?",
        "tools": ["get_next_race_info", "predict_next_race"],
        "intent": "compound_win",
    },
    {
        "question": "When and where is the next race, and who's on pole?",
        "tools": ["get_next_race_info", "predict_next_qualifying"],
        "intent": "compound_pole",
    },
    {
        "question": "Where's the next race and what's the predicted podium?",
        "tools": ["get_next_race_info", "predict_next_race"],
        "intent": "compound_podium",
    },
    {
        "question": "What is the next race? Who wins qualifying and who wins the race? Why?",
        "tools": ["get_next_race_info", "predict_next_qualifying", "predict_next_race"],
        "intent": "compound_full",
        "needs_reasoning": True,
    },
]


def _build_cases() -> list[dict]:
    cases = []
    for raw in CASES_RAW:
        tools = raw["tools"]
        cases.append(
            {
                "inputs": {"question": raw["question"]},
                "expectations": {
                    "tools": tools,
                    "compound": len(tools) > 1,
                    "needs_reasoning": raw.get("needs_reasoning", False),
                },
                "tags": {"intent": raw["intent"]},
            }
        )
        if followup := raw.get("followup"):
            cases.append(
                {
                    "inputs": {"question": followup, "prior": raw["question"]},
                    "expectations": {"tools": tools, "followup": True},
                    "tags": {"intent": "why", "pair": raw["question"]},
                }
            )
    return cases


EVAL_CASES = _build_cases()


def _ensure_dataset():
    existing = mlflow.genai.search_datasets(
        filter_string=f"name = '{DATASET_NAME}'",
        order_by=["created_time DESC"],
        max_results=1,
    )
    ds = existing[0] if existing else mlflow.genai.create_dataset(name=DATASET_NAME)
    ds.merge_records(EVAL_CASES)
    return ds


def _last_ai_text(result: dict) -> str:
    content = getattr(result["messages"][-1], "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else str(b.get("text", "")) for b in content)
    return str(content or "")


def _make_predict_fn():
    graph = build_agent()

    def predict_fn(question: str, prior: str | None = None) -> str:
        messages = []
        if prior:
            # Run the prior turn for real, then hand the *scored* call the
            # whole exchange as explicit input messages. An LLM judge only
            # ever inspects the trace of the call it's scoring — a follow-up's
            # context has to live in that call's own input, not in graph
            # state a separate prior invoke() would leave behind but the
            # judge can't see into.
            first = graph.invoke({"messages": [{"role": "user", "content": prior}]})
            messages.append({"role": "user", "content": prior})
            messages.append({"role": "assistant", "content": _last_ai_text(first)})
        messages.append({"role": "user", "content": question})

        result = graph.invoke({"messages": messages})
        return _last_ai_text(result)

    return predict_fn


@scorer
def used_expected_tools(trace, expectations) -> bool:
    """Every expected tool was called, and exactly once each — covers both
    single-worker cases (one expected tool) and compound cases (several)."""
    expected = expectations["tools"]
    names = [s.name for s in trace.search_spans(span_type="TOOL")]
    if not names:
        names = [s.name for s in trace.search_spans() if any(t in (s.name or "") for t in expected)]
    return all(names.count(t) == 1 for t in expected)


@scorer
def no_unexpected_tools(trace, expectations) -> bool:
    """No tool was called outside the expected set — catches a single-intent
    question accidentally triggering the other worker's tool, and a compound
    question accidentally skipping a worker it needed."""
    expected = set(expectations["tools"])
    names = {s.name for s in trace.search_spans(span_type="TOOL")}
    return names <= expected


@scorer
def mentions_tool_winner(outputs: str, expectations) -> bool:
    if "predict_next_race" not in expectations.get("tools", []):
        return True
    from f1_ml.models.race.predict import predict_next_race

    winner = predict_next_race().to_dict(top_n=1)["winner"]["driver_name"]
    return winner.lower() in outputs.lower()


@scorer
def mentions_pole_winner(outputs: str, expectations) -> bool:
    if "predict_next_qualifying" not in expectations.get("tools", []):
        return True
    from f1_ml.models.qualifying.predict import predict_next_qualifying

    pole = predict_next_qualifying().to_dict(top_n=1)["pole"]["driver_name"]
    return pole.lower() in outputs.lower()


@scorer
def mentions_race_name(outputs: str, expectations) -> bool:
    """For any case that hits the info tool, the merged/polished answer must
    still carry the race name — guards against the editor step dropping the
    schedule half of a compound answer."""
    if "get_next_race_info" not in expectations.get("tools", []):
        return True
    from f1_ml.inference.next_race import resolve_target_race

    race_name = resolve_target_race().race_name
    return race_name.lower() in outputs.lower()


@scorer
def short_win_answer(outputs: str, expectations) -> bool:
    if expectations.get("followup") or expectations.get("compound"):
        return True
    if expectations.get("tools") != ["predict_next_race"]:
        return True
    return len(outputs.split()) <= 40


@scorer
def why_uses_features(outputs: str, expectations) -> bool:
    if not (expectations.get("followup") or expectations.get("needs_reasoning")):
        return True
    text = outputs.lower()
    hints = ("feature", "qualifying", "median", "constructor", "last")
    return any(h in text for h in hints)


def run_eval(*, register_dataset: bool = True, llm_judges: bool = False):
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
    _enable_agent_tracking()
    mlflow.set_experiment("f1-agent")

    data = _ensure_dataset() if register_dataset else EVAL_CASES
    scorers = [
        used_expected_tools,
        no_unexpected_tools,
        mentions_tool_winner,
        mentions_pole_winner,
        mentions_race_name,
        short_win_answer,
        why_uses_features,
    ]
    if llm_judges:
        scorers.extend([ToolCallCorrectness(), ToolCallEfficiency()])
    return mlflow.genai.evaluate(
        data=data,
        predict_fn=_make_predict_fn(),
        scorers=scorers,
    )


if __name__ == "__main__":
    run_eval()
