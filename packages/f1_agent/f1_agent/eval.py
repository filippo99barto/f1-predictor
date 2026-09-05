import os
import uuid

import mlflow
from langgraph.checkpoint.memory import InMemorySaver
from mlflow.genai import scorer
from mlflow.genai.scorers import ToolCallCorrectness, ToolCallEfficiency

from f1_agent.agent import _enable_agent_tracking, build_agent

DATASET_NAME = "f1-agent-behavior"

POLE_PHRASES = [
    "Who is on pole?",
    "Who's on pole?",
    "Who do you think will win qualifying?",
    "Who's on the pole position?",
    "What will the result of qualifying be?",
]

SCHEDULE_PHRASES = [
    "When and where is the next race?",
    "When and where is the next Grand Prix?",
    "Which circuit will the next race be on?",
    "What is the next race?",
]

WIN_WHY_PAIRS = [
    ("Who will win the next race?", "Why?"),
    ("Who wins the next Grand Prix?", "Why is this the case?"),
    ("Predict the winner of the upcoming race", "What are you basing your prediction on?"),
    ("Who do you think takes the next race?", "Why do you think this will happen?"),
    ("What's your pick for the next race winner?", "Is there a reason for this?"),
]


def _win_why_cases() -> list[dict]:
    cases = []
    for first, followup in WIN_WHY_PAIRS:
        cases.append(
            {
                "inputs": {"question": first},
                "expectations": {"route": "predictor", "tool": "predict_next_race"},
                "tags": {"intent": "win", "pair": first},
            }
        )
        cases.append(
            {
                "inputs": {"question": followup, "prior": first},
                "expectations": {
                    "route": "predictor",
                    "tool": "predict_next_race",
                    "followup": True,
                },
                "tags": {"intent": "why", "pair": first},
            }
        )
    return cases


EVAL_CASES = (
    _win_why_cases()
    + [
        {
            "inputs": {"question": pole_phrase},
            "expectations": {"route": "predictor", "tool": "predict_next_qualifying"},
            "tags": {"intent": "pole"},
        }
        for pole_phrase in POLE_PHRASES
    ]
    + [
        {
            "inputs": {"question": schedule_phrase},
            "expectations": {"route": "info", "tool": "get_next_race_info"},
            "tags": {"intent": "schedule"},
        }
        for schedule_phrase in SCHEDULE_PHRASES
    ]
)


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
    graph = build_agent(checkpointer=InMemorySaver())

    def predict_fn(question: str, prior: str | None = None) -> str:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        if prior:
            graph.invoke(
                {"messages": [{"role": "user", "content": prior}]},
                config=config,
            )
        result = graph.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )
        return _last_ai_text(result)

    return predict_fn


@scorer
def used_expected_tool(trace, expectations) -> bool:
    expected = expectations["tool"]
    names = [s.name for s in trace.search_spans(span_type="TOOL")]
    if not names:
        names = [s.name for s in trace.search_spans() if expected in (s.name or "")]
    return expected in names and names.count(expected) == 1


@scorer
def no_extra_prediction_tool(trace, expectations) -> bool:
    names = {s.name for s in trace.search_spans(span_type="TOOL")}
    if expectations["tool"] == "get_next_race_info":
        return "predict_next_race" not in names and "predict_next_qualifying" not in names
    if expectations["tool"] == "predict_next_race":
        return "predict_next_qualifying" not in names
    return True


@scorer
def mentions_tool_winner(outputs: str, expectations) -> bool:
    if expectations.get("tool") != "predict_next_race":
        return True
    from f1_ml.models.race.predict import predict_next_race

    winner = predict_next_race().to_dict(top_n=1)["winner"]["driver_name"]
    return winner.lower() in outputs.lower()


@scorer
def short_win_answer(outputs: str, expectations) -> bool:
    if expectations.get("followup") or expectations.get("tool") != "predict_next_race":
        return True
    return len(outputs.split()) <= 40


@scorer
def why_uses_features(outputs: str, expectations) -> bool:
    if not expectations.get("followup"):
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
        used_expected_tool,
        no_extra_prediction_tool,
        mentions_tool_winner,
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
