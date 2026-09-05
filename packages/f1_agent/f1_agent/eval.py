import os
import uuid

import mlflow
from langgraph.checkpoint.memory import InMemorySaver
from mlflow.genai import scorer
from mlflow.genai.scorers import ToolCallCorrectness, ToolCallEfficiency

from f1_agent.agent import _enable_agent_tracking, build_agent

EVAL_CASES = [
    {
        "inputs": {"question": "Who will win the next race?"},
        "expectations": {"route": "predictor", "tool": "predict_next_race"},
        "tags": {"intent": "win"},
    },
    {
        "inputs": {"question": "Who is on pole?"},
        "expectations": {"route": "predictor", "tool": "predict_next_qualifying"},
        "tags": {"intent": "pole"},
    },
    {
        "inputs": {"question": "When and where is the next race?"},
        "expectations": {"route": "info", "tool": "get_next_race_info"},
        "tags": {"intent": "schedule"},
    },
    {
        "inputs": {
            "question": "Why?",
            "prior": "Who will win the next race?",
        },
        "expectations": {
            "route": "predictor",
            "tool": "predict_next_race",
            "followup": True,
        },
        "tags": {"intent": "why"},
    },
]


def _ensure_dataset():
    try:
        ds = mlflow.genai.create_dataset(name="f1-agent-behavior")
    except Exception:
        ds = mlflow.genai.get_dataset(name="f1-agent-behavior")
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
