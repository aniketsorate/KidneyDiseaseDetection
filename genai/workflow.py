from typing import TypedDict

from genai.prompt_service import MedicalPromptService


class ReportState(TypedDict, total=False):
    diagnosis: str
    confidence: float
    probabilities: dict[str, float]
    confidence_level: str
    prompt_report: dict
    final_report: dict


def classify_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


class KidneyReportWorkflow:
    """LangGraph report workflow with a direct Python fallback."""

    def __init__(self) -> None:
        self.prompt_service = MedicalPromptService()
        self.graph = self._build_langgraph()

    def run(self, prediction: dict) -> dict:
        state: ReportState = {
            "diagnosis": prediction["diagnosis"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
        }

        if self.graph:
            result = self.graph.invoke(state)
        else:
            result = self._run_fallback(state)

        return result["final_report"]

    def _build_langgraph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None

        graph = StateGraph(ReportState)
        graph.add_node("assess_confidence", self._assess_confidence)
        graph.add_node("generate_explanation", self._generate_explanation)
        graph.add_node("build_report", self._build_report)
        graph.set_entry_point("assess_confidence")
        graph.add_edge("assess_confidence", "generate_explanation")
        graph.add_edge("generate_explanation", "build_report")
        graph.add_edge("build_report", END)
        return graph.compile()

    def _run_fallback(self, state: ReportState) -> ReportState:
        state = self._assess_confidence(state)
        state = self._generate_explanation(state)
        return self._build_report(state)

    def _assess_confidence(self, state: ReportState) -> ReportState:
        state["confidence_level"] = classify_confidence(state["confidence"])
        return state

    def _generate_explanation(self, state: ReportState) -> ReportState:
        state["prompt_report"] = self.prompt_service.explain(
            diagnosis=state["diagnosis"],
            confidence=state["confidence"],
            probabilities=state["probabilities"],
        )
        return state

    def _build_report(self, state: ReportState) -> ReportState:
        diagnosis = state["diagnosis"]
        confidence_level = state["confidence_level"]
        caution = (
            "Confidence is low, so the image should be reviewed manually and a clearer scan may be needed."
            if confidence_level == "low"
            else "Use this as decision support only; final interpretation should come from a clinician."
        )

        state["final_report"] = {
            "diagnosis": diagnosis,
            "confidence": state["confidence"],
            "confidence_level": confidence_level,
            "probabilities": state["probabilities"],
            "ai_explanation": state["prompt_report"]["explanation"],
            "system_prompt": state["prompt_report"]["system_prompt"],
            "user_prompt": state["prompt_report"]["user_prompt"],
            "generation_mode": state["prompt_report"]["mode"],
            "safety_note": caution,
        }
        return state
