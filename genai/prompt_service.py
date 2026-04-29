import os

from dotenv import load_dotenv
from genai.prompts import SYSTEM_PROMPT, build_user_prompt


load_dotenv()


class MedicalPromptService:
    """Prompt-only GenAI report generator using LangChain and OpenRouter."""

    def explain(self, diagnosis: str, confidence: float, probabilities: dict[str, float]) -> dict:
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(diagnosis, confidence, probabilities)
        llm_answer = self._try_langchain_answer(system_prompt, user_prompt)

        if llm_answer:
            explanation = llm_answer
            mode = "langchain_prompt"
        else:
            explanation = self._local_prompt_fallback(diagnosis, confidence)
            mode = "prompt_fallback"

        return {
            "mode": mode,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "explanation": explanation,
        }

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _user_prompt(
        self,
        diagnosis: str,
        confidence: float,
        probabilities: dict[str, float],
    ) -> str:
        return build_user_prompt(diagnosis, confidence, probabilities)

    def _try_langchain_answer(self, system_prompt: str, user_prompt: str) -> str | None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{user_prompt}"),
            ]
        )
        model = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "inclusionai/ling-2.6-1t:free"),
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0.2,
        )
        chain = prompt | model
        response = chain.invoke({"user_prompt": user_prompt})
        return response.content

    def _local_prompt_fallback(self, diagnosis: str, confidence: float) -> str:
        if diagnosis.lower() == "normal":
            result_line = "The uploaded image looks closest to the normal training class."
        else:
            result_line = f"The uploaded image looks closest to the {diagnosis} training class."

        return (
            f"AI result: {result_line} The model confidence is {confidence:.2%}.\n\n"
            "This is not a final medical diagnosis because the model only sees the uploaded "
            "image and does not know symptoms, history, lab reports, scan quality, or a "
            "radiologist's interpretation.\n\n"
            "Recommended next steps: share the scan and AI result with a qualified doctor "
            "or radiologist, especially if there is pain, fever, blood in urine, urinary "
            "difficulty, weight loss, or worsening symptoms.\n\n"
            "Questions to ask a doctor: What does the scan show? Is follow-up imaging "
            "needed? Are tests or specialist review required? Are there urgent warning signs?\n\n"
            "Safety disclaimer: This tool is for educational decision support only and "
            "should not replace professional medical care."
        )
