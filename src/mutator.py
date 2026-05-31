import logging
from typing import List, Dict, Any
from llm_wrapper import GeminiClient
from evaluator import flatten_dict, filter_meaningful_fields

logger = logging.getLogger(__name__)

MUTATOR_SYSTEM_INSTRUCTION = (
    "You are an expert meta-prompt engineer. "
    "Your goal is to optimize a system instruction for an LLM that extracts structured resume data from PDFs. "
    "You will be given the current system instruction and a list of specific errors/mismatches observed "
    "in the last run. "
    "Produce a revised, updated system instruction that directly addresses these failures. "
    "The new instruction should guide the extractor LLM to avoid these mistakes. "
    "Maintain the general quality of other extractions. "
    "Do NOT output any intro, explanations, or code blocks. Output ONLY the updated system instruction text."
)

def generate_error_feedback(results: List[Dict[str, Any]]) -> str:
    """Analyzes prediction vs gold results to compile a list of mismatches."""
    feedback_lines = []
    error_count = 0
    max_errors = 10

    for res in results:
        score = res.get("score", 1.0)
        if score >= 0.98:
            continue

        name = res.get("name", "Unknown")
        gold = res.get("gold", {})
        pred = res.get("prediction", {})

        gold_flat = filter_meaningful_fields(flatten_dict(gold))
        pred_flat = filter_meaningful_fields(flatten_dict(pred))

        item_errors = []
        for k, g_val in gold_flat.items():
            if k not in pred_flat:
                item_errors.append(f"Missing field '{k}' (Expected: '{g_val}')")
            else:
                p_val = pred_flat[k]
                if isinstance(g_val, str) and isinstance(p_val, str):
                    if g_val.strip().lower() != p_val.strip().lower():
                        item_errors.append(f"Mismatched field '{k}' (Expected: '{g_val}', Got: '{p_val}')")
                elif g_val != p_val:
                    item_errors.append(f"Mismatched field '{k}' (Expected: '{g_val}', Got: '{p_val}')")

        for k, p_val in pred_flat.items():
            if k not in gold_flat:
                item_errors.append(f"Extra field '{k}' (Extracted: '{p_val}' but not present in gold)")

        if item_errors:
            feedback_lines.append(f"Resume: {name}")
            for err in item_errors[:3]:
                if error_count < max_errors:
                    feedback_lines.append(f"  - {err}")
                    error_count += 1
            if len(item_errors) > 3:
                feedback_lines.append("  - ... and others")

        if error_count >= max_errors:
            break

    if not feedback_lines:
        return "No errors observed. All extractions matched gold annotations perfectly."

    return "\n".join(feedback_lines)


def mutate_prompt(
    client: GeminiClient,
    current_instruction: str,
    results: List[Dict[str, Any]],
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.7
) -> str:
    """Calls Gemini to mutate the system instruction using the error feedback."""
    error_feedback = generate_error_feedback(results)
    
    logger.info("Feedback prepared for mutator:\n" + error_feedback)

    user_query = (
        f"Here is the current system instruction:\n"
        f"\"\"\"\n{current_instruction}\n\"\"\"\n\n"
        f"Here are the specific errors observed in the latest run:\n"
        f"{error_feedback}\n\n"
        f"Please provide an updated system instruction. "
        f"Remember, do not include any markdown format (e.g. ```), introduction, or conversational filler. "
        f"Output only the raw prompt text."
    )

    try:
        mutated_text, _, _, _ = client.mutate_prompt(
            system_instruction=MUTATOR_SYSTEM_INSTRUCTION,
            user_prompt=user_query,
            model_name=model_name,
            temperature=temperature
        )
        cleaned_text = mutated_text.strip()
        # Clean any markdown code ticks if LLM outputted them despite instructions
        if cleaned_text.startswith("```"):
            lines = cleaned_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()
            
        logger.info("Successfully generated mutated system instruction.")
        return cleaned_text
    except Exception as e:
        logger.error(f"Failed to generate mutated prompt: {e}")
        raise
