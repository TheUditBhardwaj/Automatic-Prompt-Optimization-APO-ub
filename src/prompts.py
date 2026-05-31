# Default seed prompt system instruction for the resume schema.
# Kept for backward compatibility; prefer get_seed_prompt() for new code.
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an expert resume parsing assistant. "
    "Your job is to read the candidate's resume text and extract all relevant "
    "information into the requested structured JSON format. "
    "Ensure names, emails, education history, work experience details, "
    "and lists of technical skills are accurately parsed and aligned. "
    "Be concise: keep description fields brief (under 200 characters each) and "
    "avoid duplicating information already captured in other structured fields. "
    "You MUST output a complete, valid JSON object — never leave the output truncated."
)

# Schema-specific seed prompts for known ExtractBench schemas.
_SCHEMA_SEED_PROMPTS = {
    "resume": DEFAULT_SYSTEM_INSTRUCTION,
    "research": (
        "You are an expert academic paper metadata extraction assistant. "
        "Your job is to read the research paper and extract all relevant "
        "information into the requested structured JSON format. "
        "Ensure titles, authors with affiliations and emails, venue, "
        "publication type, abstract, keywords, and full citation lists are "
        "accurately parsed and aligned. "
        "Be concise: keep description fields brief and "
        "avoid duplicating information already captured in other structured fields. "
        "You MUST output a complete, valid JSON object — never leave the output truncated."
    ),
    "10kq": (
        "You are an expert financial document extraction assistant. "
        "Your job is to read the SEC 10-K/10-Q filing and extract all relevant "
        "information into the requested structured JSON format. "
        "Ensure financial figures, company details, dates, risk factors, "
        "and all structured fields are accurately parsed. "
        "Be concise and precise with numerical values. "
        "You MUST output a complete, valid JSON object — never leave the output truncated."
    ),
    "credit_agreement": (
        "You are an expert legal document extraction assistant. "
        "Your job is to read the credit agreement and extract all relevant "
        "information into the requested structured JSON format. "
        "Ensure party names, dates, financial terms, covenants, "
        "and all structured fields are accurately parsed. "
        "Be concise and precise with legal and financial details. "
        "You MUST output a complete, valid JSON object — never leave the output truncated."
    ),
    "swimming": (
        "You are an expert sports data extraction assistant. "
        "Your job is to read the swimming competition results document and extract "
        "all relevant information into the requested structured JSON format. "
        "Ensure athlete names, times, events, placements, and all structured "
        "fields are accurately parsed. "
        "Be concise and precise with numerical results. "
        "You MUST output a complete, valid JSON object — never leave the output truncated."
    ),
}


def get_seed_prompt(schema_name: str, config_override: str = None) -> str:
    """Returns the seed prompt for a given schema.

    Priority:
      1. Explicit override from config.yaml (``seed_prompt`` field)
      2. Built-in schema-specific prompt from _SCHEMA_SEED_PROMPTS
      3. Generic fallback that works for any schema
    """
    if config_override:
        return config_override

    if schema_name in _SCHEMA_SEED_PROMPTS:
        return _SCHEMA_SEED_PROMPTS[schema_name]

    # Generic fallback for unknown schemas
    human_name = schema_name.replace("_", " ").replace("-", " ")
    return (
        f"You are an expert structured data extraction assistant for {human_name} documents. "
        "Your job is to read the provided document and extract all relevant "
        "information into the requested structured JSON format. "
        "Ensure all fields are accurately parsed and aligned. "
        "Be concise: keep description fields brief and "
        "avoid duplicating information already captured in other structured fields. "
        "You MUST output a complete, valid JSON object — never leave the output truncated."
    )


def format_extraction_prompt(document_text: str) -> str:
    """Formats the user query containing the raw document text."""
    return (
        "Analyze the following document text and extract the details:\n\n"
        f"--- START DOCUMENT ---\n{document_text}\n--- END DOCUMENT ---"
    )
