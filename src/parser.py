import json
import logging
import re
from typing import Type
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Max characters allowed for any single JSON string value before flagging hallucination
_MAX_STRING_VALUE_LEN = 800


def _looks_like_hallucination(s: str) -> bool:
    """Returns True if s is a very long string dominated by a repeating short sub-pattern.
    
    Searches for any short repeated pattern anywhere in the string (not just at the
    start), to handle cases like 'Normal prefix: 28' + '01239' * N.
    """
    if len(s) < _MAX_STRING_VALUE_LEN:
        return False
    # Sample from multiple offsets to find a repeating segment anywhere in the string
    check_offsets = [0, len(s) // 4, len(s) // 2]
    for offset in check_offsets:
        tail = s[offset:]
        if len(tail) < 40:
            continue
        for pattern_len in range(2, 21):
            pattern = tail[:pattern_len]
            repeat_count = tail.count(pattern)
            # If the pattern fills 80%+ of the tail, it's a repetition loop
            if repeat_count > (len(tail) // pattern_len) * 0.8:
                return True
    return False


def sanitize_runaway_strings(json_str: str) -> str:
    """Detects and strips hallucinated runaway string values using a character-level scanner.
    
    The Gemini model occasionally generates an extremely long repeating digit/pattern
    sequence inside a string field (e.g. '012390123901239...' repeating for thousands
    of characters), which can be terminated OR unterminated (hits token limit mid-string).
    
    A regex-based approach cannot handle the unterminated case because it requires a
    closing quote. This scanner instead tracks string state character-by-character and
    replaces runaway content once detected — even for strings with no closing quote.
    
    Runaway strings are replaced with a short placeholder so the rest of the JSON
    remains parseable.
    """
    result = []
    i = 0
    n = len(json_str)

    while i < n:
        ch = json_str[i]

        if ch != '"':
            result.append(ch)
            i += 1
            continue

        # Opening quote — collect string content character by character
        result.append('"')
        i += 1
        string_chars = []
        is_runaway = False

        while i < n:
            ch = json_str[i]
            if ch == '\\' and i + 1 < n:
                # Escaped character — consume two chars
                string_chars.append(ch)
                string_chars.append(json_str[i + 1])
                i += 2
                continue
            if ch == '"':
                # Normal end of string
                i += 1  # consume closing quote
                break

            string_chars.append(ch)
            i += 1

            # Check for hallucination periodically once we exceed the length threshold
            # Checking every single character for a 100k string is O(N^2) and hangs the process.
            if len(string_chars) > _MAX_STRING_VALUE_LEN and len(string_chars) % 500 == 0 and not is_runaway:
                # Only check the last 800 characters to keep it O(1) time
                content = ''.join(string_chars[-800:])
                if _looks_like_hallucination(content):
                    is_runaway = True
                    logger.warning(
                        f"Detected hallucinated runaway string value "
                        f"(>{_MAX_STRING_VALUE_LEN} chars with repeating pattern). "
                        "Replacing with placeholder."
                    )
                    # Consume and discard the rest of the string
                    while i < n:
                        if json_str[i] == '\\' and i + 1 < n:
                            i += 2
                        elif json_str[i] == '"':
                            i += 1  # consume closing quote (if present)
                            break
                        else:
                            i += 1
                    break  # stop inner loop

        if is_runaway:
            result.append('[content removed - hallucinated repetition]')
        else:
            result.extend(string_chars)
        result.append('"')

    return ''.join(result)
def attempt_json_recovery(truncated: str) -> str:
    """Attempts to recover a truncated JSON string by closing open brackets/braces.
    
    When an LLM hits max_output_tokens it cuts output mid-JSON, leaving unclosed
    strings, arrays, and objects. This function truncates to the last complete
    value and closes all open containers.
    """
    # Find the last position of a complete primitive or closing bracket
    # Walk backwards to find the last full comma-terminated or bracket-closed token
    s = truncated.rstrip()
    
    # Track bracket stack to know what needs closing
    stack = []
    in_string = False
    escape_next = False
    last_safe_pos = 0  # position after last 'safe' complete value
    last_safe_stack = []

    i = 0
    while i < len(s):
        ch = s[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if ch == '\\' and in_string:
            escape_next = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            i += 1
            continue
        if in_string:
            i += 1
            continue
        # Outside string
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
                last_safe_pos = i + 1
                last_safe_stack = list(stack)
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
                last_safe_pos = i + 1
                last_safe_stack = list(stack)
        elif ch == ',':
            if not stack or (stack[-1] in ('{', '[')):
                last_safe_pos = i  # safe to truncate before the trailing comma
                last_safe_stack = list(stack)
        i += 1

    if not last_safe_stack and last_safe_pos == 0:
        return s

    # Truncate to last safe position and close all open containers
    recovered = s[:last_safe_pos].rstrip().rstrip(',')
    
    # Close all open containers in reverse order using the stack at that point
    for bracket in reversed(last_safe_stack):
        if bracket == '{':
            recovered += '}'
        elif bracket == '[':
            recovered += ']'

    return recovered

def parse_json_to_schema(json_str: str, schema_class: Type[BaseModel]) -> BaseModel:
    """Parses a raw JSON string and validates it against a Pydantic model.
    
    Pipeline:
      1. Strip markdown code fences.
      2. Sanitize hallucinated runaway string values.
      3. Parse JSON; on failure attempt bracket-closing recovery.
    """
    if not json_str:
        raise ValueError("Received empty or None string for JSON parsing.")
        
    cleaned_str = json_str.strip()
    
    # Strip markdown JSON blocks if present
    if cleaned_str.startswith("```json"):
        cleaned_str = cleaned_str[7:]
    elif cleaned_str.startswith("```"):
        cleaned_str = cleaned_str[3:]
        
    if cleaned_str.endswith("```"):
        cleaned_str = cleaned_str[:-3]
        
    cleaned_str = cleaned_str.strip()

    # Sanitize hallucinated runaway strings before attempting to parse
    cleaned_str = sanitize_runaway_strings(cleaned_str)

    try:
        data = json.loads(cleaned_str)
        return schema_class.model_validate(data)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON syntax error ({e}). Attempting recovery on truncated response...")
        # Attempt to recover truncated JSON (e.g. when LLM hits max_output_tokens)
        try:
            recovered_str = attempt_json_recovery(cleaned_str)
            data = json.loads(recovered_str)
            logger.info("JSON recovery succeeded.")
            return schema_class.model_validate(data)
        except (json.JSONDecodeError, Exception) as recovery_err:
            logger.error(f"JSON recovery also failed: {recovery_err}. Raw content:\n{json_str}")
            raise ValueError(f"Invalid JSON response structure: {e}")
    except ValidationError as e:
        logger.error(f"Pydantic validation error: {e}")
        raise ValueError(f"JSON data failed validation against model {schema_class.__name__}: {e}")
