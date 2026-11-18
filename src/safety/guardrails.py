"""Medical safety guardrails, policy filters, and emergency redirects."""

import re
from typing import Optional, Tuple
from pydantic import BaseModel
from src.config import setup_logger

logger = setup_logger("safety.guardrails")

# Medical Disclaimer appended to all user-facing answers
STANDARD_DISCLAIMER = (
    "\n\n---\n"
    "*Disclaimer: This assistant provides general information extracted from published drug labels "
    "and clinical guidelines. It is NOT medical advice, diagnosis, or personalized dosing guidance. "
    "If you are experiencing a medical condition or need personalized treatment, please consult a qualified healthcare professional.*"
)

# Emergency redirect message
EMERGENCY_REDIRECT = (
    "**EMERGENCY DETECTED:** If you or someone you know is experiencing a life-threatening medical emergency, "
    "severe chest pain, shortness of breath, sudden numbness or weakness, severe bleeding, or thoughts of self-harm, "
    "please immediately call your local emergency services (e.g., 911 in the US) or visit the nearest emergency room.\n\n"
    "This AI tool cannot assist with acute emergencies."
)

# Diagnosis and personal dosing refusal message
REFUSAL_MESSAGE = (
    "**Safety Policy Notice:** This assistant can provide general factual information from official FDA labels "
    "and clinical guidelines (such as indications, contraindications, and drug interactions), but cannot diagnose conditions "
    "or provide personalized dosing recommendations for individual patients.\n\n"
    "Please consult a licensed physician or pharmacist for clinical evaluation and personalized dosing."
)

# Emergency keyword regex patterns
EMERGENCY_PATTERNS = [
    r"\b(chest\s+pain|heart\s+attack|stroke|trouble\s+breathing|difficulty\s+breathing|can'?t\s+breathe)\b",
    r"\b(suicid|kill\s+myself|end\s+my\s+life|overdose|overdosing|poisoning|severe\s+bleeding)\b",
    r"\b(unconscious|anaphylaxis|throat\s+closing|choking|loss\s+of\s+consciousness)\b",
]

# Personal diagnosis / dosing patterns
UNSAFE_REQUEST_PATTERNS = [
    r"\b(diagnose\s+me|do\s+i\s+have|what\s+disease\s+do\s+i\s+have|my\s+symptoms\s+are)\b",
    r"\b(what\s+dose\s+should\s+i\s+take|how\s+much\s+should\s+i\s+take\s+for\s+my)\b",
    r"\b(can\s+you\s+prescribe|prescribe\s+me|calculate\s+my\s+dose)\b",
]


class SafetyCheckResult(BaseModel):
    """Result of safety pre-check."""
    is_safe: bool
    refusal_reason: Optional[str] = None
    redirect_message: Optional[str] = None


def run_pre_check(query: str) -> SafetyCheckResult:
    """Evaluate user query against safety rules before retrieval."""
    cleaned_query = query.strip().lower()

    # 1. Emergency detection
    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, cleaned_query):
            logger.warning("Emergency query detected matching pattern '%s'", pattern)
            return SafetyCheckResult(
                is_safe=False,
                refusal_reason="emergency",
                redirect_message=EMERGENCY_REDIRECT,
            )

    # 2. Personal diagnosis / dosing request detection
    for pattern in UNSAFE_REQUEST_PATTERNS:
        if re.search(pattern, cleaned_query):
            logger.warning("Personal diagnosis/dosing query detected matching '%s'", pattern)
            return SafetyCheckResult(
                is_safe=False,
                refusal_reason="personal_advice",
                redirect_message=REFUSAL_MESSAGE,
            )

    return SafetyCheckResult(is_safe=True)


def scrub_phi(text: str) -> str:
    """Remove potential Protected Health Information (SSN, emails, phone numbers)."""
    # Scrub SSN: XXX-XX-XXXX
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
    # Scrub phone numbers
    text = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", text)
    # Scrub email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]", text)
    return text


def apply_post_check(answer: str) -> str:
    """Format and apply safety disclaimers to model answers."""
    cleaned = scrub_phi(answer).strip()
    if STANDARD_DISCLAIMER not in cleaned:
        cleaned = f"{cleaned}{STANDARD_DISCLAIMER}"
    return cleaned
