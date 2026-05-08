import re


def redact_email(text: str) -> str:
    return re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED_EMAIL]",
        str(text),
    )


def redact_phone(text: str) -> str:
    return re.sub(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[REDACTED_PHONE]",
        str(text),
    )


def redact_ssn(text: str) -> str:
    return re.sub(
        r"\b\d{3}-\d{2}-\d{4}\b",
        "[REDACTED_SSN]",
        str(text),
    )


def redact_member_id(text: str) -> str:
    return re.sub(
        r"\b(?:member|bene|beneficiary|patient)[-_ ]?id[:\s#]*[A-Za-z0-9_-]+\b",
        "[REDACTED_MEMBER_ID]",
        str(text),
        flags=re.IGNORECASE,
    )


def redact_sensitive_text(text: str) -> str:
    redacted = str(text)
    redacted = redact_email(redacted)
    redacted = redact_phone(redacted)
    redacted = redact_ssn(redacted)
    redacted = redact_member_id(redacted)

    return redacted


def contains_sensitive_pattern(text: str) -> bool:
    original_text = str(text)
    redacted_text = redact_sensitive_text(original_text)

    return original_text != redacted_text


if __name__ == "__main__":
    samples = [
        "What is the out-of-pocket maximum for member ID ABC123?",
        "Call 210-555-1234 about this claim.",
        "Send this to test.user@email.com.",
        "The SSN is 123-45-6789.",
        "What is the metal level for this plan?",
    ]

    for sample in samples:
        print("Original:", sample)
        print("Redacted:", redact_sensitive_text(sample))
        print("Sensitive:", contains_sensitive_pattern(sample))
        print("---")