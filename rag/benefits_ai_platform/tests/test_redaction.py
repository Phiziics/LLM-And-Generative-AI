from src.security.redact_pii import redact_sensitive_text, contains_sensitive_pattern


def test_redacts_email():
    text = "Send details to test.user@email.com"
    redacted = redact_sensitive_text(text)

    assert "[REDACTED_EMAIL]" in redacted
    assert "test.user@email.com" not in redacted


def test_redacts_phone():
    text = "Call 210-555-1234"
    redacted = redact_sensitive_text(text)

    assert "[REDACTED_PHONE]" in redacted
    assert "210-555-1234" not in redacted


def test_redacts_ssn():
    text = "SSN is 123-45-6789"
    redacted = redact_sensitive_text(text)

    assert "[REDACTED_SSN]" in redacted
    assert "123-45-6789" not in redacted


def test_detects_sensitive_pattern():
    text = "What is the deductible for member ID ABC123?"

    assert contains_sensitive_pattern(text) is True


def test_non_sensitive_text_returns_false():
    text = "What is the out-of-pocket maximum for this plan?"

    assert contains_sensitive_pattern(text) is False