# HIPAA Privacy Notes

## Purpose

This document explains how the BenefitsAI project handles healthcare and insurance data in a privacy-aware way.

## Data Used

BenefitsAI uses public and synthetic datasets:

1. CMS Exchange Public Use Files
2. CMS DE-SynPUF synthetic Medicare claims data

The project does not use real protected health information.

## Privacy Practices Demonstrated

### 1. Public and Synthetic Data

The project avoids real patient-level PHI and uses synthetic Medicare claims data for claims modeling.

### 2. Redaction

The project includes lightweight redaction functions for:

1. Email addresses
2. Phone numbers
3. Social Security numbers
4. Member or patient identifiers

### 3. Audit Logging

The project logs:

1. Timestamp
2. Task type
3. Model or system used
4. Redacted query
5. Source documents
6. Confidence score
7. Human review flag

### 4. Human Review

High-risk claim predictions are flagged for human review rather than automatic decisioning.

## Limitations

1. Regex redaction is not a complete HIPAA de-identification system.
2. Production systems should use stronger PHI detection tools.
3. Production systems should include access control, encryption, secrets management, and monitoring.
4. AI outputs should not be used for final healthcare or claim decisions without human review.

## Intended Use

This project demonstrates privacy-aware AI/ML engineering practices for healthcare and benefits administration workflows.