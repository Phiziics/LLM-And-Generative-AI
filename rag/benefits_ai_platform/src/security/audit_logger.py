from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

from src.security.redact_pii import redact_sensitive_text, contains_sensitive_pattern


class AuditLogger:
    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()

        self.audit_dir = (
            self.project_root
            / "data"
            / "04_predictions"
            / "audit_logs"
        )

        self.audit_dir.mkdir(parents=True, exist_ok=True)

        self.audit_csv_path = self.audit_dir / "ai_audit_log.csv"
        self.audit_jsonl_path = self.audit_dir / "ai_audit_log.jsonl"

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def create_record(
        self,
        user_query: str,
        task_type: str,
        model_or_system: str,
        response_summary: str,
        source_documents: list[str] | None = None,
        confidence_score: float | None = None,
        human_review_required: bool = False,
    ) -> dict:
        redacted_query = redact_sensitive_text(user_query)

        record = {
            "audit_id": self._hash_text(
                f"{datetime.now(timezone.utc).isoformat()}_{user_query}"
            )[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": task_type,
            "model_or_system": model_or_system,
            "raw_query_hash": self._hash_text(user_query),
            "redacted_query": redacted_query,
            "sensitive_pattern_detected": contains_sensitive_pattern(user_query),
            "response_summary": response_summary,
            "source_documents": source_documents if source_documents else [],
            "confidence_score": confidence_score,
            "human_review_required": human_review_required,
        }

        return record

    def log_event(
        self,
        user_query: str,
        task_type: str,
        model_or_system: str,
        response_summary: str,
        source_documents: list[str] | None = None,
        confidence_score: float | None = None,
        human_review_required: bool = False,
    ) -> dict:
        record = self.create_record(
            user_query=user_query,
            task_type=task_type,
            model_or_system=model_or_system,
            response_summary=response_summary,
            source_documents=source_documents,
            confidence_score=confidence_score,
            human_review_required=human_review_required,
        )

        self._append_csv(record)
        self._append_jsonl(record)

        return record

    def _append_csv(self, record: dict) -> None:
        csv_record = record.copy()
        csv_record["source_documents"] = json.dumps(csv_record["source_documents"])

        record_df = pd.DataFrame([csv_record])

        if self.audit_csv_path.exists():
            record_df.to_csv(
                self.audit_csv_path,
                mode="a",
                header=False,
                index=False,
            )
        else:
            record_df.to_csv(
                self.audit_csv_path,
                index=False,
            )

    def _append_jsonl(self, record: dict) -> None:
        with self.audit_jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    logger = AuditLogger()

    record = logger.log_event(
        user_query="What is the out-of-pocket maximum for member ID ABC123?",
        task_type="benefits_rag_retrieval",
        model_or_system="sentence-transformers/all-MiniLM-L6-v2 + FAISS",
        response_summary="Retrieved top CMS plan chunks for benefits question.",
        source_documents=["sample_plan_document.txt"],
        confidence_score=0.82,
        human_review_required=False,
    )

    print(record)