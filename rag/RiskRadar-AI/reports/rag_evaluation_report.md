# RiskRadar AI: RAG Evaluation Report

This report summarizes the first evaluation layer for RiskRadar AI.

## Evaluation Summary

- **demo_answers**: 5.0
- **answers_with_source_markers**: 5.0
- **source_marker_rate**: 1.0
- **average_citations_per_answer**: 3.0
- **average_citation_validity_rate**: 1.0
- **average_answer_evidence_overlap**: 0.7694996619337391
- **answers_with_quality_flags**: 0.0

## Retrieval Summary

- {'method': 'filtered', 'metric': 'test_questions', 'value': 8.0}
- {'method': 'filtered', 'metric': 'ticker_hit_rate_at_5', 'value': 1.0}
- {'method': 'filtered', 'metric': 'section_hit_rate_at_5', 'value': 1.0}
- {'method': 'filtered', 'metric': 'pair_hit_rate_at_5', 'value': 1.0}
- {'method': 'filtered', 'metric': 'average_top_distance', 'value': 0.9139382243156432}
- {'method': 'unfiltered', 'metric': 'test_questions', 'value': 8.0}
- {'method': 'unfiltered', 'metric': 'ticker_hit_rate_at_5', 'value': 1.0}
- {'method': 'unfiltered', 'metric': 'section_hit_rate_at_5', 'value': 0.875}
- {'method': 'unfiltered', 'metric': 'pair_hit_rate_at_5', 'value': 0.75}
- {'method': 'unfiltered', 'metric': 'average_top_distance', 'value': 0.8046567961573601}

## Known Failure Cases

- **Question asks about a company not in the vector database**: Validate ticker coverage before retrieval.
- **Question asks for current news**: Add a separate live-data or news retrieval layer.
- **Evidence is retrieved from the right company but wrong section**: Add query routing to choose likely SEC sections.
- **LLM makes a claim not clearly supported by evidence**: Use faithfulness evaluation and quote-level citation checks.
- **Chunk boundary cuts off context**: Improve with sentence-aware or semantic chunking.

## Conclusion
The RAG system successfully generates cited answers from retrieved SEC filing evidence. The evaluation layer checks citation usage, citation validity, answer-evidence overlap, retrieval quality, and known failure cases. Future improvements should add stronger faithfulness scoring, query routing, and human review.