# Import requests for calling local Ollama
import requests

# Import time for timing answer generation
import time


class RiskRadarGenerator:
    """
    Answer generation layer for RiskRadar AI.

    This class handles:
    - formatting retrieved evidence
    - formatting optional financial context
    - building grounded prompts
    - calling local Ollama
    - returning answer packages
    """

    def __init__(
        self,
        ollama_url="http://localhost:11434/api/generate",
        ollama_model="llama3.2:3b"
    ):
        """
        Initialize generator settings.
        """

        # Store local Ollama endpoint
        self.ollama_url = ollama_url

        # Store local Ollama model name
        self.ollama_model = ollama_model

    def format_evidence_for_prompt(self, evidence_records, max_chars_per_chunk=900):
        """
        Format retrieved SEC evidence records for the prompt.

        Each evidence chunk becomes a numbered source.
        """

        # Return empty string if no evidence exists
        if not evidence_records:
            return ""

        # Create empty list for evidence blocks
        evidence_blocks = []

        # Loop through evidence records
        for index, row in enumerate(evidence_records):

            # Create source number
            source_number = index + 1

            # Shorten chunk text so local generation is faster
            chunk_text = str(row.get("chunk_text", ""))[:max_chars_per_chunk]

            # Get citation metadata
            citation_label = row.get("citation_label", "")
            filing_url = row.get("filing_url", "")
            ticker = row.get("ticker", "")
            section_name = row.get("section_name", "")

            # Build one evidence block
            evidence_block = f"""
SOURCE {source_number}
Ticker: {ticker}
Section: {section_name}
Citation: {citation_label}
Filing URL: {filing_url}
Text:
{chunk_text}
"""

            # Store evidence block
            evidence_blocks.append(evidence_block.strip())

        # Join evidence blocks
        formatted_evidence = "\n\n---\n\n".join(evidence_blocks)

        # Return formatted evidence
        return formatted_evidence

    def format_sources(self, evidence_records):
        """
        Create a clean source list from evidence records.
        """

        # Create empty source list
        sources = []

        # Loop through evidence records
        for index, row in enumerate(evidence_records):

            # Store source metadata
            sources.append({
                "source_number": index + 1,
                "ticker": row.get("ticker", ""),
                "company_name": row.get("company_name", ""),
                "section_name": row.get("section_name", ""),
                "citation_label": row.get("citation_label", ""),
                "filing_url": row.get("filing_url", "")
            })

        # Return sources
        return sources

    def build_rag_prompt(self, question, evidence_records, financial_context=None):
        """
        Build a grounded RAG prompt.

        The model must answer only from:
        - retrieved SEC evidence
        - optional structured financial context
        """

        # Format retrieved evidence
        formatted_evidence = self.format_evidence_for_prompt(evidence_records)

        # Build financial context block only if provided
        if financial_context:
            financial_context_block = f"""
Structured SEC financial context:
{financial_context}
"""
        else:
            financial_context_block = ""

        # Build prompt
        prompt = f"""
You are RiskRadar AI, a financial risk analysis assistant.

You must answer using only the information provided below.

Allowed evidence:
1. Retrieved SEC filing evidence.
2. Structured SEC financial context, only if it is provided.

Rules:
1. Do not use outside knowledge.
2. Do not make unsupported claims.
3. If the retrieved evidence is not enough, say: "The retrieved SEC evidence is not sufficient to answer this."
4. Cite SEC text evidence using bracketed source numbers like [Source 1], [Source 2].
5. Do not invent citations.
6. Keep the answer clear, business-focused, and concise.
7. Include a short "Evidence Used" section at the end.

User question:
{question}

{financial_context_block}

Retrieved SEC filing evidence:
{formatted_evidence}

Answer:
"""

        # Return clean prompt
        return prompt.strip()

    def test_ollama_connection(self, timeout_seconds=60):
        """
        Test whether Ollama is running and the selected model responds.
        """

        # Create tiny test payload
        payload = {
            "model": self.ollama_model,
            "prompt": "Reply with exactly: Ollama is working.",
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 20
            }
        }

        # Try to call Ollama
        try:
            # Send request
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=timeout_seconds
            )

            # Raise error if request failed
            response.raise_for_status()

            # Parse response
            result = response.json()

            # Return success status
            return {
                "ok": True,
                "model": self.ollama_model,
                "response": result.get("response", "").strip()
            }

        # Handle errors safely
        except Exception as error:
            return {
                "ok": False,
                "model": self.ollama_model,
                "error": str(error)
            }

    def generate_answer_with_ollama(
        self,
        prompt,
        temperature=0.1,
        max_tokens=350,
        timeout_seconds=600
    ):
        """
        Generate an answer using local Ollama.
        """

        # Create Ollama request payload
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        # Try to call Ollama safely
        try:
            # Send request to Ollama
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=timeout_seconds
            )

            # Raise error if request failed
            response.raise_for_status()

            # Parse JSON response
            result = response.json()

            # Return generated answer
            return result.get("response", "").strip()

        # Handle timeout clearly
        except requests.exceptions.ReadTimeout:
            return (
                "Ollama timed out before finishing the answer. "
                "Use a smaller model, reduce top_k, or reduce evidence length."
            )

        # Handle any other failure clearly
        except Exception as error:
            return f"Ollama generation failed: {error}"

    def answer_question(
        self,
        question,
        evidence_records,
        financial_context=None,
        temperature=0.1,
        max_tokens=350,
        timeout_seconds=600
    ):
        """
        Build prompt, generate answer, and return final answer package.
        """

        # Start timer
        start_time = time.time()

        # If no evidence exists, return a safe fallback
        if not evidence_records:
            return {
                "question": question,
                "answer": "No relevant SEC evidence was retrieved.",
                "sources": [],
                "financial_context": financial_context,
                "financial_context_used": financial_context is not None,
                "prompt": None,
                "generation_time_seconds": 0
            }

        # Build grounded prompt
        prompt = self.build_rag_prompt(
            question=question,
            evidence_records=evidence_records,
            financial_context=financial_context
        )

        # Generate answer
        answer = self.generate_answer_with_ollama(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds
        )

        # End timer
        end_time = time.time()

        # Build clean source list
        sources = self.format_sources(evidence_records)

        # Return answer package
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "financial_context": financial_context,
            "financial_context_used": financial_context is not None,
            "prompt": prompt,
            "generation_time_seconds": round(end_time - start_time, 2)
        }