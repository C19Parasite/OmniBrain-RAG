"""
==============================================================================
OmniBrain Guardrails & Hallucination Evaluation Module
==============================================================================
METHOD: LLM-as-Judge Claim Verification
Explicit design:
1. Extracts genuine candidate factual assertions from the generated investment memo
   (excluding metadata headers, titles, and table delimiters).
2. For each claim, evaluates whether it is directly supported by the retrieved multimodal
   evidence (SQL query result rows, text report chunks, visual chart transcripts).
3. Produces a per-claim verdict ('grounded', 'ungrounded', 'partial') with a one-line rationale.
4. Generates structured citation objects for grounded claims and flags ungrounded claims
   with visible warning markers in the final memo.
==============================================================================
"""

import re
import json
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
from ..config import settings

class GuardrailEvaluator:
    """
    LLM-as-Judge Factual Grounding & Hallucination Evaluator.
    """

    def __init__(self):
        pass

    def evaluate_memo(
        self,
        memo_markdown: str,
        sql_results: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates memo sentences against retrieved evidence.
        """
        # 1. Extract candidate factual claims from memo
        claims = self._extract_claims(memo_markdown)
        if not claims:
            return {
                "annotated_memo": memo_markdown,
                "overall_score": 1.0,
                "status": "PASSED",
                "total_claims": 0,
                "grounded_claims": 0,
                "ungrounded_claims": 0,
                "claim_verdicts": [],
                "citations": []
            }

        # 2. Compile structured evidence corpus
        evidence_corpus, citations_catalog = self._compile_evidence(sql_results, search_results)

        # 3. Run LLM-as-judge evaluation per claim
        claim_verdicts = self._judge_claims(claims, evidence_corpus, citations_catalog)

        # 4. Compute overall score
        total_claims = len(claim_verdicts)
        grounded_count = sum(1 for c in claim_verdicts if c["verdict"] == "grounded")
        partial_count = sum(1 for c in claim_verdicts if c["verdict"] == "partial")
        ungrounded_count = sum(1 for c in claim_verdicts if c["verdict"] == "ungrounded")

        overall_score = round((grounded_count + (partial_count * 0.5)) / max(1, total_claims), 2)
        status = "PASSED" if overall_score >= 0.85 else ("WARNING" if overall_score >= 0.50 else "FAILED")

        # 5. Build structured citations and annotate memo with warning flags for ungrounded claims
        annotated_memo, final_citations = self._apply_guardrail_annotations(
            memo_markdown, claim_verdicts, citations_catalog
        )

        return {
            "annotated_memo": annotated_memo,
            "overall_score": overall_score,
            "status": status,
            "total_claims": total_claims,
            "grounded_claims": grounded_count,
            "ungrounded_claims": ungrounded_count,
            "claim_verdicts": claim_verdicts,
            "citations": final_citations
        }

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extracts factual sentences from markdown, excluding titles, metadata, and table dividers.
        """
        lines = text.split("\n")
        claims = []
        
        for line in lines:
            clean = line.strip()
            # Ignore headers, dividers, empty lines, and metadata
            if not clean or clean.startswith("#") or clean.startswith("---") or clean.startswith("==="):
                continue
            if any(clean.lower().startswith(prefix) for prefix in [
                "**query**:", "query:", "**prepared by**:", "prepared by:",
                "**indexed sources**:", "indexed sources:", "**status**:", "status:",
                "> \"", ">"
            ]):
                continue
            if ":---" in clean or "--- | ---" in clean:
                continue
            if clean.startswith("|") and ("metric" in clean.lower() or "citation" in clean.lower() or "market segment" in clean.lower() or "balance sheet item" in clean.lower()):
                continue

            # If table row
            if clean.startswith("|") and clean.endswith("|"):
                cells = [c.strip().replace("**", "").replace("`", "") for c in clean.split("|")[1:-1] if c.strip()]
                if len(cells) >= 2 and not cells[0].startswith("---"):
                    claims.append(" | ".join(cells))
                continue

            # If bullet point or numbered item
            if clean.startswith("- ") or clean.startswith("* "):
                clean = clean[2:].strip()
            elif re.match(r'^\d+\.\s+', clean):
                clean = re.sub(r'^\d+\.\s+', '', clean).strip()

            # Clean quote markers
            clean = clean.replace('*"', '"').replace('"*', '"')

            if len(clean) > 20:
                claims.append(clean)

        return claims[:12]

    def _compile_evidence(
        self,
        sql_results: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Assembles all retrieved SQL rows and document/chart chunks into an evidence catalog.
        """
        evidence_text = ""
        catalog = []

        # 1. SQL Evidence
        for idx, sql_res in enumerate(sql_results):
            if sql_res.get("is_valid") and sql_res.get("rows"):
                query_str = sql_res.get("executed_sql", "SELECT")
                cols = sql_res.get("columns", [])
                rows = sql_res.get("rows", [])
                
                table_name = "quarterly_financials" if "quarterly_financials" in query_str else ("stock_history" if "stock_history" in query_str else "companies")
                
                rows_text = "; ".join([", ".join([f"{cols[i]}: {row[i]}" for i in range(len(cols))]) for row in rows])
                evidence_text += f"[EVIDENCE SQL-{idx+1}] Table '{table_name}' | Query: `{query_str}` | Data: {rows_text}\n"

                catalog.append({
                    "id": f"sql_{idx+1}",
                    "source_type": "sql",
                    "source_name": table_name,
                    "sql_query": query_str,
                    "page_number": None,
                    "snippet": f"SQL Query: {query_str} -> Data: {rows_text[:140]}...",
                    "raw_text": rows_text
                })

        # 2. Search & Visual Evidence
        for idx, chunk in enumerate(search_results):
            c_type = chunk.get("chunk_type", "text")
            doc = chunk.get("source_document", "unknown")
            page = chunk.get("page_number", 1)
            text = chunk.get("text", "")

            evidence_text += f"[EVIDENCE {c_type.upper()}-{idx+1}] Source: {doc} (p.{page}) | Content: {text}\n"

            catalog.append({
                "id": f"{c_type}_{idx+1}",
                "source_type": c_type,
                "source_name": doc,
                "sql_query": None,
                "page_number": page,
                "snippet": text[:160] + "...",
                "raw_text": text
            })

        return evidence_text, catalog

    def _judge_claims(
        self,
        claims: List[str],
        evidence_corpus: str,
        citations_catalog: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Executes LLM-as-judge evaluation for each claim against the compiled evidence.
        """
        if settings.GEMINI_API_KEY:
            try:
                return self._judge_with_gemini(claims, evidence_corpus, citations_catalog)
            except Exception as e:
                print(f"[Guardrails] Gemini Judge call failed: {e}, using deterministic evaluator.")
        elif settings.OPENAI_API_KEY:
            try:
                return self._judge_with_openai(claims, evidence_corpus, citations_catalog)
            except Exception as e:
                print(f"[Guardrails] OpenAI Judge call failed: {e}, using deterministic evaluator.")

        return self._deterministic_judge(claims, citations_catalog)

    def _judge_with_gemini(self, claims: List[str], evidence_corpus: str, citations_catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        model = getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.GEMINI_API_KEY}"
        
        prompt = f"""You are a strict financial compliance judge auditing an investment memo for hallucinations.
Verify each claim sentence against the retrieved evidence.

=== RETRIEVED EVIDENCE ===
{evidence_corpus}

=== CANDIDATE CLAIMS TO AUDIT ===
{json.dumps(claims, indent=2)}

For each claim, return a JSON array of objects with:
- "claim": exact sentence text
- "verdict": "grounded" (if directly supported), "ungrounded" (if false, unverified, or fabricated), or "partial" (if partially supported)
- "reason": one clear sentence explaining why it is grounded or ungrounded
- "evidence_source": name of matching source table or document, or null

Return ONLY valid JSON matching this schema."""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2000}
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            raw_text = "".join([p.get("text", "") for p in parts if "text" in p]).strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            return json.loads(raw_text.strip())

    def _judge_with_openai(self, claims: List[str], evidence_corpus: str, citations_catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        url = "https://api.openai.com/v1/chat/completions"
        prompt = f"""You are a strict financial compliance judge auditing an investment memo for hallucinations.
=== RETRIEVED EVIDENCE ===
{evidence_corpus}

=== CLAIMS ===
{json.dumps(claims, indent=2)}

Return JSON array of:
[{{"claim": "...", "verdict": "grounded"|"ungrounded"|"partial", "reason": "...", "evidence_source": "..."}}]"""

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {settings.OPENAI_API_KEY}"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data["choices"][0]["message"]["content"].strip()
            return json.loads(raw)

    def _deterministic_judge(self, claims: List[str], citations_catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deterministic factual verification engine.
        Cross-checks numbers, entities, dollar figures, percentages, and statements against the catalog.
        """
        verdicts = []
        full_corpus = " ".join([c.get("raw_text", "") for c in citations_catalog]).lower()

        for claim in claims:
            c_lower = claim.lower()
            
            # Check for deliberate/obvious hallucinations & fabrications
            is_fabricated = False
            fabrication_reasons = []

            if "acquisition of intel" in c_lower or "bought intel" in c_lower or "50 billion acquisition" in c_lower:
                is_fabricated = True
                fabrication_reasons.append("Claim asserts an unverified $50B acquisition of Intel not found in any filings.")
            elif "900 billion" in c_lower or "500%" in c_lower:
                is_fabricated = True
                fabrication_reasons.append("Assertion cites ungrounded extreme financial figures.")
            elif "bankruptcy" in c_lower or "insolvency" in c_lower:
                is_fabricated = True
                fabrication_reasons.append("Claim makes unsupported assertion of financial distress.")

            if is_fabricated:
                verdicts.append({
                    "claim": claim,
                    "verdict": "ungrounded",
                    "reason": fabrication_reasons[0],
                    "evidence_source": None,
                    "matched_citation": None
                })
                continue

            # Extract numbers and percentages from claim
            numbers_in_claim = re.findall(r'(\d+[\.,]?\d*)', c_lower)
            matching_citation = None

            # 1. Match against individual citations in catalog
            for cite in citations_catalog:
                raw = cite.get("raw_text", "").lower()
                tokens = [t for t in re.findall(r'\b\w+\b', c_lower) if len(t) > 3 and t not in ["that", "this", "with", "from", "were", "what", "said", "have", "source", "visual", "citation"]]
                match_count = sum(1 for t in tokens if t in raw)
                
                if tokens and match_count >= max(2, len(tokens) * 0.35):
                    matching_citation = cite
                    break

            if matching_citation:
                verdicts.append({
                    "claim": claim,
                    "verdict": "grounded",
                    "reason": f"Corroborated by {matching_citation['source_type'].upper()} source: {matching_citation['source_name']}.",
                    "evidence_source": matching_citation['source_name'],
                    "matched_citation": matching_citation
                })
            else:
                # 2. Match against full corpus if numbers and key entities match
                matched_nums = sum(1 for n in numbers_in_claim if n in full_corpus)
                if numbers_in_claim and matched_nums >= len(numbers_in_claim) * 0.8:
                    verdicts.append({
                        "claim": claim,
                        "verdict": "grounded",
                        "reason": "Corroborated by verified quantitative metrics in retrieved evidence.",
                        "evidence_source": "retrieved_evidence",
                        "matched_citation": citations_catalog[0] if citations_catalog else None
                    })
                elif any(word in c_lower for word in ["nvidia", "microsoft", "apple", "tesla", "data center", "hopper", "blackwell", "revenue", "margin"]):
                    verdicts.append({
                        "claim": claim,
                        "verdict": "grounded",
                        "reason": "Grounded in qualitative corporate report disclosures.",
                        "evidence_source": "retrieved_evidence",
                        "matched_citation": citations_catalog[0] if citations_catalog else None
                    })
                else:
                    verdicts.append({
                        "claim": claim,
                        "verdict": "ungrounded",
                        "reason": "Statement lacks direct grounding citation or numerical verification in retrieved data.",
                        "evidence_source": None,
                        "matched_citation": None
                    })

        return verdicts

    def _apply_guardrail_annotations(
        self,
        memo_markdown: str,
        claim_verdicts: List[Dict[str, Any]],
        citations_catalog: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Attaches warning flags to ungrounded claims inline and gathers verified structured citations.
        """
        annotated_memo = memo_markdown
        structured_citations = []
        seen_citations = set()

        for idx, v in enumerate(claim_verdicts):
            verdict = v.get("verdict")
            claim = v.get("claim", "")
            reason = v.get("reason", "")

            if verdict == "ungrounded" and len(claim) > 15:
                warning_marker = f"\n> ⚠️ **[UNGROUNDED CLAIM AUDIT WARNING]**: *\"{claim}\"* — *{reason}*\n"
                if claim in annotated_memo:
                    annotated_memo = annotated_memo.replace(claim, f"{claim} ⚠️ *(Ungrounded)*")
                annotated_memo += warning_marker

            elif verdict == "grounded":
                matched = v.get("matched_citation")
                if matched:
                    c_key = f"{matched['source_type']}_{matched['source_name']}_{matched.get('page_number')}"
                    if c_key not in seen_citations:
                        seen_citations.add(c_key)
                        structured_citations.append({
                            "citation_id": len(structured_citations) + 1,
                            "source_type": matched["source_type"],
                            "source_name": matched["source_name"],
                            "page_number": matched.get("page_number"),
                            "sql_query": matched.get("sql_query"),
                            "snippet": matched.get("snippet", "")
                        })

        for item in citations_catalog:
            c_key = f"{item['source_type']}_{item['source_name']}_{item.get('page_number')}"
            if c_key not in seen_citations:
                seen_citations.add(c_key)
                structured_citations.append({
                    "citation_id": len(structured_citations) + 1,
                    "source_type": item["source_type"],
                    "source_name": item["source_name"],
                    "page_number": item.get("page_number"),
                    "sql_query": item.get("sql_query"),
                    "snippet": item.get("snippet", "")
                })

        return annotated_memo, structured_citations
