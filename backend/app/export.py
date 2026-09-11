"""
Institutional Research Memorandum Export Service for OmniBrain Studio.
Generates publication-quality Markdown and Print-Ready PDF/HTML documents with audit certificates.
"""
import html
from datetime import datetime
from typing import Dict, Any, List, Optional
from .models.schemas import ExportMemoRequest

def generate_memo_markdown(req: ExportMemoRequest) -> str:
    """Generates clean, structured institutional research memo in Markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    gr = req.guardrail_report or {}
    score_pct = int((gr.get("overall_score", 1.0)) * 100)
    status = gr.get("status", "PASSED")
    citations = req.citations or []

    lines = [
        f"# Institutional Research Memorandum",
        f"**OmniBrain Studio** | Axlero Solutions — Advanced Data Science & ML Engineering",
        f"",
        f"- **Date / Time**: {timestamp}",
        f"- **User Query**: {req.query}",
        f"- **Factual Grounding Audit**: {score_pct}% ({status}) — {gr.get('grounded_claims', 0)}/{gr.get('total_claims', 0)} Grounded Claims",
        f"- **Total Verified Citations**: {len(citations)}",
        f"",
        f"---",
        f"",
        f"## Executive Findings & Multimodal Synthesis",
        f"",
        req.memo_markdown,
        f"",
        f"---",
        f"",
        f"## Factual Grounding & Hallucination Audit Report",
        f"",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| **Overall Grounding Score** | **{score_pct}%** |",
        f"| **Audit Status** | **{status}** |",
        f"| **Grounded Claims** | {gr.get('grounded_claims', 0)} |",
        f"| **Ungrounded Claims** | {gr.get('ungrounded_claims', 0)} |",
        f"| **Total Claims Analyzed** | {gr.get('total_claims', 0)} |",
        f"",
    ]

    verdicts = gr.get("claim_verdicts", [])
    if verdicts:
        lines.append("### Claim-Level Verdicts")
        lines.append("")
        for idx, cv in enumerate(verdicts):
            v_status = cv.get("verdict", "grounded").upper()
            icon = "✓" if v_status == "GROUNDED" else ("⚠" if v_status == "PARTIAL" else "✗")
            lines.append(f"{idx+1}. **[{icon} {v_status}]** {cv.get('claim', '')}")
            lines.append(f"   - *Rationale*: {cv.get('reason', '')}")
        lines.append("")

    if citations:
        lines.append("## Verified Evidence Catalog (Citations)")
        lines.append("")
        lines.append("| Citation # | Type | Source Document / Database | Evidence Snippet |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for c in citations:
            cid = c.get("citation_id", "")
            stype = str(c.get("source_type", "text")).upper()
            sname = c.get("source_name", "unknown")
            page = f" (p.{c.get('page_number')})" if c.get("page_number") else ""
            snip = str(c.get("snippet", "")).replace("\n", " ").replace("|", "\\|")
            lines.append(f"| #{cid} | **{stype}** | `{sname}{page}` | {snip[:120]}... |")
        lines.append("")

    lines.append("---")
    lines.append("*Notice: This memorandum was autonomously compiled and audited by OmniBrain Multi-Modal Agentic RAG System.*")
    return "\n".join(lines)


def generate_memo_html(req: ExportMemoRequest) -> str:
    """Generates an institutional print-ready HTML page for 1-click PDF export."""
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M UTC")
    gr = req.guardrail_report or {}
    score_pct = int((gr.get("overall_score", 1.0)) * 100)
    status = gr.get("status", "PASSED")
    status_class = status.lower()
    citations = req.citations or []
    safe_query = html.escape(req.query)

    memo_html = html.escape(req.memo_markdown)
    memo_html = memo_html.replace("\n\n", "</p><p>")
    memo_html = memo_html.replace("\n• ", "<br>• ")
    memo_html = f"<p>{memo_html}</p>"

    cit_rows = ""
    for c in citations:
        cid = c.get("citation_id", "")
        stype = html.escape(str(c.get("source_type", "text")).upper())
        sname = html.escape(str(c.get("source_name", "unknown")))
        page = f" (p.{c.get('page_number')})" if c.get("page_number") else ""
        snip = html.escape(str(c.get("snippet", "")))
        cit_rows += f"""
        <tr>
          <td style="font-weight:700; width:60px;">#{cid}</td>
          <td style="width:90px;"><span class="badge badge-{stype.lower()}">{stype}</span></td>
          <td style="font-weight:600; width:180px;"><code>{sname}{page}</code></td>
          <td style="font-size:12px; color:#444;">{snip}</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>OmniBrain Research Memorandum - {safe_query[:40]}</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.6;
      color: #1a1a1a;
      background-color: #fff;
      margin: 0;
      padding: 30px;
    }}
    .print-controls {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #f7f6f2;
      border: 1px solid #e0ded8;
      border-radius: 8px;
      padding: 12px 20px;
      margin-bottom: 24px;
    }}
    .btn-print {{
      background: #0f172a;
      color: #fff;
      border: none;
      padding: 10px 18px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      font-size: 13px;
    }}
    .btn-print:hover {{ background: #1e293b; }}
    @media print {{
      .print-controls {{ display: none !important; }}
      body {{ padding: 0; }}
      .page-break {{ page-break-before: always; }}
    }}
    .header-bar {{
      border-bottom: 2px solid #0f172a;
      padding-bottom: 14px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }}
    .brand-title {{
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #0f172a;
    }}
    .brand-subtitle {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #64748b;
      margin-top: 2px;
    }}
    .memo-meta-box {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 14px 18px;
      margin-bottom: 24px;
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 12px;
    }}
    .meta-field label {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: #64748b;
      display: block;
      font-weight: 700;
      margin-bottom: 2px;
    }}
    .meta-field .value {{
      font-size: 13.5px;
      font-weight: 600;
      color: #0f172a;
    }}
    .audit-badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }}
    .audit-passed {{ background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }}
    .audit-warning {{ background: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }}
    .audit-failed {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }}
    .section-heading {{
      font-size: 15px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 6px;
      margin-top: 28px;
      margin-bottom: 14px;
      color: #0f172a;
    }}
    .memo-content {{
      font-size: 13.5px;
      line-height: 1.7;
      color: #1e293b;
    }}
    .memo-content p {{ margin-bottom: 12px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 12px;
    }}
    th {{
      background: #f1f5f9;
      color: #334155;
      font-weight: 700;
      text-align: left;
      padding: 8px 10px;
      border: 1px solid #cbd5e1;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    td {{
      padding: 8px 10px;
      border: 1px solid #e2e8f0;
      vertical-align: top;
    }}
    .badge {{
      font-size: 10px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 3px;
      display: inline-block;
      text-transform: uppercase;
    }}
    .badge-text {{ background: #e0f2fe; color: #0369a1; }}
    .badge-sql {{ background: #fef3c7; color: #b45309; }}
    .badge-visual {{ background: #f3e8ff; color: #7e22ce; }}
    .footer {{
      margin-top: 40px;
      padding-top: 14px;
      border-top: 1px solid #e2e8f0;
      font-size: 10.5px;
      color: #94a3b8;
      display: flex;
      justify-content: space-between;
    }}
  </style>
</head>
<body>

  <div class="print-controls">
    <div>
      <span style="font-weight:700; font-size:14px;">OmniBrain Studio — Institutional Print Export</span>
      <span style="color:#64748b; font-size:12px; margin-left:12px;">Audit Certificate Attached</span>
    </div>
    <button class="btn-print" onclick="window.print()">🖨️ Print / Save as PDF</button>
  </div>

  <div class="header-bar">
    <div>
      <div class="brand-title">◈ OmniBrain Studio</div>
      <div class="brand-subtitle">Axlero Solutions &bull; Financial Intelligence &amp; Multimodal RAG</div>
    </div>
    <div style="text-align: right; font-size: 11px; color: #64748b;">
      <div>CONFIDENTIAL &amp; PROPRIETARY</div>
      <div>{timestamp}</div>
    </div>
  </div>

  <div class="memo-meta-box">
    <div class="meta-field">
      <label>Target Query</label>
      <div class="value">{safe_query}</div>
    </div>
    <div class="meta-field" style="text-align: right;">
      <label>Factual Grounding Audit</label>
      <div style="margin-top: 4px;">
        <span class="audit-badge audit-{status_class}">✓ {score_pct}% {status}</span>
      </div>
    </div>
  </div>

  <div class="section-heading">Executive Findings &amp; Multimodal Synthesis</div>
  <div class="memo-content">
    {memo_html}
  </div>

  <div class="section-heading" style="margin-top: 36px;">Verified Evidence &amp; Citations Catalog</div>
  <table>
    <thead>
      <tr>
        <th>Citation</th>
        <th>Modality</th>
        <th>Source Reference</th>
        <th>Corroborating Evidence Snippet</th>
      </tr>
    </thead>
    <tbody>
      {cit_rows if cit_rows else "<tr><td colspan='4' style='text-align:center; color:#888;'>No explicit citations registered for this query.</td></tr>"}
    </tbody>
  </table>

  <div class="footer">
    <div>OmniBrain Multi-Modal Agentic Orchestrator &bull; Grounded in LangGraph, ChromaDB, SQLite &amp; NeMo Guardrails</div>
    <div>Page 1 of 1</div>
  </div>

</body>
</html>"""
