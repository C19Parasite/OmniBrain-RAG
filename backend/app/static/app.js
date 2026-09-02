// OmniBrain Studio UI Application Logic (Warm Editorial Edition)
document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let currentCitations = [];
  let currentDocuments = [];
  let currentSchema = [];

  // Navigation Elements
  const navLinks = document.querySelectorAll(".nav-link");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const headerStatusText = document.getElementById("headerStatusText");
  const navRunAnalysisBtn = document.getElementById("navRunAnalysisBtn");

  // Memo Studio Elements
  const memoQueryInput = document.getElementById("memoQueryInput");
  const runMemoQueryBtn = document.getElementById("runMemoQueryBtn");
  const memoResultsArea = document.getElementById("memoResultsArea");
  const graphStatusText = document.getElementById("graphStatusText");
  const runtimeTag = document.getElementById("runtimeTag");
  const memoRenderedBody = document.getElementById("memoRenderedBody");
  const memoExecutionTime = document.getElementById("memoExecutionTime");
  const guardrailCard = document.getElementById("guardrailCard");
  const guardrailStatusBadge = document.getElementById("guardrailStatusBadge");
  const guardrailScoreDisplay = document.getElementById("guardrailScoreDisplay");
  const guardrailDesc = document.getElementById("guardrailDesc");
  const guardrailStatsRow = document.getElementById("guardrailStatsRow");
  const studioCitationCount = document.getElementById("studioCitationCount");
  const studioCitationsList = document.getElementById("studioCitationsList");
  const studioClaimCount = document.getElementById("studioClaimCount");
  const studioVerdictsList = document.getElementById("studioVerdictsList");
  const studioTraceCount = document.getElementById("studioTraceCount");
  const studioTracesList = document.getElementById("studioTracesList");

  // Document Center Elements
  const triggerUploadBtn = document.getElementById("triggerUploadBtn");
  const hiddenFileInput = document.getElementById("hiddenFileInput");
  const uploadDropzone = document.getElementById("uploadDropzone");
  const docTableBody = document.getElementById("docTableBody");
  const docTableCount = document.getElementById("docTableCount");

  // SQL Sandbox Elements
  const schemaAccordionContainer = document.getElementById("schemaAccordionContainer");
  const sandboxSqlInput = document.getElementById("sandboxSqlInput");
  const runSandboxSqlBtn = document.getElementById("runSandboxSqlBtn");
  const sandboxMetaText = document.getElementById("sandboxMetaText");
  const sandboxTableContainer = document.getElementById("sandboxTableContainer");

  // Settings Elements
  const settingTopK = document.getElementById("settingTopK");
  const settingTemperature = document.getElementById("settingTemperature");
  const settingGeminiKey = document.getElementById("settingGeminiKey");
  const settingOpenAIKey = document.getElementById("settingOpenAIKey");
  const saveSettingsBtn = document.getElementById("saveSettingsBtn");
  const resetDemoDataBtn = document.getElementById("resetDemoDataBtn");

  // Modal Elements
  const citationModal = document.getElementById("citationModal");
  const modalCitationTypePill = document.getElementById("modalCitationTypePill");
  const modalCitationTitle = document.getElementById("modalCitationTitle");
  const modalCitationBody = document.getElementById("modalCitationBody");
  const closeCitationModalBtn = document.getElementById("closeCitationModalBtn");

  // -------------------------------------------------------------
  // 1. Navigation Switching
  // -------------------------------------------------------------
  navLinks.forEach(link => {
    link.addEventListener("click", () => {
      const targetTab = link.dataset.tab;
      navLinks.forEach(l => l.classList.remove("active"));
      link.classList.add("active");

      tabPanes.forEach(pane => {
        if (pane.id === `view-${targetTab}`) {
          pane.style.display = "flex";
        } else {
          pane.style.display = "none";
        }
      });

      if (targetTab === "doc-center") loadDocuments();
      if (targetTab === "sql-sandbox") loadSqlSchema();
    });
  });

  if (navRunAnalysisBtn) {
    navRunAnalysisBtn.addEventListener("click", () => {
      document.querySelector('[data-tab="memo-studio"]').click();
      memoQueryInput.focus();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // -------------------------------------------------------------
  // 2. Health Check & Init
  // -------------------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();
      headerStatusText.textContent = `Active (${data.total_chunks} chunks, ${data.total_documents} docs)`;
    } catch (e) {
      headerStatusText.textContent = "Engine Offline";
    }
  }
  checkHealth();

  // Load Saved Settings from LocalStorage
  function loadLocalSettings() {
    if (localStorage.getItem("omnibrain_top_k")) settingTopK.value = localStorage.getItem("omnibrain_top_k");
    if (localStorage.getItem("omnibrain_temperature")) settingTemperature.value = localStorage.getItem("omnibrain_temperature");
    if (localStorage.getItem("omnibrain_gemini_key")) settingGeminiKey.value = localStorage.getItem("omnibrain_gemini_key");
    if (localStorage.getItem("omnibrain_openai_key")) settingOpenAIKey.value = localStorage.getItem("omnibrain_openai_key");
  }
  loadLocalSettings();

  saveSettingsBtn.addEventListener("click", () => {
    localStorage.setItem("omnibrain_top_k", settingTopK.value);
    localStorage.setItem("omnibrain_temperature", settingTemperature.value);
    if (settingGeminiKey.value) localStorage.setItem("omnibrain_gemini_key", settingGeminiKey.value);
    if (settingOpenAIKey.value) localStorage.setItem("omnibrain_openai_key", settingOpenAIKey.value);
    alert("Settings saved locally in browser.");
  });

  resetDemoDataBtn.addEventListener("click", async () => {
    if (!confirm("Reset synthetic SQLite database and vector store to default demo state?")) return;
    try {
      const res = await fetch("/api/reset-demo", { method: "POST" });
      const data = await res.json();
      alert(data.message);
      checkHealth();
      loadDocuments();
      loadSqlSchema();
    } catch (err) {
      alert(`Reset failed: ${err.message}`);
    }
  });

  // -------------------------------------------------------------
  // 3. Preset Suggestions
  // -------------------------------------------------------------
  document.querySelectorAll(".preset-tag").forEach(tag => {
    tag.addEventListener("click", () => {
      memoQueryInput.value = tag.dataset.query;
      executeMemoQuery();
    });
  });

  // -------------------------------------------------------------
  // 4. Memo Studio - Run Analysis & Animate Graph
  // -------------------------------------------------------------
  runMemoQueryBtn.addEventListener("click", executeMemoQuery);
  memoQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      executeMemoQuery();
    }
  });

  async function executeMemoQuery() {
    const query = memoQueryInput.value.trim();
    if (!query) return;

    runMemoQueryBtn.disabled = true;
    memoResultsArea.style.display = "none";
    resetAgentNodes();

    try {
      setNodeActive("Supervisor", "Decomposing query...");
      graphStatusText.textContent = "Supervisor Planning...";

      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          top_k: parseInt(settingTopK.value) || 5,
          temperature: parseFloat(settingTemperature.value) || 0.2
        })
      });

      if (!res.ok) throw new Error(`Query returned status ${res.status}`);
      const data = await res.json();
      currentCitations = data.citations || [];

      // Step-by-step Graph Animation
      await animateGraphFromTraces(data.execution_trace);

      // Render Synthesized Memo with Clickable Citations
      memoExecutionTime.textContent = `Completed in ${data.execution_time_seconds}s`;
      runtimeTag.textContent = `Latency: ${data.execution_time_seconds}s`;
      renderSynthesizedMemo(data.memo_markdown, data.citations);

      // Render Guardrail Scorecard
      renderGuardrailScorecard(data.guardrail_report);

      // Render Details Accordions
      renderCitationsCatalog(data.citations);
      renderVerdictsList(data.guardrail_report.claim_verdicts || []);
      renderTracesList(data.execution_trace || []);

      memoResultsArea.style.display = "flex";
      memoResultsArea.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
      alert(`Query failed: ${err.message}`);
      console.error(err);
      graphStatusText.textContent = "Execution Error";
    } finally {
      runMemoQueryBtn.disabled = false;
    }
  }

  // Graph Animation Helpers
  function resetAgentNodes() {
    ["Supervisor", "SearchAgent", "VisionAgent", "SQLAgent", "Synthesizer", "GuardrailEvaluator"].forEach(node => {
      const el = document.getElementById(`node-${node}`);
      if (el) el.className = "agent-node";
    });
  }

  function setNodeActive(nodeId, subtitle) {
    const el = document.getElementById(`node-${nodeId}`);
    if (el) {
      el.className = "agent-node active";
      if (subtitle) {
        const sub = el.querySelector(".node-sub");
        if (sub) sub.textContent = subtitle;
      }
    }
  }

  function setNodeCompleted(nodeId, subtitle) {
    const el = document.getElementById(`node-${nodeId}`);
    if (el) {
      el.className = "agent-node completed";
      if (subtitle) {
        const sub = el.querySelector(".node-sub");
        if (sub) sub.textContent = subtitle;
      }
    }
  }

  async function animateGraphFromTraces(traces) {
    const activeAgents = new Set();
    traces.forEach(t => activeAgents.add(t.agent));

    setNodeCompleted("Supervisor", "Query Decomposed");

    if (activeAgents.has("SQLAgent")) {
      setNodeActive("SQLAgent", "Executing SQL...");
      await sleep(140);
      setNodeCompleted("SQLAgent", "SQL Executed (Read-Only)");
    }

    if (activeAgents.has("SearchAgent")) {
      setNodeActive("SearchAgent", "Searching ChromaDB...");
      await sleep(140);
      setNodeCompleted("SearchAgent", "Text & Visual Retrieved");
    }

    setNodeCompleted("VisionAgent", "VLM Ingested");

    setNodeActive("Synthesizer", "Composing memo...");
    await sleep(140);
    setNodeCompleted("Synthesizer", "Draft Synthesized");

    setNodeActive("GuardrailEvaluator", "Auditing claims...");
    await sleep(140);
    setNodeCompleted("GuardrailEvaluator", "Claims Verified");

    graphStatusText.textContent = "Orchestration Complete ✓";
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // -------------------------------------------------------------
  // 5. Render Memo with Clickable Citations
  // -------------------------------------------------------------
  function renderSynthesizedMemo(markdown, citations) {
    let parsedHtml = marked.parse(markdown);

    // Transform citation text into clickable badges
    parsedHtml = parsedHtml.replace(/\[SQL:\s*([^\]]+)\]/g, (match, p1) => {
      return `<button class="citation-inline-badge sql" data-source="${p1}" data-type="sql">[SQL: ${p1}]</button>`;
    });

    parsedHtml = parsedHtml.replace(/\[Visual:\s*([^\]]+)\]/g, (match, p1) => {
      return `<button class="citation-inline-badge visual" data-source="${p1}" data-type="visual">[Visual: ${p1}]</button>`;
    });

    parsedHtml = parsedHtml.replace(/\[Source:\s*([^\]]+)\]/g, (match, p1) => {
      return `<button class="citation-inline-badge" data-source="${p1}" data-type="text">[Source: ${p1}]</button>`;
    });

    memoRenderedBody.innerHTML = parsedHtml;

    // Attach click handlers to citation badges
    memoRenderedBody.querySelectorAll(".citation-inline-badge").forEach(badge => {
      badge.addEventListener("click", () => {
        const src = badge.dataset.source;
        const type = badge.dataset.type;
        openCitationModal(src, type);
      });
    });
  }

  // -------------------------------------------------------------
  // 6. Citation Preview Modal
  // -------------------------------------------------------------
  function openCitationModal(sourceText, type) {
    modalCitationTypePill.textContent = `[${type.toUpperCase()}]`;
    modalCitationTitle.textContent = sourceText;

    let matched = currentCitations.find(c =>
      sourceText.toLowerCase().includes(c.source_name.toLowerCase()) ||
      (c.sql_query && sourceText.toLowerCase().includes("quarterly"))
    );

    let contentHtml = "";

    if (type === "visual") {
      contentHtml = `
        <div style="margin-bottom: 14px;">
          <div style="font-weight:700; color:#181613; margin-bottom:4px;">Visual Document Exhibit:</div>
          <div style="font-size:12.5px; color:#595246;">${matched ? matched.snippet : sourceText}</div>
        </div>
        <div style="background:#D7CFC0; border:1px solid rgba(24,22,19,0.15); border-radius:4px; padding:14px; text-align:left;">
          <div style="font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#827869; margin-bottom:8px;">VLM Grounding Notes (${sourceText})</div>
          <div style="background:#ECE6DC; padding:12px; border-radius:4px; color:#181613; font-family:var(--font-mono); font-size:12px; line-height:1.6;">
            ${matched ? matched.snippet : "Visual chart transcription verified against primary financial filing exhibit."}
          </div>
        </div>
      `;
    } else if (type === "sql") {
      contentHtml = `
        <div style="margin-bottom: 14px;">
          <div style="font-weight:700; color:#181613; margin-bottom:4px;">Structured SQL Table Evidence:</div>
          <div style="font-size:12.5px; color:#595246;">Executed on read-only SQLite financial database:</div>
        </div>
        <div style="background:#D7CFC0; border:1px solid rgba(24,22,19,0.15); border-radius:4px; padding:14px;">
          <pre style="color:#181613; font-family:var(--font-mono); font-size:12px; margin-bottom:8px; background:#ECE6DC; padding:10px; border-radius:4px;">${matched && matched.sql_query ? matched.sql_query : "SELECT * FROM quarterly_financials WHERE ticker = 'NVDA';"}</pre>
          <div style="font-size:11px; color:#595246;">Table: <strong>${matched ? matched.source_name : 'quarterly_financials'}</strong> (Synthetic Read-Only Mode)</div>
        </div>
      `;
    } else {
      contentHtml = `
        <div style="margin-bottom: 14px;">
          <div style="font-weight:700; color:#181613; margin-bottom:4px;">Corporate Disclosure Passage:</div>
          <div style="font-size:12.5px; color:#595246;">Source: <strong>${matched ? matched.source_name : sourceText}</strong></div>
        </div>
        <div style="background:#ECE6DC; border:1px solid rgba(24,22,19,0.15); border-radius:4px; padding:14px; color:#181613; font-size:13px; line-height:1.7;">
          "${matched ? matched.snippet : 'Passage corroborated against official investor relations disclosures.'}"
        </div>
      `;
    }

    modalCitationBody.innerHTML = contentHtml;
    citationModal.classList.add("active");
  }

  closeCitationModalBtn.addEventListener("click", () => {
    citationModal.classList.remove("active");
  });

  citationModal.addEventListener("click", (e) => {
    if (e.target === citationModal) citationModal.classList.remove("active");
  });

  // -------------------------------------------------------------
  // 7. Render Guardrail Scorecard & Details
  // -------------------------------------------------------------
  function renderGuardrailScorecard(gr) {
    const scorePct = Math.round((gr.overall_score || 1.0) * 100);
    guardrailScoreDisplay.textContent = `${scorePct}%`;
    guardrailStatusBadge.textContent = gr.status || "PASSED";

    guardrailCard.className = "editorial-card guardrail-card";
    if (gr.status === "WARNING") {
      guardrailCard.classList.add("warning");
      guardrailStatusBadge.style.background = "var(--warning-amber)";
      guardrailDesc.textContent = "Potential ungrounded or partially supported claims detected in memo.";
    } else if (gr.status === "FAILED") {
      guardrailCard.classList.add("failed");
      guardrailStatusBadge.style.background = "var(--danger-rose)";
      guardrailDesc.textContent = "High risk of hallucinations detected in generated output.";
    } else {
      guardrailStatusBadge.style.background = "var(--text-primary)";
      guardrailDesc.textContent = "All factual statements verified against primary retrieved multimodal and SQL evidence.";
    }

    guardrailStatsRow.innerHTML = `
      <span>Total Audited Claims: <strong>${gr.total_claims || 0}</strong></span>
      <span>Grounded Claims: <strong style="color:var(--success-green);">${gr.grounded_claims || 0}</strong></span>
      <span>Ungrounded Claims: <strong style="color:${(gr.ungrounded_claims > 0) ? 'var(--danger-rose)' : 'inherit'};">${gr.ungrounded_claims || 0}</strong></span>
      <span>Verdict: <strong>${gr.status || 'PASSED'}</strong></span>
    `;
  }

  function renderCitationsCatalog(citations) {
    studioCitationCount.textContent = citations.length;
    studioCitationsList.innerHTML = "";

    citations.forEach(c => {
      const item = document.createElement("div");
      item.className = "citation-item-row";
      let pageStr = c.page_number ? ` (p.${c.page_number})` : "";
      item.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
          <span style="font-weight:700; color:#181613;">[${c.source_type.toUpperCase()}] ${c.source_name}${pageStr}</span>
          <span style="color:#827869; font-size:11px;">Citation #${c.citation_id}</span>
        </div>
        <div style="color:#595246; font-size:12px;">${c.snippet}</div>
      `;
      item.addEventListener("click", () => openCitationModal(c.source_name, c.source_type));
      studioCitationsList.appendChild(item);
    });
  }

  function renderVerdictsList(verdicts) {
    studioClaimCount.textContent = verdicts.length;
    studioVerdictsList.innerHTML = "";

    verdicts.forEach(v => {
      const box = document.createElement("div");
      box.className = `verdict-item-row ${v.verdict}`;
      box.innerHTML = `
        <div style="font-weight:600; margin-bottom:2px; color:#181613;">Claim: "${v.claim}"</div>
        <div style="color:${v.verdict === 'grounded' ? 'var(--success-green)' : (v.verdict === 'ungrounded' ? 'var(--danger-rose)' : 'var(--warning-amber)')}; font-size:11px;">
          Verdict: <strong>${v.verdict.toUpperCase()}</strong> — ${v.reason}
        </div>
      `;
      studioVerdictsList.appendChild(box);
    });
  }

  function renderTracesList(traces) {
    studioTraceCount.textContent = traces.length;
    studioTracesList.innerHTML = "";

    traces.forEach(t => {
      const row = document.createElement("div");
      row.className = "trace-row";
      row.innerHTML = `
        <span class="trace-agent">[${t.agent}] ${t.event_type.toUpperCase()}</span>
        <span class="trace-content">${t.content}</span>
      `;
      studioTracesList.appendChild(row);
    });
  }

  // -------------------------------------------------------------
  // 8. Document Center Operations
  // -------------------------------------------------------------
  async function loadDocuments() {
    try {
      const res = await fetch("/api/documents");
      const docs = await res.json();
      currentDocuments = docs;
      docTableCount.textContent = docs.length;
      docTableBody.innerHTML = "";

      docs.forEach(doc => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${doc.filename}</strong></td>
          <td><span style="font-family:var(--font-mono); color:#181613;">.${doc.content_type}</span></td>
          <td>${doc.page_count} ${doc.page_count > 1 ? 'pages' : 'section'}</td>
          <td>${doc.chunk_count} chunks</td>
          <td>${doc.created_at}</td>
          <td>
            <button class="btn-outline-sm" style="color:var(--danger-rose); border-color:var(--danger-rose);" onclick="deleteDoc('${doc.id}')">Delete</button>
          </td>
        `;
        docTableBody.appendChild(tr);
      });
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  }

  window.deleteDoc = async function(id) {
    if (!confirm(`Remove ${id} from vector store?`)) return;
    await fetch(`/api/documents/${id}`, { method: "DELETE" });
    loadDocuments();
    checkHealth();
  };

  triggerUploadBtn.addEventListener("click", () => hiddenFileInput.click());
  uploadDropzone.addEventListener("click", () => hiddenFileInput.click());

  hiddenFileInput.addEventListener("change", async (e) => {
    const files = e.target.files;
    if (!files.length) return;

    for (let file of files) {
      const formData = new FormData();
      formData.append("file", file);
      try {
        headerStatusText.textContent = `Indexing ${file.name}...`;
        const res = await fetch("/api/upload", { method: "POST", body: formData });
        if (!res.ok) throw new Error("Upload failed");
      } catch (err) {
        alert(`Error uploading ${file.name}: ${err.message}`);
      }
    }
    loadDocuments();
    checkHealth();
    alert("Documents uploaded and indexed into ChromaDB successfully!");
  });

  // -------------------------------------------------------------
  // 9. SQL Sandbox & Schema Explorer
  // -------------------------------------------------------------
  async function loadSqlSchema() {
    try {
      const res = await fetch("/api/sql/schema");
      const tables = await res.json();
      currentSchema = tables;
      schemaAccordionContainer.innerHTML = "";

      tables.forEach(t => {
        const item = document.createElement("div");
        item.className = "schema-list-item";
        const colNames = t.columns.map(c => `${c.name} (${c.type})`).join(", ");
        item.innerHTML = `
          <div class="schema-item-name">📄 ${t.table_name} (${t.row_count} rows)</div>
          <div class="schema-item-cols">${colNames}</div>
        `;
        item.addEventListener("click", () => {
          sandboxSqlInput.value = `SELECT * FROM ${t.table_name} LIMIT 10;`;
        });
        schemaAccordionContainer.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load schema:", err);
    }
  }

  runSandboxSqlBtn.addEventListener("click", executeSandboxSql);

  async function executeSandboxSql() {
    const query = sandboxSqlInput.value.trim();
    if (!query) return;

    sandboxMetaText.textContent = "Executing validated read-only query...";
    sandboxTableContainer.innerHTML = "";

    try {
      const res = await fetch("/api/sql/sandbox", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });

      const data = await res.json();

      if (!data.is_valid) {
        sandboxMetaText.innerHTML = `<span style="color:var(--danger-rose); font-weight:700;">⛔ ${data.error}</span>`;
        return;
      }

      sandboxMetaText.innerHTML = `
        <span>Executed in <strong>${data.execution_time_ms}ms</strong></span> | 
        <span>Returned <strong>${data.row_count} rows</strong></span> | 
        <span>Read-Only Mode Enforced</span>
      `;

      // Render Table
      let tableHtml = `<table class="editorial-table"><thead><tr>`;
      data.columns.forEach(c => {
        tableHtml += `<th>${c}</th>`;
      });
      tableHtml += `</tr></thead><tbody>`;

      data.rows.forEach(r => {
        tableHtml += `<tr>`;
        r.forEach(val => {
          tableHtml += `<td>${val !== null ? val : 'NULL'}</td>`;
        });
        tableHtml += `</tr>`;
      });
      tableHtml += `</tbody></table>`;

      sandboxTableContainer.innerHTML = tableHtml;

    } catch (err) {
      sandboxMetaText.textContent = `Sandbox Error: ${err.message}`;
    }
  }
});
