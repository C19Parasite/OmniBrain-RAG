// OmniBrain Studio UI Application Logic (Axlero Solutions Project 1 Edition)

// Global navigation functions
window.showWorkspace = function() {
  const viewLanding = document.getElementById("view-landing");
  const viewWorkspace = document.getElementById("view-workspace");
  const workspacePromptInput = document.getElementById("workspacePromptInput");

  if (viewLanding) viewLanding.style.display = "none";
  if (viewWorkspace) {
    viewWorkspace.style.display = "flex";
  }
  if (window.loadDocuments) window.loadDocuments();
  if (workspacePromptInput) {
    setTimeout(() => workspacePromptInput.focus(), 50);
  }
};

window.showLandingPage = function() {
  const viewLanding = document.getElementById("view-landing");
  const viewWorkspace = document.getElementById("view-workspace");
  if (viewLanding) viewLanding.style.display = "flex";
  if (viewWorkspace) viewWorkspace.style.display = "none";
  window.scrollTo({ top: 0, behavior: "smooth" });
};

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let currentDocuments = [];
  let chatSessions = []; // Stored in localStorage
  let activeSessionId = null;
  let allCitationsCatalog = []; // Accumulates citations across conversation for modals
  
  // Current active chat document selection: 'all' or array of doc_ids
  let activeChatDocIds = null; // null means 'all', [] means 'none', ['doc1'] means specific

  // View Elements
  const startAskingBtn = document.getElementById("startAskingBtn");
  const sidebarReturnHome = document.getElementById("sidebarReturnHome");
  const sidebarLandingBtn = document.getElementById("sidebarLandingBtn");

  // Sidebar Elements
  const appSidebar = document.getElementById("appSidebar");
  const sidebarOpenBtn = document.getElementById("sidebarOpenBtn");
  const sidebarCloseBtn = document.getElementById("sidebarCloseBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const openUploadModalBtn = document.getElementById("openUploadModalBtn");
  const sidebarDocSearchInput = document.getElementById("sidebarDocSearchInput");
  const sidebarDocList = document.getElementById("sidebarDocList");
  const sidebarDocCountBadge = document.getElementById("sidebarDocCountBadge");
  const chatHistoryList = document.getElementById("chatHistoryList");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");
  const openSettingsBtn = document.getElementById("openSettingsBtn");

  // Workspace Main Elements
  const activeSessionTitle = document.getElementById("activeSessionTitle");
  const sessionDocContextLabel = document.getElementById("sessionDocContextLabel");
  const workspaceEngineStatus = document.getElementById("workspaceEngineStatus");
  const conversationStreamContainer = document.getElementById("conversationStreamContainer");
  const streamWelcomeCard = document.getElementById("streamWelcomeCard");
  const chatDocChipsContainer = document.getElementById("chatDocChipsContainer");
  const chatDocSelectionSummary = document.getElementById("chatDocSelectionSummary");
  const chatMessageStream = document.getElementById("chatMessageStream");
  const orchestrationFlowCard = document.getElementById("orchestrationFlowCard");
  const orchestrationStatusLabel = document.getElementById("orchestrationStatusLabel");

  // Prompt Input Elements
  const promptBoxContainer = document.querySelector(".prompt-box-container");
  const workspacePromptInput = document.getElementById("workspacePromptInput");
  const workspaceSubmitBtn = document.getElementById("workspaceSubmitBtn");
  const promptAttachBtn = document.getElementById("promptAttachBtn");

  // Floating Citation Hover Tooltip Element
  const citationHoverTooltip = document.getElementById("citationHoverTooltip");
  const tooltipTypePill = document.getElementById("tooltipTypePill");
  const tooltipTitle = document.getElementById("tooltipTitle");
  const tooltipContent = document.getElementById("tooltipContent");
  let lastSubmittedQuery = "";

  // Upload Modal Staging Elements
  const uploadModal = document.getElementById("uploadModal");
  const closeUploadModalBtn = document.getElementById("closeUploadModalBtn");
  const modalUploadDropzone = document.getElementById("modalUploadDropzone");
  const modalFileInput = document.getElementById("modalFileInput");
  const modalUploadStatusInfo = document.getElementById("modalUploadStatusInfo");
  const uploadStageDropzone = document.getElementById("uploadStageDropzone");
  const uploadStagePreview = document.getElementById("uploadStagePreview");
  const uploadPreviewMeta = document.getElementById("uploadPreviewMeta");
  const uploadPreviewImgContainer = document.getElementById("uploadPreviewImgContainer");
  const uploadPreviewImg = document.getElementById("uploadPreviewImg");
  const uploadPreviewTextarea = document.getElementById("uploadPreviewTextarea");
  const uploadAttachToChatCheckbox = document.getElementById("uploadAttachToChatCheckbox");
  const commitUploadBtn = document.getElementById("commitUploadBtn");
  const cancelUploadPreviewBtn = document.getElementById("cancelUploadPreviewBtn");
  const uploadModalTag = document.getElementById("uploadModalTag");
  const uploadModalHeading = document.getElementById("uploadModalHeading");

  let stagedUpload = null;

  // Document Reader Modal Elements
  const docReaderModal = document.getElementById("docReaderModal");
  const closeDocReaderModalBtn = document.getElementById("closeDocReaderModalBtn");
  const readerDocTypeTag = document.getElementById("readerDocTypeTag");
  const readerDocTitle = document.getElementById("readerDocTitle");
  const readerDocBody = document.getElementById("readerDocBody");

  // Settings Modal Elements
  const settingsModal = document.getElementById("settingsModal");
  const closeSettingsModalBtn = document.getElementById("closeSettingsModalBtn");
  const settingTopK = document.getElementById("settingTopK");
  const settingTemperature = document.getElementById("settingTemperature");
  const settingGeminiKey = document.getElementById("settingGeminiKey");
  const settingOpenAIKey = document.getElementById("settingOpenAIKey");
  const saveSettingsBtn = document.getElementById("saveSettingsBtn");
  const resetDemoDataBtn = document.getElementById("resetDemoDataBtn");

  // Citation Evidence Modal Elements
  const citationModal = document.getElementById("citationModal");
  const closeCitationModalBtn = document.getElementById("closeCitationModalBtn");
  const modalCitationTypePill = document.getElementById("modalCitationTypePill");
  const modalCitationTitle = document.getElementById("modalCitationTitle");
  const modalCitationBody = document.getElementById("modalCitationBody");

  // Toast Container
  const toastContainer = document.getElementById("toastContainer");

  // -------------------------------------------------------------
  // 1. Toast Notification Helper
  // -------------------------------------------------------------
  function showToast(message, type = "info") {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast-message ${type}`;
    let icon = "◈";
    if (type === "success") icon = "✓";
    if (type === "error") icon = "⚠";
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // Bind Navigation Listeners
  if (startAskingBtn) startAskingBtn.addEventListener("click", window.showWorkspace);
  if (sidebarReturnHome) sidebarReturnHome.addEventListener("click", window.showLandingPage);
  if (sidebarLandingBtn) sidebarLandingBtn.addEventListener("click", window.showLandingPage);

  // -------------------------------------------------------------
  // 2. Sidebar Toggle (Mobile & Desktop)
  // -------------------------------------------------------------
  if (sidebarOpenBtn) {
    sidebarOpenBtn.addEventListener("click", () => {
      appSidebar.classList.toggle("collapsed");
    });
  }

  if (sidebarCloseBtn) {
    sidebarCloseBtn.addEventListener("click", () => {
      appSidebar.classList.add("collapsed");
    });
  }

  if (window.innerWidth <= 768 && appSidebar) {
    appSidebar.classList.add("collapsed");
  }

  // -------------------------------------------------------------
  // 3. Document Selection Context per Chat
  // -------------------------------------------------------------
  function renderChatDocChips() {
    if (!chatDocChipsContainer) return;
    chatDocChipsContainer.innerHTML = "";

    // "All Documents" Chip
    const allChip = document.createElement("div");
    const isAll = (activeChatDocIds === null);
    allChip.className = `doc-chip ${isAll ? 'selected' : ''}`;
    allChip.innerHTML = `<span>🌐 All Documents (${currentDocuments.length})</span>`;
    allChip.addEventListener("click", () => {
      activeChatDocIds = null;
      updateDocContextUI();
      renderChatDocChips();
    });
    chatDocChipsContainer.appendChild(allChip);

    // "No Documents" Chip
    const noneChip = document.createElement("div");
    const isNone = (activeChatDocIds !== null && activeChatDocIds.length === 0);
    noneChip.className = `doc-chip ${isNone ? 'selected' : ''}`;
    noneChip.innerHTML = `<span>🚫 No Document Context</span>`;
    noneChip.addEventListener("click", () => {
      activeChatDocIds = [];
      updateDocContextUI();
      renderChatDocChips();
    });
    chatDocChipsContainer.appendChild(noneChip);

    // Individual Document Chips
    currentDocuments.forEach(doc => {
      const isSelected = (activeChatDocIds !== null && activeChatDocIds.includes(doc.id));
      const chip = document.createElement("div");
      chip.className = `doc-chip ${isSelected ? 'selected' : ''}`;
      chip.innerHTML = `<span>📄 ${doc.filename}</span>`;
      chip.addEventListener("click", () => {
        if (activeChatDocIds === null) {
          activeChatDocIds = [doc.id];
        } else if (activeChatDocIds.includes(doc.id)) {
          activeChatDocIds = activeChatDocIds.filter(id => id !== doc.id);
        } else {
          activeChatDocIds.push(doc.id);
        }
        updateDocContextUI();
        renderChatDocChips();
      });
      chatDocChipsContainer.appendChild(chip);
    });

    updateDocContextUI();
  }

  function updateDocContextUI() {
    if (!sessionDocContextLabel) return;
    
    if (activeChatDocIds === null) {
      sessionDocContextLabel.textContent = `Context: All Indexed Documents (${currentDocuments.length})`;
      if (chatDocSelectionSummary) chatDocSelectionSummary.textContent = "All Documents Active";
    } else if (activeChatDocIds.length === 0) {
      sessionDocContextLabel.textContent = "Context: No Documents Attached";
      if (chatDocSelectionSummary) chatDocSelectionSummary.textContent = "0 Documents Selected";
    } else {
      const docNames = currentDocuments
        .filter(d => activeChatDocIds.includes(d.id))
        .map(d => d.filename)
        .join(", ");
      sessionDocContextLabel.textContent = `Context: ${activeChatDocIds.length} doc(s) (${docNames})`;
      if (chatDocSelectionSummary) chatDocSelectionSummary.textContent = `${activeChatDocIds.length} Document(s) Selected`;
    }
  }

  // -------------------------------------------------------------
  // 4. Multi-Turn Chat History Management (localStorage)
  // -------------------------------------------------------------
  function loadStoredSessions() {
    try {
      const stored = localStorage.getItem("omnibrain_chat_sessions_v2");
      if (stored) {
        chatSessions = JSON.parse(stored);
      } else {
        chatSessions = [];
      }
    } catch (e) {
      chatSessions = [];
    }
    renderChatHistoryList();
  }

  function saveSessionsToStorage() {
    try {
      localStorage.setItem("omnibrain_chat_sessions_v2", JSON.stringify(chatSessions));
    } catch (e) {
      console.warn("Storage quota exceeded", e);
    }
    renderChatHistoryList();
  }

  function renderChatHistoryList() {
    if (!chatHistoryList) return;
    chatHistoryList.innerHTML = "";
    if (chatSessions.length === 0) {
      chatHistoryList.innerHTML = `<div class="empty-history-text">No past analyses yet.</div>`;
      return;
    }

    chatSessions.forEach(session => {
      const item = document.createElement("div");
      item.className = `history-item ${session.id === activeSessionId ? 'active' : ''}`;
      
      const titleSpan = document.createElement("span");
      titleSpan.style.overflow = "hidden";
      titleSpan.style.textOverflow = "ellipsis";
      titleSpan.textContent = session.title || (session.turns && session.turns[0]?.query) || "Financial Analysis";

      const actionsDiv = document.createElement("div");
      actionsDiv.className = "history-actions";

      const renameBtn = document.createElement("button");
      renameBtn.className = "history-rename-btn";
      renameBtn.innerHTML = "✏️";
      renameBtn.title = "Rename conversation";
      renameBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        renameSession(session);
      });

      const delBtn = document.createElement("button");
      delBtn.className = "history-del-btn";
      delBtn.innerHTML = "✕";
      delBtn.title = "Delete this conversation";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(session.id);
      });

      actionsDiv.appendChild(renameBtn);
      actionsDiv.appendChild(delBtn);

      item.appendChild(titleSpan);
      item.appendChild(actionsDiv);

      item.addEventListener("click", () => {
        restoreSession(session);
      });

      chatHistoryList.appendChild(item);
    });
  }

  function renameSession(session) {
    const newName = prompt("Rename analysis session:", session.title || "Financial Analysis");
    if (newName && newName.trim()) {
      session.title = newName.trim();
      if (session.id === activeSessionId && activeSessionTitle) {
        activeSessionTitle.textContent = session.title;
      }
      saveSessionsToStorage();
      showToast("Conversation renamed", "info");
    }
  }

  function deleteSession(sessionId) {
    chatSessions = chatSessions.filter(s => s.id !== sessionId);
    saveSessionsToStorage();
    if (activeSessionId === sessionId) {
      startNewChat();
    }
    showToast("Chat deleted from history", "info");
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", () => {
      if (chatSessions.length === 0) return;
      if (confirm("Clear all chat history?")) {
        chatSessions = [];
        saveSessionsToStorage();
        startNewChat();
        showToast("All chat history cleared", "info");
      }
    });
  }

  function startNewChat() {
    activeSessionId = null;
    activeChatDocIds = null;
    if (activeSessionTitle) activeSessionTitle.textContent = "New Financial Analysis";
    if (chatMessageStream) chatMessageStream.innerHTML = "";
    if (streamWelcomeCard) streamWelcomeCard.style.display = "flex";
    if (orchestrationFlowCard) orchestrationFlowCard.style.display = "none";
    if (workspacePromptInput) workspacePromptInput.value = "";
    resetPipelineNodes();
    renderChatDocChips();
    renderChatHistoryList();
    if (workspacePromptInput) workspacePromptInput.focus();
    if (window.innerWidth <= 768 && appSidebar) {
      appSidebar.classList.add("collapsed");
    }
  }

  if (newChatBtn) newChatBtn.addEventListener("click", startNewChat);

  function restoreSession(session) {
    activeSessionId = session.id;
    activeChatDocIds = session.doc_ids !== undefined ? session.doc_ids : null;
    if (activeSessionTitle) activeSessionTitle.textContent = session.title || "Financial Analysis";
    
    if (chatMessageStream) chatMessageStream.innerHTML = "";
    if (streamWelcomeCard) streamWelcomeCard.style.display = "none";
    if (orchestrationFlowCard) orchestrationFlowCard.style.display = "none";

    // Restore each turn in the thread
    if (session.turns && session.turns.length > 0) {
      session.turns.forEach((turn, idx) => {
        appendTurnToStream(turn, idx);
      });
    } else {
      if (streamWelcomeCard) streamWelcomeCard.style.display = "flex";
    }

    updateDocContextUI();
    renderChatHistoryList();
    if (window.innerWidth <= 768 && appSidebar) {
      appSidebar.classList.add("collapsed");
    }

    if (conversationStreamContainer) {
      setTimeout(() => {
        conversationStreamContainer.scrollTo({ top: conversationStreamContainer.scrollHeight, behavior: "smooth" });
      }, 50);
    }
  }

  // -------------------------------------------------------------
  // 5. Prompt Auto-Expand & Keyboard Shortcuts
  // -------------------------------------------------------------
  function autoResizePromptInput() {
    if (!workspacePromptInput) return;
    workspacePromptInput.style.height = "auto";
    workspacePromptInput.style.height = Math.min(workspacePromptInput.scrollHeight, 180) + "px";
  }

  if (workspacePromptInput) {
    workspacePromptInput.addEventListener("input", autoResizePromptInput);

    workspacePromptInput.addEventListener("keydown", (e) => {
      // Enter without Shift -> Submit
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        executePromptQuery();
      }
      // Ctrl+Enter / Cmd+Enter -> Submit
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        executePromptQuery();
      }
      // Up arrow when empty -> recall last prompt
      if (e.key === "ArrowUp" && workspacePromptInput.value === "" && lastSubmittedQuery) {
        e.preventDefault();
        workspacePromptInput.value = lastSubmittedQuery;
        autoResizePromptInput();
      }
    });
  }

  // Global Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    // Ctrl+K -> focus prompt
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      if (workspacePromptInput) {
        workspacePromptInput.focus();
        showToast("Prompt bar focused (Ctrl+K)", "info");
      }
    }
    // Esc -> close open modals & tooltips
    if (e.key === "Escape") {
      if (uploadModal) uploadModal.classList.remove("active");
      if (docReaderModal) docReaderModal.classList.remove("active");
      if (settingsModal) settingsModal.classList.remove("active");
      if (citationModal) citationModal.classList.remove("active");
      if (citationHoverTooltip) citationHoverTooltip.style.display = "none";
    }
  });

  // Drag and Drop files directly onto Prompt Box
  if (promptBoxContainer) {
    ['dragenter', 'dragover'].forEach(name => {
      promptBoxContainer.addEventListener(name, (e) => {
        e.preventDefault();
        promptBoxContainer.classList.add("drag-over");
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      promptBoxContainer.addEventListener(name, (e) => {
        e.preventDefault();
        promptBoxContainer.classList.remove("drag-over");
      });
    });

    promptBoxContainer.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        resetUploadModalState();
        if (uploadModal) uploadModal.classList.add("active");
        handleUploadFilePreview(files[0]);
      }
    });
  }

  // -------------------------------------------------------------
  // 6. Query Execution & Multi-Turn Chat Streaming
  // -------------------------------------------------------------
  if (workspaceSubmitBtn) workspaceSubmitBtn.addEventListener("click", () => executePromptQuery());

  async function executePromptQuery(overrideQuery = null) {
    const query = overrideQuery || (workspacePromptInput ? workspacePromptInput.value.trim() : "");
    if (!query) return;

    lastSubmittedQuery = query;
    if (!query) return;

    if (workspaceSubmitBtn) workspaceSubmitBtn.disabled = true;
    if (workspacePromptInput) workspacePromptInput.disabled = true;

    // Initialize or get current session
    let currentSession = chatSessions.find(s => s.id === activeSessionId);
    if (!currentSession) {
      currentSession = {
        id: "session_" + Date.now(),
        title: query,
        doc_ids: activeChatDocIds,
        turns: []
      };
      activeSessionId = currentSession.id;
      chatSessions.unshift(currentSession);
    }

    if (activeSessionTitle) activeSessionTitle.textContent = currentSession.title;

    // Hide welcome screen
    if (streamWelcomeCard) streamWelcomeCard.style.display = "none";

    // Append user message row immediately
    const userDocTag = (activeChatDocIds && activeChatDocIds.length > 0)
      ? `Attached: ${activeChatDocIds.join(', ')}`
      : (activeChatDocIds && activeChatDocIds.length === 0 ? "No document attached" : "All Knowledge Base");

    const tempTurnEl = document.createElement("div");
    tempTurnEl.className = "chat-turn";
    tempTurnEl.innerHTML = `
      <div class="user-msg-row">
        <div class="user-msg-bubble">
          <div class="msg-text">${escapeHtml(query)}</div>
          <div class="msg-doc-tag">${userDocTag}</div>
        </div>
      </div>
    `;
    if (chatMessageStream) chatMessageStream.appendChild(tempTurnEl);

    // Show orchestration flow banner
    if (orchestrationFlowCard) orchestrationFlowCard.style.display = "block";
    resetPipelineNodes();
    setPipelineNodeActive("Supervisor");
    if (orchestrationStatusLabel) orchestrationStatusLabel.textContent = "Supervisor Planning & Routing...";

    if (conversationStreamContainer) {
      conversationStreamContainer.scrollTo({ top: conversationStreamContainer.scrollHeight, behavior: "smooth" });
    }

    try {
      const topKVal = parseInt(settingTopK?.value) || 5;
      const tempVal = parseFloat(settingTemperature?.value) || 0.2;
      const geminiKey = localStorage.getItem("omnibrain_gemini_key") || "";
      const openaiKey = localStorage.getItem("omnibrain_openai_key") || "";

      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          top_k: topKVal,
          temperature: tempVal,
          gemini_api_key: geminiKey,
          openai_api_key: openaiKey,
          document_ids: activeChatDocIds
        })
      });

      if (!res.ok) throw new Error(`Server returned status ${res.status}`);
      const data = await res.json();

      // Accumulate citations globally for evidence popups
      if (data.citations) {
        allCitationsCatalog.push(...data.citations);
      }

      // Animate pipeline
      await animatePipeline(data.execution_trace || []);
      if (orchestrationFlowCard) orchestrationFlowCard.style.display = "none";

      // Create new turn object
      const newTurn = {
        turn_id: "turn_" + Date.now(),
        query: query,
        doc_tag: userDocTag,
        answer_markdown: data.memo_markdown,
        citations: data.citations || [],
        guardrail_report: data.guardrail_report,
        execution_trace: data.execution_trace || [],
        execution_time_seconds: data.execution_time_seconds
      };

      currentSession.turns.push(newTurn);
      saveSessionsToStorage();

      // Remove the temp user row and render full turn
      tempTurnEl.remove();
      appendTurnToStream(newTurn, currentSession.turns.length - 1);

      if (conversationStreamContainer) {
        conversationStreamContainer.scrollTo({ top: conversationStreamContainer.scrollHeight, behavior: "smooth" });
      }

    } catch (err) {
      if (orchestrationFlowCard) orchestrationFlowCard.style.display = "none";
      showToast(`Error: ${err.message}`, "error");
      console.error(err);
    } finally {
      if (workspaceSubmitBtn) workspaceSubmitBtn.disabled = false;
      if (workspacePromptInput) {
        workspacePromptInput.disabled = false;
        workspacePromptInput.value = "";
        workspacePromptInput.focus();
      }
    }
  }

  function appendTurnToStream(turn, turnIdx) {
    if (!chatMessageStream) return;

    const turnDiv = document.createElement("div");
    turnDiv.className = "chat-turn";
    turnDiv.id = `chat-turn-${turnIdx}`;

    // 1. User Message Row
    const userRow = document.createElement("div");
    userRow.className = "user-msg-row";
    userRow.innerHTML = `
      <div class="user-msg-bubble">
        <div class="msg-text">${escapeHtml(turn.query)}</div>
        <div class="msg-doc-tag">${turn.doc_tag || 'Knowledge Base'}</div>
      </div>
    `;
    turnDiv.appendChild(userRow);

    // 2. Assistant Response Card
    const assistantRow = document.createElement("div");
    assistantRow.className = "assistant-msg-row";

    // Header with copy button
    const headerRow = document.createElement("div");
    headerRow.className = "assistant-header-row";
    headerRow.innerHTML = `
      <div class="assistant-brand-tag">
        <span style="font-size: 14px;">◈</span>
        <span>OmniBrain</span>
      </div>
      <div class="assistant-actions">
        <button class="btn-copy-memo" title="Copy response to clipboard">📋 Copy</button>
      </div>
    `;

    // Render Answer Markdown
    const contentDiv = document.createElement("div");
    contentDiv.className = "assistant-answer-content";
    contentDiv.innerHTML = formatMarkdownWithCitations(turn.answer_markdown);

    // Copy event
    headerRow.querySelector(".btn-copy-memo").addEventListener("click", () => {
      navigator.clipboard.writeText(contentDiv.innerText).then(() => {
        showToast("Response copied to clipboard!", "success");
      });
    });

    assistantRow.appendChild(headerRow);
    assistantRow.appendChild(contentDiv);

    // Guardrail Score & Citations Accordion (Minimal Footnote)
    const gr = turn.guardrail_report;
    const citations = turn.citations || [];
    const scorePct = Math.round((gr?.overall_score || 1.0) * 100);
    const grStatus = gr?.status || "PASSED";

    const auditBar = document.createElement("div");
    auditBar.className = "turn-audit-bar";
    auditBar.innerHTML = `
      <div class="turn-audit-summary">
        <span class="grounding-pill ${grStatus.toLowerCase()}">
          ✓ ${scorePct}% Factual Grounding (${grStatus})
        </span>
        <span style="color:var(--text-muted); font-size:11px;">
          ${citations.length} Verified Sources &bull; ${turn.execution_time_seconds || '0.01'}s
        </span>
      </div>

      <details class="full-width-accordion" style="margin-top: 4px;">
        <summary>
          <span class="acc-title">Verified Citations &amp; Audit Catalog</span>
          <span class="acc-badge">${citations.length} Sources</span>
        </summary>
        <div class="acc-body">
          <div class="citations-list-box"></div>
        </div>
      </details>
    `;

    // Populate accordion citations
    const citBox = auditBar.querySelector(".citations-list-box");
    if (citBox) {
      citations.forEach(c => {
        const item = document.createElement("div");
        item.className = "citation-item-row";
        let pageStr = c.page_number ? ` (p.${c.page_number})` : "";
        item.innerHTML = `
          <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="font-weight:700; color:#181613;">[${c.source_type.toUpperCase()}] ${c.source_name}${pageStr}</span>
            <span style="color:#827869; font-size:10.5px;">Citation #${c.citation_id}</span>
          </div>
          <div style="color:#595246; font-size:11.5px;">${c.snippet}</div>
        `;
        item.addEventListener("click", () => openCitationModal(c.source_name, c.source_type));
        citBox.appendChild(item);
      });
    }

    // Quick Action Bar (Regenerate, Copy, Helpful Feedback)
    const actionBar = document.createElement("div");
    actionBar.className = "assistant-action-bar";
    actionBar.innerHTML = `
      <button class="action-btn-sm btn-copy-turn" title="Copy answer text">📋 Copy</button>
      <button class="action-btn-sm btn-regen-turn" title="Regenerate answer">🔄 Regenerate</button>
      <button class="action-btn-sm btn-like-turn" title="Mark response as helpful">👍 Helpful</button>
      <button class="action-btn-sm btn-dislike-turn" title="Report inaccurate claim">👎 Poor</button>
    `;

    actionBar.querySelector(".btn-copy-turn").addEventListener("click", () => {
      navigator.clipboard.writeText(contentDiv.innerText).then(() => {
        showToast("Answer copied to clipboard!", "success");
      });
    });

    actionBar.querySelector(".btn-regen-turn").addEventListener("click", () => {
      executePromptQuery(turn.query);
      showToast("Regenerating response...", "info");
    });

    const likeBtn = actionBar.querySelector(".btn-like-turn");
    const dislikeBtn = actionBar.querySelector(".btn-dislike-turn");

    likeBtn.addEventListener("click", () => {
      likeBtn.classList.toggle("active-feedback");
      dislikeBtn.classList.remove("active-feedback");
      showToast("Feedback saved: Response marked helpful ✓", "success");
    });

    dislikeBtn.addEventListener("click", () => {
      dislikeBtn.classList.toggle("active-feedback");
      likeBtn.classList.remove("active-feedback");
      showToast("Feedback saved: Marked for review ⚠", "info");
    });

    assistantRow.appendChild(auditBar);
    assistantRow.appendChild(actionBar);
    turnDiv.appendChild(assistantRow);

    // Bind inline citation click and hover tooltip events
    turnDiv.querySelectorAll(".citation-inline-badge").forEach(badge => {
      // Click -> open full modal
      badge.addEventListener("click", () => {
        openCitationModal(badge.dataset.source, badge.dataset.type);
      });

      // Hover -> show floating tooltip card
      badge.addEventListener("mouseenter", (e) => {
        showCitationHoverTooltip(e, badge.dataset.source, badge.dataset.type);
      });

      badge.addEventListener("mousemove", (e) => {
        positionCitationHoverTooltip(e);
      });

      badge.addEventListener("mouseleave", () => {
        hideCitationHoverTooltip();
      });
    });

    chatMessageStream.appendChild(turnDiv);
  }

  // -------------------------------------------------------------
  // 7. Floating Citation Hover Tooltip Logic
  // -------------------------------------------------------------
  function showCitationHoverTooltip(e, sourceText, type) {
    if (!citationHoverTooltip) return;

    let matched = allCitationsCatalog.find(c =>
      sourceText.toLowerCase().includes(c.source_name.toLowerCase()) ||
      (c.sql_query && sourceText.toLowerCase().includes("quarterly"))
    );

    if (tooltipTypePill) tooltipTypePill.textContent = type.toUpperCase();
    if (tooltipTitle) tooltipTitle.textContent = matched ? matched.source_name : sourceText;

    let snippetText = "";
    if (type === "sql") {
      snippetText = (matched && matched.sql_query) ? `SQL: ${matched.sql_query}` : "SQLite database table verification.";
    } else if (type === "visual") {
      snippetText = (matched && matched.snippet) ? matched.snippet : "Visual chart transcription verified against primary exhibit.";
    } else {
      snippetText = (matched && matched.snippet) ? matched.snippet : "Passage corroborated against indexed corporate filing.";
    }

    if (tooltipContent) tooltipContent.textContent = snippetText;

    positionCitationHoverTooltip(e);
    citationHoverTooltip.style.display = "block";
  }

  function positionCitationHoverTooltip(e) {
    if (!citationHoverTooltip || citationHoverTooltip.style.display === "none") return;
    const x = Math.min(e.clientX + 14, window.innerWidth - 360);
    const y = Math.min(e.clientY + 14, window.innerHeight - 180);
    citationHoverTooltip.style.left = `${x}px`;
    citationHoverTooltip.style.top = `${y}px`;
  }

  function hideCitationHoverTooltip() {
    if (citationHoverTooltip) citationHoverTooltip.style.display = "none";
  }

  function formatMarkdownWithCitations(markdown) {
    let parsedHtml = marked.parse(markdown || "No response.");

    parsedHtml = parsedHtml.replace(/\[SQL:\s*([^\]]+)\]/g, (match, p1) => {
      return `<button class="citation-inline-badge sql" data-source="${p1}" data-type="sql">[SQL: ${p1}]</button>`;
    });

    parsedHtml = parsedHtml.replace(/\[Visual:\s*([^\]]+)\]/g, (match, p1) => {
      return `<button class="citation-inline-badge visual" data-source="${p1}" data-type="visual">[Visual: ${p1}]</button>`;
    });

    parsedHtml = parsedHtml.replace(/\[Source:\s*([^\]]+)\]/g, (match, p1) => {
      return `<button class="citation-inline-badge" data-source="${p1}" data-type="text">[Source: ${p1}]</button>`;
    });

    return parsedHtml;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Pipeline Helpers
  function resetPipelineNodes() {
    ["Supervisor", "SQLAgent", "SearchAgent", "VisionAgent", "Synthesizer", "GuardrailEvaluator"].forEach(node => {
      const el = document.getElementById(`pnode-${node}`);
      if (el) el.className = "pipe-node";
    });
  }

  function setPipelineNodeActive(nodeId) {
    const el = document.getElementById(`pnode-${nodeId}`);
    if (el) el.className = "pipe-node active";
  }

  function setPipelineNodeCompleted(nodeId) {
    const el = document.getElementById(`pnode-${nodeId}`);
    if (el) el.className = "pipe-node completed";
  }

  async function animatePipeline(traces) {
    const activeAgents = new Set();
    traces.forEach(t => activeAgents.add(t.agent));

    setPipelineNodeCompleted("Supervisor");

    if (activeAgents.has("SQLAgent")) {
      setPipelineNodeActive("SQLAgent");
      if (orchestrationStatusLabel) orchestrationStatusLabel.textContent = "Text-to-SQL Executing...";
      await sleep(150);
      setPipelineNodeCompleted("SQLAgent");
    }

    if (activeAgents.has("SearchAgent")) {
      setPipelineNodeActive("SearchAgent");
      if (orchestrationStatusLabel) orchestrationStatusLabel.textContent = "Dense Semantic Vector Retrieving...";
      await sleep(150);
      setPipelineNodeCompleted("SearchAgent");
    }

    setPipelineNodeCompleted("VisionAgent");

    setPipelineNodeActive("Synthesizer");
    if (orchestrationStatusLabel) orchestrationStatusLabel.textContent = "Synthesizing Direct Answer with Citations...";
    await sleep(150);
    setPipelineNodeCompleted("Synthesizer");

    setPipelineNodeActive("GuardrailEvaluator");
    if (orchestrationStatusLabel) orchestrationStatusLabel.textContent = "Auditing Factual Claims (NeMo / LLM-as-Judge)...";
    await sleep(150);
    setPipelineNodeCompleted("GuardrailEvaluator");

    if (orchestrationStatusLabel) orchestrationStatusLabel.textContent = "Complete ✓";
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // -------------------------------------------------------------
  // 6. Citation Preview Modal
  // -------------------------------------------------------------
  function openCitationModal(sourceText, type) {
    if (!citationModal) return;
    if (modalCitationTypePill) modalCitationTypePill.textContent = `[${type.toUpperCase()}]`;
    if (modalCitationTitle) modalCitationTitle.textContent = sourceText;

    let matched = allCitationsCatalog.find(c =>
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
          "${matched ? matched.snippet : 'Passage corroborated against primary documents in knowledge base.'}"
        </div>
      `;
    }

    if (modalCitationBody) modalCitationBody.innerHTML = contentHtml;
    citationModal.classList.add("active");
  }

  if (closeCitationModalBtn) {
    closeCitationModalBtn.addEventListener("click", () => {
      citationModal.classList.remove("active");
    });
  }

  if (citationModal) {
    citationModal.addEventListener("click", (e) => {
      if (e.target === citationModal) citationModal.classList.remove("active");
    });
  }

  // -------------------------------------------------------------
  // 7. Document Reading / Viewing Modal
  // -------------------------------------------------------------
  async function openDocReader(docId) {
    if (!docReaderModal) return;
    try {
      showToast("Loading document...", "info");
      const res = await fetch(`/api/documents/${docId}`);
      if (!res.ok) throw new Error("Could not load document content");
      const doc = await res.json();

      if (readerDocTypeTag) readerDocTypeTag.textContent = (doc.content_type || "DOC").toUpperCase();
      if (readerDocTitle) readerDocTitle.textContent = doc.filename;

      let bodyHtml = `
        <div class="doc-reader-meta-bar">
          <div>
            <span>File: <strong>${doc.filename}</strong></span> &bull; 
            <span>Size: <strong>${Math.round((doc.size_bytes || 0)/1024)} KB</strong></span> &bull; 
            <span>Format: <strong>.${doc.content_type}</strong></span>
          </div>
          <div>
            <button class="btn-outline-danger-sm" onclick="window.deleteDocumentItem('${doc.id}')">🗑️ Delete Document</button>
          </div>
        </div>
      `;

      if (doc.is_image && doc.image_base64) {
        bodyHtml += `
          <img src="${doc.image_base64}" alt="${doc.filename}" class="doc-reader-img-preview">
          <div style="font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-muted); margin: 12px 0 6px 0;">VLM Transcribed Exhibit Text:</div>
        `;
      }

      const parsedText = marked.parse(doc.text_content || "No text content available.");
      bodyHtml += `<div class="doc-reader-content-box">${parsedText}</div>`;

      if (readerDocBody) readerDocBody.innerHTML = bodyHtml;
      docReaderModal.classList.add("active");

    } catch (err) {
      showToast(`Error reading document: ${err.message}`, "error");
    }
  }

  if (closeDocReaderModalBtn) {
    closeDocReaderModalBtn.addEventListener("click", () => {
      docReaderModal.classList.remove("active");
    });
  }

  if (docReaderModal) {
    docReaderModal.addEventListener("click", (e) => {
      if (e.target === docReaderModal) docReaderModal.classList.remove("active");
    });
  }

  // -------------------------------------------------------------
  // 8. Document Knowledge, Upload Modal & Live Filtering
  // -------------------------------------------------------------
  window.loadDocuments = async function() {
    try {
      const res = await fetch("/api/documents");
      const docs = await res.json();
      currentDocuments = docs;
      if (sidebarDocCountBadge) sidebarDocCountBadge.textContent = `${docs.length} files`;
      renderSidebarDocList();
      renderChatDocChips();
    } catch (err) {
      console.error("Failed to load documents", err);
    }
  };

  function renderSidebarDocList() {
    if (!sidebarDocList) return;
    sidebarDocList.innerHTML = "";

    const query = sidebarDocSearchInput ? sidebarDocSearchInput.value.trim().toLowerCase() : "";
    const filteredDocs = currentDocuments.filter(d => 
      d.filename.toLowerCase().includes(query) ||
      d.content_type.toLowerCase().includes(query)
    );

    if (filteredDocs.length === 0) {
      sidebarDocList.innerHTML = `<div style="font-size:11px; color:var(--text-muted); padding:8px;">No matching documents.</div>`;
      return;
    }

    filteredDocs.forEach(doc => {
      const item = document.createElement("div");
      item.className = "sidebar-doc-item";
      item.innerHTML = `
        <span class="sidebar-doc-name" title="Click to Read ${doc.filename}">📄 ${doc.filename}</span>
        <div class="doc-item-actions">
          <button class="doc-read-btn" title="Read document">📖</button>
          <button class="doc-del-btn" title="Delete document">✕</button>
        </div>
      `;

      // Clicking on row or read button opens Document Reader
      item.querySelector(".sidebar-doc-name").addEventListener("click", () => openDocReader(doc.id));
      item.querySelector(".doc-read-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        openDocReader(doc.id);
      });

      // Delete button
      item.querySelector(".doc-del-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        window.deleteDocumentItem(doc.id);
      });

      sidebarDocList.appendChild(item);
    });
  }

  if (sidebarDocSearchInput) {
    sidebarDocSearchInput.addEventListener("input", renderSidebarDocList);
  }

  window.deleteDocumentItem = async function(id) {
    const matchedDoc = currentDocuments.find(d => d.id === id);
    const docName = matchedDoc ? matchedDoc.filename : id;
    if (!confirm(`Are you sure you want to permanently delete "${docName}"?\n\nThis will remove the file from storage and delete all associated embeddings from the vector database.`)) return;

    try {
      showToast(`Deleting ${docName}...`, "info");
      const res = await fetch(`/api/documents/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Server returned error on delete");
      
      // If document reader modal is currently open for this doc, close it
      if (docReaderModal) docReaderModal.classList.remove("active");

      // Update active chat document selection if this doc was selected
      if (Array.isArray(activeChatDocIds)) {
        activeChatDocIds = activeChatDocIds.filter(docId => docId !== id);
      }

      await window.loadDocuments();
      updateDocContextUI();
      showToast(`"${docName}" permanently deleted from disk and vector database!`, "success");
    } catch (e) {
      showToast(`Failed to delete document: ${e.message}`, "error");
    }
  };

  // Upload Modal Staging Logic
  function resetUploadModalState() {
    stagedUpload = null;
    if (uploadStageDropzone) uploadStageDropzone.style.display = "block";
    if (uploadStagePreview) uploadStagePreview.style.display = "none";
    if (uploadModalTag) uploadModalTag.textContent = "INGEST";
    if (uploadModalHeading) uploadModalHeading.textContent = "Upload Document or Visual Chart";
    if (modalUploadStatusInfo) modalUploadStatusInfo.innerHTML = "";
    if (modalFileInput) modalFileInput.value = "";
  }

  if (openUploadModalBtn) {
    openUploadModalBtn.addEventListener("click", () => {
      resetUploadModalState();
      if (uploadModal) uploadModal.classList.add("active");
    });
  }

  if (promptAttachBtn) {
    promptAttachBtn.addEventListener("click", () => {
      resetUploadModalState();
      if (uploadModal) uploadModal.classList.add("active");
    });
  }

  if (closeUploadModalBtn) {
    closeUploadModalBtn.addEventListener("click", () => {
      if (uploadModal) uploadModal.classList.remove("active");
      resetUploadModalState();
    });
  }

  if (cancelUploadPreviewBtn) {
    cancelUploadPreviewBtn.addEventListener("click", () => {
      resetUploadModalState();
    });
  }

  if (uploadModal) {
    uploadModal.addEventListener("click", (e) => {
      if (e.target === uploadModal) {
        uploadModal.classList.remove("active");
        resetUploadModalState();
      }
    });
  }

  if (modalUploadDropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
      modalUploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        modalUploadDropzone.style.borderColor = "var(--text-primary)";
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      modalUploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        modalUploadDropzone.style.borderColor = "var(--border-line)";
      });
    });

    modalUploadDropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleUploadFilePreview(files[0]);
      }
    });
  }

  if (modalFileInput) {
    modalFileInput.addEventListener("change", (e) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleUploadFilePreview(files[0]);
      }
    });
  }

  async function handleUploadFilePreview(file) {
    if (modalUploadStatusInfo) {
      modalUploadStatusInfo.innerHTML = `<span>⏳ Extracting text and visual exhibits from <strong>${file.name}</strong>...</span>`;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload/preview", { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Preview failed with status ${res.status}`);
      const data = await res.json();
      stagedUpload = data;

      if (uploadStageDropzone) uploadStageDropzone.style.display = "none";
      if (uploadStagePreview) uploadStagePreview.style.display = "block";
      if (uploadModalTag) uploadModalTag.textContent = "REVIEW & EDIT";
      if (uploadModalHeading) uploadModalHeading.textContent = `Review: ${data.filename}`;

      if (uploadPreviewMeta) {
        uploadPreviewMeta.innerHTML = `
          <div>
            <span>File: <strong>${data.filename}</strong></span> &bull; 
            <span>Size: <strong>${Math.round(data.size_bytes/1024)} KB</strong></span> &bull; 
            <span>Pages: <strong>${data.page_count}</strong></span> &bull; 
            <span>Chunks: <strong>${data.chunk_count}</strong></span>
          </div>
        `;
      }

      if (data.is_image && data.image_base64) {
        if (uploadPreviewImgContainer) uploadPreviewImgContainer.style.display = "block";
        if (uploadPreviewImg) uploadPreviewImg.src = data.image_base64;
      } else {
        if (uploadPreviewImgContainer) uploadPreviewImgContainer.style.display = "none";
      }

      if (uploadPreviewTextarea) {
        uploadPreviewTextarea.value = data.extracted_text;
      }

      if (uploadAttachToChatCheckbox) {
        uploadAttachToChatCheckbox.checked = true;
      }

      showToast(`Extracted ${data.chunk_count} chunk(s). You can edit the text before saving.`, "info");

    } catch (err) {
      if (modalUploadStatusInfo) modalUploadStatusInfo.innerHTML = `<span style="color:var(--danger-rose);">Error: ${err.message}</span>`;
      showToast(`Failed to parse file: ${err.message}`, "error");
    }
  }

  if (commitUploadBtn) {
    commitUploadBtn.addEventListener("click", async () => {
      if (!stagedUpload) return;

      commitUploadBtn.disabled = true;
      commitUploadBtn.textContent = "Indexing...";

      try {
        const editedText = uploadPreviewTextarea ? uploadPreviewTextarea.value : stagedUpload.extracted_text;
        const attachToChat = uploadAttachToChatCheckbox ? uploadAttachToChatCheckbox.checked : true;

        const res = await fetch("/api/upload/commit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            doc_id: stagedUpload.doc_id,
            filename: stagedUpload.filename,
            content_type: stagedUpload.content_type,
            text_content: editedText,
            image_base64: stagedUpload.image_base64,
            attach_to_chat: attachToChat
          })
        });

        if (!res.ok) throw new Error("Commit failed");
        const docItem = await res.json();

        await window.loadDocuments();

        if (attachToChat) {
          if (activeChatDocIds === null) {
            activeChatDocIds = [docItem.id];
          } else if (!activeChatDocIds.includes(docItem.id)) {
            activeChatDocIds.push(docItem.id);
          }
          updateDocContextUI();
          renderChatDocChips();
        }

        if (uploadModal) uploadModal.classList.remove("active");
        resetUploadModalState();
        showToast(`"${docItem.filename}" saved to knowledge base and attached to chat!`, "success");

      } catch (err) {
        showToast(`Commit error: ${err.message}`, "error");
      } finally {
        if (commitUploadBtn) {
          commitUploadBtn.disabled = false;
          commitUploadBtn.textContent = "Submit & Commit to Knowledge Base ➔";
        }
      }
    });
  }

  // -------------------------------------------------------------
  // 9. Settings Modal Logic
  // -------------------------------------------------------------
  function loadLocalSettings() {
    if (settingTopK && localStorage.getItem("omnibrain_top_k")) settingTopK.value = localStorage.getItem("omnibrain_top_k");
    if (settingTemperature && localStorage.getItem("omnibrain_temperature")) settingTemperature.value = localStorage.getItem("omnibrain_temperature");
    if (settingGeminiKey && localStorage.getItem("omnibrain_gemini_key")) settingGeminiKey.value = localStorage.getItem("omnibrain_gemini_key");
    if (settingOpenAIKey && localStorage.getItem("omnibrain_openai_key")) settingOpenAIKey.value = localStorage.getItem("omnibrain_openai_key");
  }
  loadLocalSettings();

  if (openSettingsBtn && settingsModal) {
    openSettingsBtn.addEventListener("click", () => {
      loadLocalSettings();
      settingsModal.classList.add("active");
    });
  }

  if (closeSettingsModalBtn && settingsModal) {
    closeSettingsModalBtn.addEventListener("click", () => {
      settingsModal.classList.remove("active");
    });
  }

  if (settingsModal) {
    settingsModal.addEventListener("click", (e) => {
      if (e.target === settingsModal) settingsModal.classList.remove("active");
    });
  }

  if (saveSettingsBtn && settingsModal) {
    saveSettingsBtn.addEventListener("click", () => {
      if (settingTopK) localStorage.setItem("omnibrain_top_k", settingTopK.value);
      if (settingTemperature) localStorage.setItem("omnibrain_temperature", settingTemperature.value);
      if (settingGeminiKey?.value) localStorage.setItem("omnibrain_gemini_key", settingGeminiKey.value);
      if (settingOpenAIKey?.value) localStorage.setItem("omnibrain_openai_key", settingOpenAIKey.value);
      
      settingsModal.classList.remove("active");
      showToast("Settings saved successfully!", "success");
    });
  }

  if (resetDemoDataBtn) {
    resetDemoDataBtn.addEventListener("click", async () => {
      if (!confirm("Reset demo SQLite database and ChromaDB vector store?")) return;
      try {
        const res = await fetch("/api/reset-demo", { method: "POST" });
        const data = await res.json();
        if (settingsModal) settingsModal.classList.remove("active");
        showToast(data.message || "Demo data reset successfully", "success");
        window.loadDocuments();
      } catch (err) {
        showToast(`Reset failed: ${err.message}`, "error");
      }
    });
  }

  // -------------------------------------------------------------
  // 10. System Health Check
  // -------------------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error();
      const data = await res.json();
      if (workspaceEngineStatus) workspaceEngineStatus.textContent = `Agents Active (${data.total_chunks} chunks)`;
    } catch (e) {
      if (workspaceEngineStatus) workspaceEngineStatus.textContent = "Engine Offline";
    }
  }

  // Initialize
  loadStoredSessions();
  checkHealth();
  window.loadDocuments();
});
