// WebResearcher Modern Minimalist UI Controller
// Supports 4 Mode-Specific Dynamic Visualizations (IterResearch, WebWeaver, TTS, ReAct)

document.addEventListener("DOMContentLoaded", () => {
  // Common Elements
  const questionInput = document.getElementById("question-input");
  const paradigmCards = document.querySelectorAll(".paradigm-card");
  const advancedToggle = document.getElementById("advanced-toggle");
  const advancedPanel = document.getElementById("advanced-panel");
  const maxRoundsInput = document.getElementById("max-rounds");
  const ttsBranchesInput = document.getElementById("tts-branches");
  const customInstructionInput = document.getElementById("custom-instruction");
  
  const startBtn = document.getElementById("start-btn");
  const cancelBtn = document.getElementById("cancel-btn");
  
  const monitorCard = document.getElementById("monitor-card");
  const taskStatusBadge = document.getElementById("task-status-badge");
  const currentModeIndicator = document.getElementById("current-mode-indicator");
  const metricTimer = document.getElementById("metric-timer");
  const metricRound = document.getElementById("metric-round");
  const metricTokens = document.getElementById("metric-tokens");
  
  const thinkingToggle = document.getElementById("thinking-toggle");
  const thinkingText = document.getElementById("thinking-text");

  // Mode View Containers
  const viewIter = document.getElementById("view-iter");
  const viewWebweaver = document.getElementById("view-webweaver");
  const viewTts = document.getElementById("view-tts");
  const viewReact = document.getElementById("view-react");

  // 1. Iter Elements
  const iterReportDraft = document.getElementById("iter-report-draft");
  const iterCurrentPlan = document.getElementById("iter-current-plan");
  const iterToolsList = document.getElementById("iter-tools-list");

  // 2. WebWeaver Elements
  const stepPlanner = document.getElementById("step-planner");
  const stepWriter = document.getElementById("step-writer");
  const stepCitation = document.getElementById("step-citation");
  const webweaverOutlineTree = document.getElementById("webweaver-outline-tree");
  const webweaverWriterPreview = document.getElementById("webweaver-writer-preview");
  const webweaverMemoryList = document.getElementById("webweaver-memory-list");
  const memoryBankCount = document.getElementById("memory-bank-count");

  // 3. TTS Elements
  const ttsBranchesGrid = document.getElementById("tts-branches-grid");
  const synthesizerBadge = document.getElementById("synthesizer-badge");
  const ttsSynthesizerContent = document.getElementById("tts-synthesizer-content");

  // 4. ReAct Elements
  const reactTimelineFeed = document.getElementById("react-timeline-feed");

  // Final Report Elements
  const reportCard = document.getElementById("report-card");
  const reportMarkdownBody = document.getElementById("report-markdown-body");
  const reportMetaInfo = document.getElementById("report-meta-info");
  const copyReportBtn = document.getElementById("copy-report-btn");
  const downloadReportBtn = document.getElementById("download-report-btn");

  // State
  let selectedMode = "iter";
  let activeTaskId = null;
  let eventSource = null;
  let timerInterval = null;
  let startTime = null;
  let rawReportContent = "";

  const modeDisplayMap = {
    iter: "IterResearch (状态机)",
    webweaver: "WebWeaver (双智能体大纲)",
    tts: "Test-Time Scaling (多路裁决)",
    react: "ReAct (经典基准)",
  };

  // 1. Paradigm Selection
  paradigmCards.forEach(card => {
    card.addEventListener("click", () => {
      paradigmCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      selectedMode = card.getAttribute("data-mode");
    });
  });

  // 2. Advanced Settings Toggle
  advancedToggle.addEventListener("click", () => {
    advancedPanel.classList.toggle("hidden");
    const isHidden = advancedPanel.classList.contains("hidden");
    advancedToggle.innerHTML = `<span class="toggle-icon">${isHidden ? "▸" : "▾"}</span> 高级参数设置 (探索轮数、提示偏置)`;
  });

  // 3. Thinking Box Collapse Toggle
  thinkingToggle.addEventListener("click", () => {
    thinkingText.classList.toggle("hidden");
    const isHidden = thinkingText.classList.contains("hidden");
    thinkingToggle.querySelector(".thinking-expand-icon").textContent = isHidden ? "展开" : "收起";
  });

  // 4. Switch Dynamic Mode Views
  function switchModeView(mode) {
    viewIter.classList.add("hidden");
    viewWebweaver.classList.add("hidden");
    viewTts.classList.add("hidden");
    viewReact.classList.add("hidden");

    if (mode === "webweaver") {
      viewWebweaver.classList.remove("hidden");
    } else if (mode === "tts") {
      viewTts.classList.remove("hidden");
    } else if (mode === "react") {
      viewReact.classList.remove("hidden");
    } else {
      viewIter.classList.remove("hidden");
    }

    currentModeIndicator.textContent = modeDisplayMap[mode] || "IterResearch";
  }

  // 5. Start Research
  startBtn.addEventListener("click", async () => {
    const question = questionInput.value.trim();
    if (!question) {
      alert("请输入研究课题！");
      return;
    }

    // Reset UI
    startBtn.disabled = true;
    cancelBtn.classList.remove("hidden");
    monitorCard.classList.remove("hidden");
    reportCard.classList.add("hidden");
    
    thinkingText.textContent = "等待模型生成思维链...";
    metricRound.textContent = "0";
    metricTokens.textContent = "0";
    taskStatusBadge.textContent = "正在调度任务...";

    // Reset Mode-Specific views
    switchModeView(selectedMode);
    resetModeViews();

    // Start Timer
    startTime = Date.now();
    clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      metricTimer.textContent = `${elapsed}s`;
    }, 100);

    const payload = {
      question: question,
      mode: selectedMode,
      max_rounds: parseInt(maxRoundsInput.value, 10) || 5,
      instruction: customInstructionInput.value.trim(),
      num_parallel_agents: parseInt(ttsBranchesInput.value, 10) || 2,
    };

    try {
      const resp = await fetch("/api/research/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error(`HTTP error! status: ${resp.status}`);
      }

      const data = await resp.json();
      activeTaskId = data.task_id;
      listenToEventStream(activeTaskId);

    } catch (err) {
      alert(`启动任务失败: ${err.message}`);
      resetExecutionState();
    }
  });

  // 6. Reset Mode Specific Views
  function resetModeViews() {
    // Iter
    iterReportDraft.innerHTML = "<em>第一轮探索中，暂无草稿内容...</em>";
    iterCurrentPlan.innerHTML = "<strong>💭 当前轮次规划:</strong><p>等待模型分析线索与规划下一步意图...</p>";
    iterToolsList.innerHTML = "";

    // WebWeaver
    stepPlanner.className = "step-node active";
    stepWriter.className = "step-node";
    stepCitation.className = "step-node";
    webweaverOutlineTree.textContent = "等待 Planner 输出第一版研究大纲草案...";
    webweaverWriterPreview.innerHTML = "<em>等待 Planner 构建完毕后交由 Writer 撰写...</em>";
    webweaverMemoryList.innerHTML = '<div class="empty-hint">等待智能体收集互联网证据并分配全局唯一 ID...</div>';
    memoryBankCount.textContent = "0 条证据";

    // TTS
    ttsBranchesGrid.innerHTML = "";
    synthesizerBadge.textContent = "等待各路探索汇总";
    ttsSynthesizerContent.innerHTML = "<p class='synthesizer-status-text'>各路探索分支正在并发展开独立调研...</p>";

    // ReAct
    reactTimelineFeed.innerHTML = "";
  }

  // 7. Cancel Task
  cancelBtn.addEventListener("click", async () => {
    if (!activeTaskId) return;
    try {
      await fetch(`/api/research/cancel/${activeTaskId}`, { method: "POST" });
      taskStatusBadge.textContent = "已取消";
      resetExecutionState();
    } catch (err) {
      console.error(err);
    }
  });

  // 8. SSE Event Listener
  function listenToEventStream(taskId) {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource(`/api/research/stream/${taskId}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleStreamEvent(data);
      } catch (err) {
        console.error("Failed to parse SSE payload:", err, event.data);
      }
    };

    eventSource.onerror = (err) => {
      console.warn("SSE connection closed or lost:", err);
      if (eventSource) {
        eventSource.close();
      }
      resetExecutionState();
    };
  }

  // 9. Central Stream Event Dispatcher
  function handleStreamEvent(data) {
    const type = data.type;

    // 1. Status & Common
    if (type === "status") {
      taskStatusBadge.textContent = data.status || "运行中";
    } 
    else if (type === "thinking") {
      thinkingText.textContent = data.content || "";
    } 
    else if (type === "round") {
      metricRound.textContent = data.round || "1";
      
      // Iter Mode Updates
      if (selectedMode === "iter") {
        if (data.plan) {
          iterCurrentPlan.innerHTML = `<strong>💭 第 ${data.round} 轮规划:</strong><p>${data.plan}</p>`;
        }
        if (data.report && data.report.length > 30) {
          iterReportDraft.innerHTML = window.marked ? marked.parse(data.report) : `<pre>${data.report}</pre>`;
        }
      }
      // ReAct Mode Updates
      else if (selectedMode === "react") {
        addReactStep(data.round, data.plan || "思考中...", null, null);
      }
    } 
    else if (type === "tool") {
      const toolName = data.tool_name || "Tool";
      const argsStr = JSON.stringify(data.tool_args || {});
      const obsPreview = data.observation ? data.observation.slice(0, 180) + "..." : "";

      if (selectedMode === "iter") {
        const item = document.createElement("div");
        item.className = "action-item";
        item.innerHTML = `<strong>🛠️ 调用工具: ${toolName}</strong><p style="color:var(--text-secondary); margin-top:2px;">入参: <code>${argsStr}</code></p>`;
        iterToolsList.prepend(item);
      } else if (selectedMode === "react") {
        addReactStep(metricRound.textContent, null, `${toolName}: ${argsStr}`, obsPreview);
      }
    }

    // 2. WebWeaver Mode Specific Events
    else if (type === "phase") {
      const ph = data.phase || "";
      if (selectedMode === "webweaver") {
        if (ph.includes("planner")) {
          stepPlanner.className = "step-node active";
          stepWriter.className = "step-node";
          stepCitation.className = "step-node";
        } else if (ph.includes("writer")) {
          stepPlanner.className = "step-node";
          stepWriter.className = "step-node active";
          stepCitation.className = "step-node";
        } else if (ph.includes("citation")) {
          stepPlanner.className = "step-node";
          stepWriter.className = "step-node";
          stepCitation.className = "step-node active";
        }
      } else if (selectedMode === "tts" && ph.includes("synthesizer")) {
        synthesizerBadge.textContent = "正在交叉比对与消除矛盾";
        ttsSynthesizerContent.innerHTML = `<p class="synthesizer-status-text" style="color:var(--accent-primary); font-weight:600;">⚡ 首席裁决官已接收所有分支报告，正在执行事实交叉审计、消除矛盾并综合生成终极研报...</p>`;
      }
    }
    else if (type === "outline") {
      if (data.outline) {
        webweaverOutlineTree.innerHTML = window.marked ? marked.parse(data.outline) : `<pre>${data.outline}</pre>`;
      }
    }
    else if (type === "memory_bank_update") {
      const items = data.items || [];
      memoryBankCount.textContent = `${items.length} 条证据`;
      if (items.length > 0) {
        webweaverMemoryList.innerHTML = "";
        items.forEach(it => {
          const card = document.createElement("div");
          card.className = "memory-card";
          card.innerHTML = `
            <div class="memory-card-header">
              <span class="evidence-id-badge">${it.id}</span>
              ${it.url ? `<a href="${it.url}" target="_blank" class="memory-card-title">${it.title || '网页证据'}</a>` : `<span class="memory-card-title">${it.title || '计算证据'}</span>`}
            </div>
            <div class="memory-card-snippet">${it.summary || ''}</div>
          `;
          webweaverMemoryList.appendChild(card);
        });
      }
    }
    else if (type === "write") {
      const content = data.total_written || data.section || "";
      if (content) {
        webweaverWriterPreview.innerHTML = window.marked ? marked.parse(content) : `<p>${content}</p>`;
        webweaverWriterPreview.scrollTop = webweaverWriterPreview.scrollHeight;
      }
    }

    // 3. TTS Mode Specific Events
    else if (type === "tts_branches_init") {
      const branches = data.branches || [];
      ttsBranchesGrid.innerHTML = "";
      branches.forEach(b => {
        const bCard = document.createElement("div");
        bCard.className = "branch-card";
        bCard.id = `branch-card-${b.branch_id}`;
        bCard.innerHTML = `
          <div class="branch-card-header">
            <span class="branch-name">分支 ${b.branch_id}</span>
            <span class="badge badge-indigo">T = ${b.temperature}</span>
          </div>
          <div class="branch-bias-tag">${b.bias || '独立探索视角'}</div>
          <div class="branch-body" id="branch-body-${b.branch_id}">
            <em>分支准备中...</em>
          </div>
        `;
        ttsBranchesGrid.appendChild(bCard);
      });
    }
    else if (type === "branch_event") {
      const bId = data.branch_id;
      const bBody = document.getElementById(`branch-body-${bId}`);
      if (bBody) {
        // Clear placeholder text on first real event
        const placeholder = bBody.querySelector("em");
        if (placeholder) {
          bBody.removeChild(placeholder);
        }

        const innerType = data.inner_type || data.type;
        if (innerType === "round") {
          if (data.plan) {
            const item = document.createElement("div");
            item.className = "branch-step-item";
            item.style.marginBottom = "8px";
            item.innerHTML = `<span style="font-weight:600; color:var(--text-primary);">[R${data.round}] 💭 规划:</span> ${data.plan}`;
            bBody.appendChild(item);
          }
          if (data.report && data.report.length > 50 && !data.report.includes("已并入下方")) {
            const rptItem = document.createElement("div");
            rptItem.className = "branch-report-preview";
            rptItem.style.cssText = "font-size:12px; color:var(--text-secondary); background:var(--bg-card); padding:8px; border-radius:6px; margin-top:4px; max-height:120px; overflow-y:auto;";
            rptItem.innerHTML = window.marked ? marked.parse(data.report.slice(0, 500) + "...") : data.report.slice(0, 500);
            bBody.appendChild(rptItem);
          }
        } else if (innerType === "tool") {
          const item = document.createElement("div");
          item.className = "branch-tool-item";
          item.style.cssText = "font-size:12px; color:var(--accent-primary); font-weight:600; margin-bottom:6px;";
          item.innerHTML = `🛠️ 调用: <code>${data.tool_name}</code>`;
          bBody.appendChild(item);
        }

        bBody.scrollTop = bBody.scrollHeight;
      }
    }

    // 4. Final Event (Early Preview)
    else if (type === "final") {
      const finalAns = data.answer || data.prediction || data.report || "";
      if (finalAns && finalAns.length > 50 && !finalAns.includes("已并入下方")) {
        rawReportContent = finalAns;
        if (window.marked) {
          reportMarkdownBody.innerHTML = marked.parse(finalAns);
        } else {
          reportMarkdownBody.innerHTML = `<pre>${finalAns}</pre>`;
        }
        reportCard.classList.remove("hidden");
      }
    }

    // 5. Completion Event
    else if (type === "complete") {
      clearInterval(timerInterval);
      taskStatusBadge.textContent = "已完成";
      metricTokens.textContent = data.total_tokens || "0";
      
      let finalReport = data.prediction || data.report || rawReportContent || "";
      if (data.report && data.report.length > (data.prediction || "").length && !data.report.includes("已并入下方")) {
        finalReport = data.report;
      }
      rawReportContent = finalReport;

      // Render Final Markdown
      if (window.marked && finalReport) {
        reportMarkdownBody.innerHTML = marked.parse(finalReport);
      } else {
        reportMarkdownBody.innerHTML = `<pre>${finalReport}</pre>`;
      }

      reportMetaInfo.textContent = `耗时: ${data.elapsed_time ? data.elapsed_time.toFixed(1) + 's' : ''} | 步数: ${data.total_steps || 0} | 消耗 Token: ${data.total_tokens || 0}`;
      reportCard.classList.remove("hidden");
      reportCard.scrollIntoView({ behavior: "smooth" });

      if (selectedMode === "webweaver") {
        stepCitation.className = "step-node active";
      } else if (selectedMode === "tts") {
        synthesizerBadge.textContent = "裁决完成";
      }

      resetExecutionState();
    }
  }

  // Helper for ReAct Timeline
  function addReactStep(round, thought, action, obs) {
    const step = document.createElement("div");
    step.className = "timeline-step";
    const timeStr = new Date().toLocaleTimeString();
    step.innerHTML = `
      <div class="timeline-step-header">
        <span style="font-weight:700; color:var(--text-primary);">第 ${round} 步</span>
        <span>${timeStr}</span>
      </div>
      ${thought ? `<p><strong>💭 思考:</strong> ${thought}</p>` : ''}
      ${action ? `<p style="color:var(--accent-text);"><strong>🛠️ 行动:</strong> <code>${action}</code></p>` : ''}
      ${obs ? `<p style="color:var(--text-secondary); background:var(--bg-subtle); padding:6px; border-radius:4px;"><strong>👁️ 观察:</strong> ${obs}</p>` : ''}
    `;
    reactTimelineFeed.prepend(step);
  }

  function resetExecutionState() {
    startBtn.disabled = false;
    cancelBtn.classList.add("hidden");
    clearInterval(timerInterval);
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }

  // 10. Copy Markdown Button
  copyReportBtn.addEventListener("click", () => {
    if (!rawReportContent) return;
    navigator.clipboard.writeText(rawReportContent).then(() => {
      const orig = copyReportBtn.textContent;
      copyReportBtn.textContent = "✅ 已复制到剪贴板！";
      setTimeout(() => { copyReportBtn.textContent = orig; }, 2000);
    });
  });

  // 11. Download Markdown Button
  downloadReportBtn.addEventListener("click", () => {
    if (!rawReportContent) return;
    const blob = new Blob([rawReportContent], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `WebResearcher_Report_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  });
});
