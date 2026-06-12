let history = [];
let streaming = false;

const fileInput = document.getElementById("fileInput");
const attachBtn = document.getElementById("attachBtn");
const docList = document.getElementById("docList");
const messages = document.getElementById("messages");
const emptyState = document.getElementById("emptyState");
const msgInput = document.getElementById("msgInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const dropOverlay = document.getElementById("dropOverlay");

loadStats();

async function loadStats() {
  try {
    const d = await fetch("/api/stats").then((r) => r.json());
    renderDocList(d.docs);
    sendBtn.disabled = d.vectors === 0;
  } catch {}
}

function renderDocList(docs) {
  docList.innerHTML = "";
  if (docs.length === 0) {
    const li = document.createElement("li");
    li.className = "doc-empty";
    li.textContent = "No documents indexed yet";
    docList.appendChild(li);
    return;
  }
  docs.forEach((name) => {
    const ext = name.split(".").pop().toLowerCase();
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `
        <div class="doc-badge ${["pdf", "txt", "md"].includes(ext) ? ext : "txt"}">${ext.toUpperCase()}</div>
        <span class="doc-name" title="${name}">${name}</span>
      `;
    docList.appendChild(li);
  });
}

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) doUpload(e.target.files);
});

let dragCounter = 0;

window.addEventListener("dragenter", (e) => {
  if (e.dataTransfer && e.dataTransfer.types.includes("Files")) {
    dragCounter++;
    dropOverlay.classList.add("show");
  }
});

window.addEventListener("dragleave", () => {
  dragCounter--;
  if (dragCounter <= 0) {
    dragCounter = 0;
    dropOverlay.classList.remove("show");
  }
});

window.addEventListener("dragover", (e) => e.preventDefault());

window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropOverlay.classList.remove("show");
  if (e.dataTransfer.files.length) doUpload(e.dataTransfer.files);
});

async function doUpload(files) {
  attachBtn.disabled = true;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  try {
    const d = await fetch("/api/upload", { method: "POST", body: fd }).then(
      (r) => r.json(),
    );
    toast(
      `Indexed ${d.chunks.toLocaleString()} chunks from ${d.files.length} file(s) in ${d.elapsed}s`,
    );
    await loadStats();
  } catch {
    toast("Upload failed. Please try again.");
  } finally {
    attachBtn.disabled = false;
    fileInput.value = "";
  }
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("on"));
    btn.classList.add("on");
    const v = btn.dataset.v;
    document.getElementById("chatView").classList.toggle("off", v !== "chat");
    document.getElementById("evalView").classList.toggle("off", v !== "eval");
    clearBtn.style.display = v === "chat" ? "" : "none";
  });
});

msgInput.addEventListener("input", autoResize);
msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMsg();
  }
});
sendBtn.addEventListener("click", sendMsg);

function autoResize() {
  msgInput.style.height = "auto";
  msgInput.style.height = Math.min(msgInput.scrollHeight, 120) + "px";
}

async function sendMsg() {
  const text = msgInput.value.trim();
  if (!text || streaming) return;

  streaming = true;
  sendBtn.disabled = true;
  msgInput.value = "";
  msgInput.style.height = "auto";

  emptyState.style.display = "none";
  addBubble("user", text);

  const typingRow = document.createElement("div");
  typingRow.className = "typing-row";
  typingRow.innerHTML = `<div class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
  messages.appendChild(typingRow);
  scrollBottom();

  let fullText = "";
  let bubble = null;
  let sources = [];

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: history,
        confidence_threshold: 0,
      }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const lines = decoder.decode(value).split("\n");
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") break;

        try {
          const p = JSON.parse(payload);
          if (p.text !== undefined) {
            fullText += p.text;
            if (!bubble) {
              typingRow.className = "row ai";
              typingRow.innerHTML = `<div class="msg-wrap"><div class="bubble"></div></div>`;
              bubble = typingRow.querySelector(".bubble");
            }
            bubble.innerHTML = renderStreaming(fullText);
            scrollBottom();
          }
          if (p.sources) sources = p.sources;
        } catch {}
      }
    }

    if (bubble) {
      bubble.innerHTML = renderMarkdown(fullText);
      typingRow.querySelector(".msg-wrap").appendChild(makeCopyBtn(fullText));
    }

    if (sources.length) {
      const srcRow = document.createElement("div");
      srcRow.className = "row ai";
      srcRow.innerHTML = `<div class="sources">${sources
        .map(
          (s) =>
            `<span class="src-pill">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline;margin-right:3px">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>${s}</span>`,
        )
        .join("")}</div>`;
      messages.appendChild(srcRow);
    }

    history.push([text, fullText]);
    if (history.length > 12) history.shift();
  } catch {
    if (!bubble) {
      typingRow.className = "row ai";
      typingRow.innerHTML = `<div class="bubble" style="color:var(--red)">Something went wrong. Please try again.</div>`;
    }
  }

  streaming = false;
  sendBtn.disabled = false;
  scrollBottom();
}

clearBtn.addEventListener("click", () => {
  history = [];
  messages.innerHTML = "";
  messages.appendChild(emptyState);
  emptyState.style.display = "";
});

document.getElementById("evalBtn").addEventListener("click", runEval);

async function runEval() {
  const q = document.getElementById("evalQ").value.trim();
  const a = document.getElementById("evalA").value.trim();
  const ctx = document.getElementById("evalCtx").value;
  if (!q || !a) {
    toast("Please provide both a question and an answer.");
    return;
  }

  const btn = document.getElementById("evalBtn");
  btn.disabled = true;
  btn.innerHTML = '<div class="spin"></div><span>Evaluating…</span>';

  try {
    const d = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, answer: a, context: ctx }),
    }).then((r) => r.json());

    if (d.error) {
      toast(d.error);
      return;
    }
    renderMetrics(d);
  } catch {
    toast("Evaluation failed. Please try again.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Evaluation";
  }
}

function renderMetrics(d) {
  const card = document.getElementById("evalResultCard");
  const grid = document.getElementById("metricsGrid");
  card.style.display = "";

  const defs = [
    { key: "context_relevance", label: "Context Relevance" },
    { key: "faithfulness", label: "Faithfulness" },
    { key: "answer_relevance", label: "Answer Relevance" },
    { key: "overall_score", label: "Overall Score" },
  ];

  grid.innerHTML = defs
    .map((m) => {
      const v = d[m.key];
      const pct = Math.round(v * 100);
      const c = v >= 0.7 ? "good" : v >= 0.4 ? "ok" : "poor";
      return `
        <div class="metric">
          <div class="metric-name">${m.label}</div>
          <div class="metric-val c-${c}">${v.toFixed(3)}</div>
          <div class="metric-bar"><div class="metric-fill f-${c}" style="width:${pct}%"></div></div>
        </div>`;
    })
    .join("");

  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function addBubble(role, text) {
  const row = document.createElement("div");
  row.className = `row ${role}`;
  if (role === "ai") {
    const wrap = document.createElement("div");
    wrap.className = "msg-wrap";
    const bub = document.createElement("div");
    bub.className = "bubble";
    bub.innerHTML = renderMarkdown(text);
    wrap.appendChild(bub);
    wrap.appendChild(makeCopyBtn(text));
    row.appendChild(wrap);
  } else {
    row.innerHTML = `<div class="bubble">${renderStreaming(text)}</div>`;
  }
  messages.appendChild(row);
  scrollBottom();
}

function scrollBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function esc(t) {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderStreaming(text) {
  return esc(text).replace(/\n/g, "<br>");
}

function inlineFormat(text) {
  return esc(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>");
}

function renderMarkdown(raw) {
  const segments = [];
  let rem = raw,
    idx = 0;
  const codeRe = /```(\w*)\n?([\s\S]*?)```/g;
  let m;

  while ((m = codeRe.exec(rem)) !== null) {
    if (m.index > idx) segments.push({ t: "text", c: rem.slice(idx, m.index) });
    segments.push({ t: "code", lang: m[1], c: m[2].trim() });
    idx = m.index + m[0].length;
  }
  if (idx < rem.length) segments.push({ t: "text", c: rem.slice(idx) });

  return segments
    .map((s) => {
      if (s.t === "code") {
        return `<pre><code>${esc(s.c)}</code></pre>`;
      }
      const lines = s.c.split("\n");
      const out = [];
      let inList = false;

      for (const line of lines) {
        const hm = line.match(/^(#{1,4})\s+(.+)/);
        const lm = line.match(/^[-*]\s+(.+)/);
        if (hm) {
          if (inList) {
            out.push("</ul>");
            inList = false;
          }
          const lvl = hm[1].length;
          out.push(`<h${lvl}>${inlineFormat(hm[2])}</h${lvl}>`);
        } else if (lm) {
          if (!inList) {
            out.push("<ul>");
            inList = true;
          }
          out.push(`<li>${inlineFormat(lm[1])}</li>`);
        } else {
          if (inList) {
            out.push("</ul>");
            inList = false;
          }
          out.push(line.trim() ? inlineFormat(line) + "<br>" : "<br>");
        }
      }
      if (inList) out.push("</ul>");
      return out.join("");
    })
    .join("");
}

const ICON_COPY = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
const ICON_CHECK = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;

function makeCopyBtn(text) {
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.innerHTML = `${ICON_COPY}<span>Copy</span>`;
  btn.addEventListener("click", () => {
    navigator.clipboard.writeText(text).then(() => {
      btn.classList.add("copied");
      btn.innerHTML = `${ICON_CHECK}<span>Copied</span>`;
      setTimeout(() => {
        btn.classList.remove("copied");
        btn.innerHTML = `${ICON_COPY}<span>Copy</span>`;
      }, 2000);
    });
  });
  return btn;
}

let toastTimer;
function toast(msg) {
  clearTimeout(toastTimer);
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  toastTimer = setTimeout(() => el.classList.remove("show"), 3200);
}