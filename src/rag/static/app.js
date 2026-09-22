"use strict";

const form = document.getElementById("question-form");
const question = document.getElementById("question");
const submit = document.getElementById("submit");
const feedback = document.getElementById("feedback");
const results = document.getElementById("results");
const welcome = document.getElementById("welcome");
const newConversation = document.getElementById("new-conversation");
let liveMode = false;
let remaining = null;
let history = [];
let inFlight = false;
let exchangeNumber = 0;

function rememberTurn(text, answer) {
  const reply = answer.parts.length ? answer.parts.map(part => part.text).join("\n") : answer.message;
  history.push({ question: text, answer: Array.from(reply).slice(0, 4000).join(""), status: answer.status });
  while (history.length > 4 || history.reduce((total, turn) => total + turn.question.length + turn.answer.length, 0) > 12000) history.shift();
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function sourceLink(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : null;
  } catch { return null; }
}

function showAnswer(answer, target, turn) {
  target.append(element("span", answer.mode === "generated" ? "Referat" : "Najdeni odlomki", "reply-label"));
  if (!answer.parts.length) target.append(element("p", answer.message, "answer-text"));
  const citations = new Map(answer.citations.map((item, index) => [item.id, { ...item, number: index + 1 }]));
  const used = new Set();
  for (const part of answer.parts) {
    const paragraph = element(answer.mode === "generated" ? "p" : "blockquote", part.text, "answer-text");
    for (const id of new Set(part.citation_ids)) {
      const citation = citations.get(id);
      if (!citation) continue;
      used.add(id);
      const reference = element("a", `[${citation.number}]`, "citation-ref");
      reference.href = `#source-${turn}-${citation.number}`;
      reference.setAttribute("aria-label", `Vir ${citation.number}: ${citation.title}`);
      reference.addEventListener("click", event => {
        event.preventDefault();
        const source = document.getElementById(`source-${turn}-${citation.number}`);
        if (source) { source.open = true; source.focus(); source.scrollIntoView({ block: "center" }); }
      });
      paragraph.append(reference);
    }
    target.append(paragraph);
  }
  if (used.size) {
    const sources = element("div", undefined, "sources");
    sources.append(element("p", "VIRI", "sources-label"));
    for (const [id, citation] of citations) {
      if (!used.has(id)) continue;
      const source = element("details", undefined, "source");
      source.id = `source-${turn}-${citation.number}`;
      source.tabIndex = -1;
      const location = citation.section ? ` · ${citation.section}` : citation.page ? ` · str. ${citation.page}` : "";
      source.append(element("summary", `[${citation.number}] ${citation.title}${location}${citation.article ? ` · ${citation.article}. člen` : ""}`));
      source.append(element("blockquote", citation.text));
      const url = sourceLink(citation.url);
      if (url) {
        const link = element("a", "Odpri vir ↗", "source-url");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        source.append(link);
      }
      if (citation.captured_at) source.append(element("p", `Zajeto: ${citation.captured_at.slice(0, 10)}. Stran se je lahko od takrat spremenila.`, "source-date"));
      if (citation.review_status !== "reviewed") source.append(element("p", "Veljavnost vira še ni pregledana.", "warning"));
      sources.append(source);
    }
    target.append(sources);
  }
  if (answer.warnings.length) {
    const notes = element("details", undefined, "answer-notes");
    notes.append(element("summary", "O zanesljivosti odgovora"));
    for (const warning of answer.warnings) notes.append(element("p", warning, "warning"));
    target.append(notes);
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (submit.disabled || inFlight) return;
  const text = question.value.trim();
  if (text.length < 2) { feedback.textContent = "Vnesite vsaj dva znaka."; return; }
  submit.disabled = true;
  inFlight = true;
  newConversation.disabled = true;
  question.readOnly = true;
  welcome.hidden = true;
  feedback.textContent = "Iščem vire in pripravljam odgovor …";
  const exchange = element("section", undefined, "exchange");
  exchange.append(element("p", text, "user-question"));
  const reply = element("div", undefined, "assistant-reply");
  reply.setAttribute("aria-busy", "true");
  exchange.append(reply);
  results.append(exchange);
  const turn = ++exchangeNumber;
  while (results.children.length > 20) results.firstElementChild.remove();
  try {
    const response = await fetch("/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, history }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Odgovora ni bilo mogoče pripraviti.");
    showAnswer(data, reply, turn);
    rememberTurn(text, data);
    question.value = "";
    feedback.textContent = "";
    reply.scrollIntoView({ block: "start" });
  } catch (error) {
    feedback.textContent = error instanceof Error ? error.message : "Strežnik ni dosegljiv.";
    reply.append(element("p", "Odgovora ni bilo mogoče pripraviti. Poskusite znova.", "warning"));
  } finally {
    reply.setAttribute("aria-busy", "false");
    inFlight = false;
    newConversation.disabled = false;
    question.readOnly = false;
    await refreshStatus();
  }
});

newConversation.addEventListener("click", () => {
  if (inFlight) return;
  history = [];
  results.replaceChildren();
  welcome.hidden = false;
  question.value = "";
  feedback.textContent = "";
  question.focus();
});

for (const button of document.querySelectorAll("[data-question]")) {
  button.addEventListener("click", () => {
    if (inFlight) return;
    question.value = button.dataset.question;
    question.focus();
  });
}

async function refreshStatus() {
  try {
    const response = await fetch("/status");
    if (!response.ok) throw new Error("Status ni dosegljiv.");
    const status = await response.json();
    liveMode = status.mode === "generated";
    remaining = status.questions_remaining ?? null;
    submit.disabled = inFlight || (liveMode && remaining === 0);
    document.getElementById("mode-notice").textContent = liveMode ? "" : "Lokalni prikaz: izvirni odlomki iz dokumentov.";
    document.getElementById("session-usage").textContent = liveMode && remaining !== null
      ? (remaining === 0 ? "Omejitev vprašanj je dosežena." : `Na voljo še ${remaining} vprašanj`)
      : "";
  } catch {
    submit.disabled = true;
    feedback.textContent = "Povezava ni uspela. Osvežite stran.";
  }
}
refreshStatus();
