"use strict";

const form = document.getElementById("question-form");
const question = document.getElementById("question");
const submit = document.getElementById("submit");
const feedback = document.getElementById("feedback");
const results = document.getElementById("results");
const newConversation = document.getElementById("new-conversation");
let liveMode = false;
let remaining = null;
let history = [];
let inFlight = false;

function rememberTurn(text, answer) {
  const reply = answer.parts.length ? answer.parts.map(part => part.text).join("\n") : answer.message;
  history.push({ question: text, answer: Array.from(reply).slice(0, 4000).join(""), status: answer.status });
  while (history.length > 4 || history.reduce((total, turn) => total + turn.question.length + turn.answer.length, 0) > 12000) history.shift();
  document.getElementById("conversation-notice").textContent = `Za nadaljevanje se uporabi do ${history.length} zadnjih izmenjav. Ob osvežitvi zavihka se pogovor izbriše.`;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showAnswer(answer, target) {
  target.append(element("h2", answer.message));
  for (const warning of answer.warnings) target.append(element("p", warning, "warning"));
  const citations = new Map(answer.citations.map((item, index) => [item.id, { ...item, number: index + 1 }]));
  for (const part of answer.parts) {
    const card = element("article", undefined, "evidence");
    const cited = part.citation_ids.map(id => citations.get(id)).filter(Boolean);
    card.append(element("p", (answer.mode === "generated" ? "ODGOVOR " : "IZVIRNI ODLOMEK ") + cited.map(item => `[${item.number}]`).join(" "), "eyebrow"));
    card.append(element(answer.mode === "generated" ? "p" : "blockquote", part.text));
    for (const citation of cited) {
      const source = element("details");
      const location = citation.section ? ` · ${citation.section}` : ` · str. ${citation.page}`;
      source.append(element("summary", `[${citation.number}] ${citation.title}${location}` + (citation.article ? ` · ${citation.article}. člen` : "")));
      if (citation.captured_at) source.append(element("p", "Spletni posnetek: " + citation.captured_at + ". Izvirna stran se je lahko od takrat spremenila.", "version"));
      source.append(element("p", citation.review_status === "reviewed" ? "Vir pregledan" : "Veljavnost vira še ni pregledana", "warning"));
      if (answer.mode === "generated") {
        source.append(element("blockquote", citation.text));
        const url = citation.url ? new URL(citation.url) : null;
        if (url && ["https:", "http:"].includes(url.protocol)) {
          const link = element("a", "Odpri izvirni dokument ↗", "source-url");
          link.href = url.href;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          source.append(link);
        }
      } else source.append(element("p", citation.url, "source-url"));
      source.append(element("p", "Različica (SHA-256): " + citation.version, "version"));
      card.append(source);
    }
    target.append(card);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submit.disabled) return;
  const text = question.value.trim();
  if (text.length < 2) {
    feedback.textContent = "Vnesite vsaj dva znaka.";
    return;
  }
  submit.disabled = true;
  inFlight = true;
  newConversation.disabled = true;
  feedback.textContent = liveMode ? "Iščem vire in pripravljam odgovor z Azure …" : "Iščem po lokalnih odlomkih …";
  if (results.querySelector(".empty")) results.replaceChildren();
  const exchange = element("section", undefined, "exchange");
  exchange.append(element("p", "Vi", "eyebrow"));
  exchange.append(element("p", text, "user-question"));
  const reply = element("div", undefined, "assistant-reply");
  exchange.append(reply);
  results.append(exchange);
  while (results.children.length > 20) results.firstElementChild.remove();
  try {
    const response = await fetch("/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, history }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Iskanje ni uspelo.");
    showAnswer(data, reply);
    rememberTurn(text, data);
    question.value = "";
    feedback.textContent = data.mode === "generated" ? "Končano. Preverite odgovor in navedene vire." : "Končano. Uporabljeni so bili samo lokalni podatki.";
  } catch (error) {
    feedback.textContent = error instanceof Error ? error.message : "Lokalni strežnik ni dosegljiv.";
    reply.append(element("p", "Odgovor ni bil pridobljen. Ta poskus ni dodan kontekstu pogovora.", "warning"));
  } finally {
    inFlight = false;
    newConversation.disabled = false;
    await refreshStatus();
  }
});

newConversation.addEventListener("click", () => {
  if (inFlight) return;
  history = [];
  results.replaceChildren();
  question.value = "";
  document.getElementById("conversation-notice").textContent = "Nov pogovor. Zgodovina prejšnjega pogovora se ne bo uporabila.";
  feedback.textContent = "Pogovor je počiščen. Preostala omejitev API-klicev se ni spremenila.";
  question.focus();
});

for (const button of document.querySelectorAll("[data-question]")) {
  button.addEventListener("click", () => {
    question.value = button.dataset.question;
    question.focus();
  });
}

async function refreshStatus() {
return fetch("/status").then(response => {
  if (!response.ok) throw new Error("Status ni dosegljiv.");
  return response.json();
}).then(status => {
  liveMode = status.mode === "generated";
  remaining = status.questions_remaining ?? null;
  submit.disabled = inFlight || (liveMode && remaining === 0);
  submit.textContent = "Pošlji";
  document.getElementById("mode-notice").textContent = liveMode
    ? ""
    : "Prikazani so izvirni odlomki, ne ustvarjeni odgovori.";
  document.getElementById("session-usage").textContent = liveMode
    ? (remaining === 0 ? "Omejitev vprašanj je dosežena." : `Preostala vprašanja: ${remaining}`)
    : "";
}).catch(() => { submit.disabled = true; feedback.textContent = "Povezava ni uspela. Osvežite stran."; });
}
refreshStatus();
