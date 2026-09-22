"use strict";

// Exercise the actual browser script with a small DOM and HTTP test double.
// No dependencies, browser services, databases, or network connections are used.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.listeners = {};
    this.textContent = "";
    this.value = "";
    this.disabled = false;
    this.className = "";
    this.dataset = {};
  }
  append(...children) {
    for (const child of children) { child.parent = this; this.children.push(child); }
  }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  focus() { this.focused = true; }
  get firstElementChild() { return this.children[0]; }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  querySelector(selector) {
    for (const child of this.children) {
      if (selector === "." + child.className) return child;
      const found = child.querySelector(selector);
      if (found) return found;
    }
    return null;
  }
}

function answer(text = "Synthetic response", status = "evidence_found") {
  return { mode: "generated", status, message: status === "needs_clarification" ? "Which programme?" : "Answer", parts: status === "evidence_found" ? [{ text, citation_ids: ["current"] }] : [], warnings: [], citations: [{ id: "current", title: "Fixture", page: 1, article: "1", text: "Synthetic evidence", url: "https://example.org/fixture.pdf#page=1", version: "fixture", review_status: "pending" }] };
}

async function browser(options = {}) {
  const ids = new Map();
  const get = id => {
    if (!ids.has(id)) ids.set(id, new Element());
    return ids.get(id);
  };
  const empty = new Element();
  empty.className = "empty";
  get("results").append(empty);
  const requests = [];
  let allowance = options.allowance ?? 100;
  const fetch = async (url, init) => {
    if (url === "/status") return { ok: true, json: async () => ({ mode: "generated", questions_remaining: allowance, embedding_requests: requests.length, chat_requests: requests.length, documents: 1, chunks: 1 }) };
    assert.equal(url, "/ask");
    requests.push(JSON.parse(init.body));
    allowance--;
    if (options.failAt === requests.length) return { ok: false, json: async () => ({ error: "Synthetic provider failure" }) };
    const result = options.response ? options.response(requests.length) : answer();
    return { ok: true, json: async () => result };
  };
  const source = fs.readFileSync(path.join(__dirname, "../../src/rag/static/app.js"), "utf8");
  vm.runInNewContext(source, { document: { getElementById: get, createElement: tag => new Element(tag), querySelectorAll: () => [] }, fetch, URL });
  await new Promise(resolve => setImmediate(resolve));
  return {
    get, requests,
    async ask(text) { get("question").value = text; await get("question-form").listeners.submit({ preventDefault() {} }); },
    reset() { get("new-conversation").listeners.click(); },
  };
}

test("follow-up retains both exchanges and sends prior context", async () => {
  const app = await browser();
  await app.ask("How do I register a thesis topic?");
  await app.ask("And what is the deadline?");
  assert.equal(app.get("results").children.length, 2);
  assert.deepEqual(app.requests[0].history, []);
  assert.equal(app.requests[1].history[0].question, "How do I register a thesis topic?");
  assert.equal(app.requests[1].history[0].answer, "Synthetic response");
  assert.equal(app.get("question").value, "");
});

test("failed answer stays visible but is excluded from subsequent context", async () => {
  const app = await browser({ failAt: 2 });
  await app.ask("First successful question");
  await app.ask("Second failed question");
  assert.equal(app.get("question").value, "Second failed question");
  await app.ask("Third question");
  assert.equal(app.requests[2].history.length, 1);
  assert.equal(app.requests[2].history[0].question, "First successful question");
  assert.equal(app.get("results").children.length, 3);
});

test("new conversation clears context but cannot replenish live allowance", async () => {
  const app = await browser({ allowance: 2 });
  await app.ask("First topic");
  app.reset();
  assert.equal(app.get("results").children.length, 0);
  await app.ask("Second topic");
  assert.deepEqual(app.requests[1].history, []);
  assert.equal(app.get("submit").disabled, true);
  app.reset();
  await app.ask("Over budget");
  assert.equal(app.requests.length, 2);
  assert.equal(app.get("submit").disabled, true);
});

test("history is bounded independently of the visible transcript", async () => {
  const app = await browser({ response: () => answer("x".repeat(4000)) });
  for (let index = 0; index < 23; index++) await app.ask("Question " + index);
  for (const request of app.requests) {
    assert.ok(request.history.length <= 4);
    assert.ok(request.history.reduce((sum, item) => sum + item.question.length + item.answer.length, 0) <= 12000);
  }
  assert.equal(app.get("results").children.length, 20);
  assert.equal(app.requests.at(-1).history.at(-1).question, "Question 21");
});

test("clarification replies and independent tabs use their own context", async () => {
  const a = await browser({ response: count => count === 1 ? answer("", "needs_clarification") : answer() });
  const b = await browser();
  await a.ask("What is the deadline?");
  await a.ask("Master's programme");
  await b.ask("Unrelated new question");
  assert.equal(a.requests[1].history[0].status, "needs_clarification");
  assert.equal(a.requests[1].history[0].answer, "Which programme?");
  assert.deepEqual(b.requests[0].history, []);
});

test("local source without a URL keeps its answer and omits the external link", async () => {
  const app = await browser({ response: () => {
    const result = answer();
    result.citations[0].url = "";
    result.citations[0].page = null;
    result.citations[0].section = "Local document section";
    return result;
  } });
  await app.ask("Question using a local document");
  await app.ask("Follow-up question");
  assert.equal(app.requests[1].history[0].answer, "Synthetic response");
  assert.equal(app.get("results").querySelector(".source-url"), null);
});
