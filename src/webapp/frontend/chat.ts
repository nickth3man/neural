const API_CHAT = "/api/chat";
const API_CHAT_STREAM = "/api/chat/stream";
const DEFAULT_TOP_K = 5;
const DEFAULT_RERANK_TOP_N = 20;
const DEFAULT_HYBRID_LEXICAL_K = 20;
const DEFAULT_RRF_K = 60;
const JSON_HEADERS: HeadersInit = { "Content-Type": "application/json" };
const MAX_HISTORY_ITEMS = 40;
const RETRYABLE_STATUS = new Set([502, 503, 504]);
const RETRY_DELAYS_MS = [300, 800];

type ChatRole = "user" | "assistant";

interface ChatTurn {
  role: ChatRole;
  content: string;
}

interface CitationPayload {
  rank?: number | null;
  episode_title?: string | null;
  source_file?: string | null;
  start_timestamp?: string | null;
  end_timestamp?: string | null;
  score?: number | string | null;
  chunk_text?: string | null;
  metadata?: Record<string, unknown> | null;
}

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error(`Missing element #${id}`);
  }
  return node as T;
}

const ui = {
  q: el<HTMLTextAreaElement>("q"),
  send: el<HTMLButtonElement>("send"),
  stop: el<HTMLButtonElement>("stop"),
  retrievalOnly: el<HTMLInputElement>("retrievalOnly"),
  streaming: el<HTMLInputElement>("streaming"),
  rerank: el<HTMLInputElement>("rerank"),
  hybrid: el<HTMLInputElement>("hybrid"),
  topK: el<HTMLInputElement>("topK"),
  rerankTopN: el<HTMLInputElement>("rerankTopN"),
  hybridLexicalK: el<HTMLInputElement>("hybridLexicalK"),
  rrfK: el<HTMLInputElement>("rrfK"),
  episodeType: el<HTMLInputElement>("episodeType"),
  guestName: el<HTMLInputElement>("guestName"),
  speaker: el<HTMLInputElement>("speaker"),
  team: el<HTMLInputElement>("team"),
  topic: el<HTMLInputElement>("topic"),
  sourceFile: el<HTMLInputElement>("sourceFile"),
  answerBox: el<HTMLDivElement>("answerBox"),
  answer: el<HTMLDivElement>("answer"),
  err: el<HTMLDivElement>("err"),
  citeBox: el<HTMLDivElement>("citeBox"),
  citations: el<HTMLDivElement>("citations"),
};

let history: ChatTurn[] = [];
let activeAbort: AbortController | null = null;
let pendingUserMessage = "";

function trimHistory(): void {
  while (history.length > MAX_HISTORY_ITEMS) {
    history.shift();
  }
}

function appendToHistory(userMsg: string, assistantPlain: string): void {
  history.push({ role: "user", content: userMsg });
  history.push({ role: "assistant", content: assistantPlain });
  trimHistory();
}

function newAbortSignal(): AbortSignal {
  activeAbort?.abort();
  activeAbort = new AbortController();
  return activeAbort.signal;
}

function abortInFlight(): void {
  activeAbort?.abort();
}

function setLoading(loading: boolean): void {
  ui.send.disabled = loading;
  ui.send.classList.toggle("btn-primary--loading", loading);
  ui.send.setAttribute("aria-busy", loading ? "true" : "false");
  ui.stop.hidden = !loading;
}

function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

function trimToNull(s: string): string | null {
  const t = String(s).trim();
  return t ? t : null;
}

function asText(value: unknown): string {
  return value == null ? "" : String(value);
}

function readPositiveInt(inputEl: HTMLInputElement, fallback: number): number {
  const n = parseInt(inputEl.value, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function buildRequestBody(message: string): Record<string, unknown> {
  return {
    message,
    top_k: readPositiveInt(ui.topK, DEFAULT_TOP_K),
    retrieval_only: ui.retrievalOnly.checked,
    history: history.map((t) => ({ role: t.role, content: t.content })),
    episode_type: trimToNull(ui.episodeType.value),
    guest_name: trimToNull(ui.guestName.value),
    speaker: trimToNull(ui.speaker.value),
    team: trimToNull(ui.team.value),
    topic: trimToNull(ui.topic.value),
    source_file: trimToNull(ui.sourceFile.value),
    rerank: ui.rerank.checked,
    rerank_top_n: readPositiveInt(ui.rerankTopN, DEFAULT_RERANK_TOP_N),
    hybrid: ui.hybrid.checked,
    hybrid_lexical_k: readPositiveInt(ui.hybridLexicalK, DEFAULT_HYBRID_LEXICAL_K),
    rrf_k: readPositiveInt(ui.rrfK, DEFAULT_RRF_K),
  };
}

async function fetchWithRetry(url: string, init: RequestInit): Promise<Response> {
  const { signal } = init;
  let last: Response | undefined;
  for (let attempt = 0; attempt <= 2; attempt += 1) {
    if (signal?.aborted) {
      throw new DOMException("Aborted", "AbortError");
    }
    const res = await fetch(url, init);
    last = res;
    if (res.ok || !RETRYABLE_STATUS.has(res.status) || attempt === 2) {
      return res;
    }
    const delayMs = RETRY_DELAYS_MS[attempt] ?? 800;
    await new Promise<void>((resolve, reject) => {
      const t = window.setTimeout(resolve, delayMs);
      signal?.addEventListener(
        "abort",
        () => {
          window.clearTimeout(t);
          reject(new DOMException("Aborted", "AbortError"));
        },
        { once: true },
      );
    });
  }
  return last as Response;
}

function postJsonWithRetry(url: string, body: object, signal: AbortSignal): Promise<Response> {
  return fetchWithRetry(url, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
    signal,
  });
}

function clearAnswerElement(): void {
  ui.answer.classList.remove("answer-content--rich");
  ui.answer.innerHTML = "";
}

/** Clears error/citations and shows panels. Skips placeholder when streaming (answer starts empty). */
function resetResponsePanels(options: { streaming: boolean }): void {
  ui.err.textContent = "";
  ui.answerBox.hidden = false;
  ui.citeBox.hidden = false;
  clearAnswerElement();
  ui.answer.textContent = options.streaming ? "" : "…";
  ui.citations.replaceChildren();
}

function parseSseBlock(rawBlock: string): { eventName: string; payload: Record<string, unknown> } | null {
  const lines = rawBlock.split("\n");
  let eventName = "message";
  const dataParts: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event: ")) {
      eventName = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      dataParts.push(line.slice(6));
    }
  }
  if (dataParts.length === 0) {
    return null;
  }
  const dataJoined = dataParts.join("\n");
  try {
    return { eventName, payload: JSON.parse(dataJoined) as Record<string, unknown> };
  } catch {
    return null;
  }
}

function showStreamError(message: string): void {
  ui.err.textContent = message;
}

function applyAnswerDisplay(answerPlain: string, html: string | null): void {
  if (html) {
    ui.answer.classList.add("answer-content--rich");
    ui.answer.innerHTML = html;
  } else {
    ui.answer.classList.remove("answer-content--rich");
    ui.answer.textContent = answerPlain;
  }
}

function handleSseDone(payload: Record<string, unknown>): void {
  const err = typeof payload.error === "string" ? payload.error : undefined;
  if (err) {
    showStreamError(err);
  }
  const answerPlain = typeof payload.answer === "string" ? payload.answer : "";
  const html = typeof payload.answer_html === "string" ? payload.answer_html : null;
  applyAnswerDisplay(answerPlain, html);
  if (!err) {
    appendToHistory(pendingUserMessage, answerPlain);
  }
}

function applySsePayload(eventName: string, payload: Record<string, unknown>): void {
  if (eventName === "citations") {
    renderCitations((payload.citations as CitationPayload[]) ?? []);
    return;
  }
  if (eventName === "answer") {
    ui.answer.textContent += typeof payload.delta === "string" ? payload.delta : "";
    return;
  }
  if (eventName === "error") {
    showStreamError(typeof payload.message === "string" ? payload.message : "Streaming failed");
    return;
  }
  if (eventName === "done") {
    handleSseDone(payload);
  }
}

function applyParsedSseBlocks(blocks: string[]): void {
  for (const block of blocks) {
    if (!block.trim()) {
      continue;
    }
    const parsed = parseSseBlock(block);
    if (parsed) {
      applySsePayload(parsed.eventName, parsed.payload);
    }
  }
}

function flushSseBuffer(buffer: string, finalize: boolean): string {
  const events = buffer.split("\n\n");
  if (!finalize) {
    const rest = events.pop() ?? "";
    applyParsedSseBlocks(events);
    return rest;
  }
  applyParsedSseBlocks(events);
  return "";
}

async function readErrorDetail(res: Response): Promise<string> {
  const text = await res.text();
  if (!text) {
    return res.statusText;
  }
  try {
    const data = JSON.parse(text) as { detail?: string };
    return data.detail ?? res.statusText;
  } catch {
    return text;
  }
}

function setAnswerError(message: string): void {
  clearAnswerElement();
  ui.answer.textContent = "";
  ui.err.textContent = message;
}

async function fetchJsonResponse(message: string, signal: AbortSignal): Promise<void> {
  const res = await postJsonWithRetry(API_CHAT, buildRequestBody(message), signal);
  if (!res.ok) {
    setAnswerError(await readErrorDetail(res));
    return;
  }
  const data = (await res.json()) as Record<string, unknown>;
  const text = typeof data.answer === "string" ? data.answer : "";
  const html = typeof data.answer_html === "string" ? data.answer_html : null;
  applyAnswerDisplay(text || "(empty)", html);
  const errStr = typeof data.error === "string" ? data.error : "";
  ui.err.textContent = errStr;
  renderCitations((data.citations as CitationPayload[]) ?? []);
  if (!errStr) {
    appendToHistory(message, text);
  }
}

async function streamResponse(message: string, signal: AbortSignal): Promise<void> {
  pendingUserMessage = message;
  const res = await postJsonWithRetry(API_CHAT_STREAM, buildRequestBody(message), signal);
  if (!res.ok) {
    setAnswerError(await readErrorDetail(res));
    return;
  }
  if (!res.body) {
    setAnswerError("Streaming not supported in this browser");
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      buffer = flushSseBuffer(buffer, false);
    }
    flushSseBuffer(buffer, true);
  } finally {
    reader.releaseLock();
  }
}

function appendCitationMetadata(container: HTMLElement, meta: Record<string, unknown> | null | undefined): void {
  if (!meta) {
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "cite-metadata";
  const strong = document.createElement("strong");
  strong.textContent = "Metadata:";
  wrap.appendChild(strong);
  const parts: string[] = [];
  if (meta.episode_type) {
    parts.push(` episode ${String(meta.episode_type)}`);
  }
  if (meta.speaker) {
    parts.push(` speaker ${String(meta.speaker)}`);
  }
  const teamsRaw = meta.mentioned_teams;
  const teams = Array.isArray(teamsRaw)
    ? teamsRaw.filter(Boolean).join(", ")
    : "";
  if (teams) {
    parts.push(` teams ${teams}`);
  }
  if (parts.length) {
    wrap.appendChild(document.createTextNode(parts.join("")));
  }
  container.appendChild(wrap);
}

function appendMetaItem(row: HTMLElement, icon: string, text: string): void {
  const span = document.createElement("span");
  span.className = "cite-meta-item";
  span.textContent = `${icon} ${text}`;
  row.appendChild(span);
}

function formatScore(score: unknown): string {
  return typeof score === "number" ? score.toFixed(3) : String(score ?? "");
}

function createCiteItemElement(c: CitationPayload): HTMLElement {
  const detailsEl = document.createElement("details");
  detailsEl.className = "cite-item";
  detailsEl.open = true;

  const summary = document.createElement("summary");
  summary.className = "cite-summary";

  const rank = document.createElement("span");
  rank.className = "cite-rank";
  rank.textContent = asText(c.rank);

  const headerContent = document.createElement("div");
  headerContent.className = "cite-header-content";

  const episode = document.createElement("div");
  episode.className = "cite-episode";
  episode.textContent = asText(c.episode_title);

  const metaRow = document.createElement("div");
  metaRow.className = "cite-meta";
  appendMetaItem(metaRow, "📁", asText(c.source_file));
  const start = asText(c.start_timestamp);
  const end = asText(c.end_timestamp);
  appendMetaItem(metaRow, "⏱️", `${start}–${end}`);

  headerContent.appendChild(episode);
  headerContent.appendChild(metaRow);

  const scoreEl = document.createElement("span");
  scoreEl.className = "cite-score";
  scoreEl.textContent = `score ${formatScore(c.score)}`;

  const expand = document.createElement("span");
  expand.className = "cite-expand";

  summary.appendChild(rank);
  summary.appendChild(headerContent);
  summary.appendChild(scoreEl);
  summary.appendChild(expand);

  const detailsInner = document.createElement("div");
  detailsInner.className = "cite-details";
  appendCitationMetadata(detailsInner, c.metadata ?? undefined);
  const quote = document.createElement("p");
  quote.className = "cite-text";
  quote.textContent = asText(c.chunk_text);
  detailsInner.appendChild(quote);

  detailsEl.appendChild(summary);
  detailsEl.appendChild(detailsInner);
  return detailsEl;
}

function renderCitations(citations: CitationPayload[]): void {
  ui.citations.replaceChildren();
  for (const c of citations) {
    ui.citations.appendChild(createCiteItemElement(c));
  }
}

ui.send.addEventListener("click", async () => {
  const message = ui.q.value.trim();
  if (!message) {
    return;
  }
  const signal = newAbortSignal();
  setLoading(true);
  const isStreaming = ui.streaming.checked;
  resetResponsePanels({ streaming: isStreaming });
  try {
    if (isStreaming) {
      await streamResponse(message, signal);
    } else {
      await fetchJsonResponse(message, signal);
    }
  } catch (e) {
    if (isAbortError(e)) {
      ui.err.textContent = "Cancelled";
    } else {
      setAnswerError(String(e));
    }
  } finally {
    setLoading(false);
  }
});

ui.stop.addEventListener("click", () => {
  abortInFlight();
});
