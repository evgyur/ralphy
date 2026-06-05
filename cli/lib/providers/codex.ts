// Codex OAuth connector — uses the local Codex CLI ChatGPT login.
//
// Auth lives in ~/.codex/auth.json (or CODEX_HOME/auth.json). This connector
// calls the same Codex Responses backend that the Codex CLI uses for ChatGPT
// login sessions, so Ralphy can run GPT-5.5 text and GPT Image 2 without a
// separate OPENAI_API_KEY.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { logGeneration } from "../gen-log.js";
import { loadHermesCodexAuth } from "../hermes-env.js";
import {
  assetPath,
  protectExistingAsset,
  writeImageFromUrlOrDataUri,
  logFailure,
  retryTransient,
  TerminalProviderError,
  TransientPayloadError,
} from "./shared.js";
import { withConcurrency } from "./concurrency.js";
import type {
  RalphyConnector,
  CallLLMOptions,
  CallLLMResult,
  GenerateImageInput,
  GenerateResult,
} from "./types.js";

const ID = "codex";
const LABEL = "Codex OAuth";
const ENV_VAR = "CODEX_HOME";
const SIGNUP_URL = "https://developers.openai.com/codex/cli/";
const BASE_URL = "https://chatgpt.com/backend-api/codex";

const DEFAULT_LLM_MODEL = "gpt-5.5";
const DEFAULT_IMAGE_MODEL = "gpt-image-2";
const DEFAULT_INSTRUCTIONS = "You are a concise assistant.";

const IMAGE_PRICE_PER_GEN: Record<string, number> = {
  "gpt-image-2": 0.20,
  "gpt-image-2-codex": 0.20,
  "gpt-image-1.5": 0.12,
};
const IMAGE_PRICE_FALLBACK = 0.15;

type CodexAuth = {
  accessToken: string;
  accountId?: string;
  path: string;
};

type SseEvent = {
  type?: string;
  delta?: string;
  text?: string;
  part?: { text?: string; type?: string };
  item?: {
    type?: string;
    result?: string;
    revised_prompt?: string;
    content?: Array<{ type?: string; text?: string }>;
  };
  response?: unknown;
};

export function codexHome(): string {
  return process.env.CODEX_HOME || path.join(os.homedir(), ".codex");
}

export function codexAuthPath(): string {
  return path.join(codexHome(), "auth.json");
}

export function loadCodexAuth(): CodexAuth | null {
  const authPath = codexAuthPath();
  try {
    const raw = JSON.parse(fs.readFileSync(authPath, "utf8")) as {
      auth_mode?: string;
      tokens?: {
        access_token?: string;
        account_id?: string;
      };
    };
    const accessToken = raw.tokens?.access_token;
    const accountId = raw.tokens?.account_id;
    if (!accessToken) return null;
    return { accessToken, accountId, path: authPath };
  } catch {
    return loadHermesCodexAuth();
  }
}

export function hasCodexOAuth(): boolean {
  return loadCodexAuth() !== null;
}

function requireCodexAuth(): CodexAuth {
  const auth = loadCodexAuth();
  if (auth) return auth;
  throw new TerminalProviderError(
    `Codex OAuth is not configured. Run "codex login" or "hermes login --provider openai-codex" and confirm ${codexAuthPath()} or ~/.hermes/auth.json contains ChatGPT auth tokens.`,
  );
}

function flattenContent(content: CallLLMOptions["messages"][number]["content"]): string {
  if (typeof content === "string") return content;
  return content
    .map((part) => {
      if (part.type === "text") return part.text;
      if (part.type === "image_url") return `[image: ${part.image_url.url}]`;
      if (part.type === "file") return `[file: ${part.file.filename}]`;
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

function codexContent(content: CallLLMOptions["messages"][number]["content"]): Array<Record<string, unknown>> {
  if (typeof content === "string") return [{ type: "input_text", text: content }];
  return content.map((part) => {
    if (part.type === "text") return { type: "input_text", text: part.text };
    if (part.type === "image_url") return { type: "input_image", image_url: part.image_url.url };
    if (part.type === "file") {
      return {
        type: "input_file",
        filename: part.file.filename,
        file_data: part.file.file_data,
      };
    }
    return { type: "input_text", text: "" };
  });
}

function parseSse(text: string): SseEvent[] {
  const events: SseEvent[] = [];
  for (const block of text.split(/\r?\n\r?\n/u)) {
    const dataLines = block
      .split(/\r?\n/u)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.replace(/^data:\s?/u, ""));
    if (dataLines.length === 0) continue;
    const data = dataLines.join("\n");
    if (!data || data === "[DONE]") continue;
    try {
      events.push(JSON.parse(data) as SseEvent);
    } catch {
      // Ignore keepalive / malformed event fragments.
    }
  }
  return events;
}

function textFromEvents(events: SseEvent[]): string {
  const deltas: string[] = [];
  const completed: string[] = [];
  for (const ev of events) {
    if (ev.type === "response.output_text.delta" && typeof ev.delta === "string") deltas.push(ev.delta);
    if (ev.type === "response.output_text.done" && typeof ev.text === "string") completed.push(ev.text);
    if (ev.type === "response.output_item.done" && ev.item?.type === "message") {
      for (const c of ev.item.content ?? []) {
        if (c.type === "output_text" && typeof c.text === "string") completed.push(c.text);
      }
    }
  }
  return deltas.length > 0 ? deltas.join("") : completed.at(-1) ?? "";
}

function imageResultFromEvents(events: SseEvent[]): string | null {
  for (const ev of events) {
    if (ev.item?.type === "image_generation_call" && typeof ev.item.result === "string") {
      return ev.item.result;
    }
  }
  return null;
}

function codexImageSize(size: string): "1024x1024" | "1024x1536" | "1536x1024" {
  const m = size.match(/^(\d+)\s*x\s*(\d+)$/i);
  if (!m) return "1024x1536";
  const w = Number(m[1]);
  const h = Number(m[2]);
  if (!isFinite(w) || !isFinite(h) || w <= 0 || h <= 0) return "1024x1536";
  if (Math.abs(w - h) / Math.max(w, h) < 0.05) return "1024x1024";
  return w > h ? "1536x1024" : "1024x1536";
}

async function codexResponses(body: Record<string, unknown>, model: string, kind: "text" | "image", signal?: AbortSignal): Promise<{ events: SseEvent[]; raw: string; latencyMs: number }> {
  const auth = requireCodexAuth();
  const t0 = Date.now();
  const resp = await withConcurrency(ID, model, kind, () =>
    fetch(`${BASE_URL}/responses`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${auth.accessToken}`,
        ...(auth.accountId ? { "chatgpt-account-id": auth.accountId } : {}),
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        "User-Agent": "ralphy-codex-oauth",
      },
      body: JSON.stringify(body),
      signal,
    }),
  );
  const raw = await resp.text().catch(() => "");
  const latencyMs = Date.now() - t0;
  if (!resp.ok) {
    const message = `${ID} responses ${resp.status}: ${raw.slice(0, 500)}`;
    if (resp.status >= 400 && resp.status < 500) throw new TerminalProviderError(message);
    throw new Error(message);
  }
  return { events: parseSse(raw), raw, latencyMs };
}

function baseBody(model: string, instructions: string, input: Array<Record<string, unknown>>): Record<string, unknown> {
  return {
    model,
    instructions: instructions || DEFAULT_INSTRUCTIONS,
    input,
    tool_choice: "auto",
    parallel_tool_calls: false,
    reasoning: { effort: "low", summary: "auto" },
    store: false,
    stream: true,
    include: [],
    text: { verbosity: "low" },
  };
}

export async function callLLM(opts: CallLLMOptions): Promise<CallLLMResult> {
  const model = opts.model ?? DEFAULT_LLM_MODEL;
  const endpoint = opts.endpoint ?? "codex/responses";
  const instructions =
    opts.messages
      .filter((m) => m.role === "system")
      .map((m) => flattenContent(m.content))
      .join("\n\n") || DEFAULT_INSTRUCTIONS;
  const input = opts.messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      type: "message",
      role: m.role,
      content: codexContent(m.content),
    }));
  const body = baseBody(model, instructions, input);
  body.tools = [];
  if (opts.jsonMode) body.text = { format: { type: "json_object" }, verbosity: "low" };

  return retryTransient(
    async (attempt) => {
      const res = await codexResponses(body, model, "text");
      const text = textFromEvents(res.events);
      if (!text) {
        throw new TransientPayloadError(`${ID} responses returned empty text. Raw: ${res.raw.slice(0, 500)}`);
      }
      if (opts.projectId) {
        await logGeneration(opts.projectId, {
          provider: ID,
          model,
          endpoint,
          kind: "text",
          input: { model, messages: opts.messages.length, slot: opts.slot, project: opts.projectId },
          output: { bytes: text.length },
          status: "ok",
          latency_ms: res.latencyMs,
          attempt,
        });
      }
      return { text, raw: res.raw, provider: ID, model, latencyMs: res.latencyMs };
    },
    {
      noRetry: opts.noRetry,
      onTransientFailure: opts.projectId
        ? async (err, attempt) => {
            await logGeneration(opts.projectId!, {
              provider: ID,
              model,
              endpoint,
              kind: "text",
              input: { model, messages: opts.messages.length, slot: opts.slot, project: opts.projectId },
              status: "error",
              error: err instanceof Error ? err.message : String(err),
              attempt,
              note: `transient retry ${attempt}`,
            });
          }
        : undefined,
    },
  );
}

export async function generateImage(input: GenerateImageInput): Promise<GenerateResult> {
  const t0 = Date.now();
  const requestedModel = input.model ?? DEFAULT_IMAGE_MODEL;
  const toolModel = requestedModel === "gpt-image-2-codex" ? requestedModel : requestedModel;
  const size = codexImageSize(input.size ?? "1080x1920");
  const prompt = [
    input.prompt,
    input.negativePrompt ? `Negative prompt - avoid: ${input.negativePrompt}` : "",
    input.refs?.length ? `Reference images were supplied: ${input.refs.join("\n")}` : "",
  ].filter(Boolean).join("\n\n");
  const body = baseBody("gpt-5.5", "Use the image generation tool exactly once. Do not answer with code.", [
    {
      type: "message",
      role: "user",
      content: [{ type: "input_text", text: prompt }],
    },
  ]);
  body.tools = [{ type: "image_generation", model: toolModel, size }];

  const net = await retryTransient<{ b64: string; raw: string; latencyMs: number; _attempt: number }>(
    async (attempt) => {
      const res = await codexResponses(body, "gpt-5.5", "image", input.signal);
      const b64 = imageResultFromEvents(res.events);
      if (!b64) {
        throw new TransientPayloadError(`${ID} image_generation returned no image result. Raw: ${res.raw.slice(0, 500)}`);
      }
      return { b64, raw: res.raw, latencyMs: res.latencyMs, _attempt: attempt };
    },
    {
      noRetry: input.noRetry,
      onTransientFailure: async (err, attempt) => {
        await logFailure(input, ID, requestedModel, "image", { prompt, size, refs: input.refs?.length ?? 0 }, err, t0, attempt);
      },
    },
  );

  const dataUri = `data:image/png;base64,${net.b64}`;
  const imgDest = assetPath(input.projectId, "images", `${input.slot}.png`);
  await protectExistingAsset(imgDest, input.overwrite);
  const localPath = await writeImageFromUrlOrDataUri(dataUri, imgDest);
  const result: GenerateResult = {
    url: dataUri,
    localPath,
    costUsd: IMAGE_PRICE_PER_GEN[requestedModel] ?? IMAGE_PRICE_FALLBACK,
    latencyMs: Date.now() - t0,
    model: requestedModel,
  };
  await logGeneration(input.projectId, {
    slot: input.slot,
    provider: ID,
    model: requestedModel,
    endpoint: `${ID}/responses:image_generation`,
    kind: "image",
    input: { slot: input.slot, project: input.projectId, prompt: input.prompt, size, refs: input.refs ?? [] },
    output: { url: "[data-uri]", local: localPath },
    status: "ok",
    latency_ms: result.latencyMs,
    cost_usd: result.costUsd,
    attempt: net._attempt,
    note: input.note ?? input.slot,
  });
  return result;
}

export const codexConnector: RalphyConnector = {
  id: ID,
  label: LABEL,
  envVar: ENV_VAR,
  signupUrl: SIGNUP_URL,
  capabilities: ["text", "image"],
  available: hasCodexOAuth,
  callLLM,
  generateImage,
};
