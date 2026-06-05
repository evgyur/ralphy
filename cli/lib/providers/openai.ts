// OpenAI connector — direct OpenAI API path for text / image.
//
// Ralphy's registry resolves this before OpenRouter when OPENAI_API_KEY is
// present, so text synthesis and image generation can run without OpenRouter.
// Video remains on OpenRouter for now because the existing Ralphy video flow is
// built around OpenRouter's async video job contract.

import { logGeneration } from "../gen-log.js";
import {
  assetPath,
  protectExistingAsset,
  writeImageFromUrlOrDataUri,
  resolveImageRef,
  logFailure,
  requireProviderKey,
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

const ID = "openai";
const LABEL = "OpenAI";
const ENV_VAR = "OPENAI_API_KEY";
const SIGNUP_URL = "https://platform.openai.com/api-keys";
const BASE_URL = "https://api.openai.com/v1";

const DEFAULT_LLM_MODEL = "gpt-5.5";
const DEFAULT_IMAGE_MODEL = "gpt-image-2";

const IMAGE_PRICE_PER_GEN: Record<string, number> = {
  "gpt-image-2": 0.20,
  "gpt-image-1.5": 0.12,
  "chatgpt-image-latest": 0.12,
};
const IMAGE_PRICE_FALLBACK = 0.15;

function requireKey(): void {
  requireProviderKey({ envVar: ENV_VAR, label: LABEL, signupUrl: SIGNUP_URL });
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

function openAiContent(content: CallLLMOptions["messages"][number]["content"]): Array<Record<string, unknown>> {
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

function responseText(json: unknown): string {
  const j = json as {
    output_text?: string;
    output?: Array<{ content?: Array<{ text?: string; type?: string }> }>;
  };
  if (typeof j.output_text === "string") return j.output_text;
  const parts: string[] = [];
  for (const item of j.output ?? []) {
    for (const c of item.content ?? []) {
      if (typeof c.text === "string") parts.push(c.text);
    }
  }
  return parts.join("\n");
}

// ─── text (Responses API) ───────────────────────────────────────────────────

export async function callLLM(opts: CallLLMOptions): Promise<CallLLMResult> {
  requireKey();
  const apiKey = process.env[ENV_VAR]!;
  const model = opts.model ?? DEFAULT_LLM_MODEL;
  const endpoint = opts.endpoint ?? "openai/responses";

  const instructions = opts.messages
    .filter((m) => m.role === "system")
    .map((m) => flattenContent(m.content))
    .join("\n\n");
  const input = opts.messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role,
      content: openAiContent(m.content),
    }));

  const body: Record<string, unknown> = {
    model,
    input,
    max_output_tokens: opts.maxTokens ?? 2048,
    temperature: opts.temperature ?? 0.3,
  };
  if (instructions) body.instructions = instructions;
  if (opts.jsonMode) {
    body.text = { format: { type: "json_object" } };
  }

  return retryTransient(
    async (attempt) => {
      const t0 = Date.now();
      const resp = await withConcurrency(ID, model, "text", () =>
        fetch(`${BASE_URL}/responses`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body),
        }),
      );
      const latencyMs = Date.now() - t0;

      if (!resp.ok) {
        const errText = await resp.text().catch(() => "");
        const message = `${ID} responses ${resp.status}: ${errText.slice(0, 500)}`;
        if (resp.status >= 400 && resp.status < 500) {
          throw new TerminalProviderError(message);
        }
        throw new Error(message);
      }

      const json = await resp.json();
      const text = responseText(json);
      if (!text) {
        throw new TransientPayloadError(
          `${ID} responses returned empty text on ${model}. Raw: ${JSON.stringify(json).slice(0, 500)}`,
        );
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
          latency_ms: latencyMs,
          attempt,
        });
      }

      return { text, raw: json, provider: ID, model, latencyMs };
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

// ─── image (Images API) ─────────────────────────────────────────────────────

function openAiImageSize(size: string): "1024x1024" | "1024x1536" | "1536x1024" {
  const m = size.match(/^(\d+)\s*x\s*(\d+)$/i);
  if (!m) return "1024x1536";
  const w = Number(m[1]);
  const h = Number(m[2]);
  if (!isFinite(w) || !isFinite(h) || w <= 0 || h <= 0) return "1024x1536";
  if (Math.abs(w - h) / Math.max(w, h) < 0.05) return "1024x1024";
  return w > h ? "1536x1024" : "1024x1536";
}

function dataUriToBlob(dataUri: string): { blob: Blob; filename: string } {
  const m = dataUri.match(/^data:([^;,]+)?(;base64)?,(.*)$/s);
  if (!m) throw new Error("Malformed data: URI in image reference");
  const mime = m[1] || "image/png";
  const raw = m[3] || "";
  const bytes = m[2]
    ? Buffer.from(raw, "base64")
    : Buffer.from(decodeURIComponent(raw), "utf8");
  const ext = mime.includes("jpeg") ? "jpg" : mime.includes("webp") ? "webp" : "png";
  return { blob: new Blob([bytes], { type: mime }), filename: `ref.${ext}` };
}

async function refToBlob(ref: string, index: number): Promise<{ blob: Blob; filename: string }> {
  const resolved = await resolveImageRef(ref);
  if (resolved.startsWith("data:")) {
    const parsed = dataUriToBlob(resolved);
    return { ...parsed, filename: `ref-${index}.${parsed.filename.split(".").pop() ?? "png"}` };
  }
  if (resolved.startsWith("http://") || resolved.startsWith("https://")) {
    const resp = await fetch(resolved);
    if (!resp.ok) throw new Error(`Could not fetch image ref ${index}: ${resp.status}`);
    const mime = resp.headers.get("content-type") || "image/png";
    const bytes = Buffer.from(await resp.arrayBuffer());
    const ext = mime.includes("jpeg") ? "jpg" : mime.includes("webp") ? "webp" : "png";
    return { blob: new Blob([bytes], { type: mime }), filename: `ref-${index}.${ext}` };
  }
  throw new Error(`Unsupported resolved image reference: ${resolved.slice(0, 80)}`);
}

export async function generateImage(input: GenerateImageInput): Promise<GenerateResult> {
  requireKey();
  const t0 = Date.now();
  const apiKey = process.env[ENV_VAR]!;
  const model = input.model ?? DEFAULT_IMAGE_MODEL;
  const size = openAiImageSize(input.size ?? "1080x1920");
  const endpoint = input.refs && input.refs.length > 0 ? "images/edits" : "images/generations";

  const prompt = [
    input.prompt,
    input.negativePrompt ? `Negative prompt — avoid: ${input.negativePrompt}` : "",
  ].filter(Boolean).join("\n\n");

  type ImageNetResult = { url: string; rawJson: unknown };
  const net = await retryTransient<ImageNetResult & { _attempt: number }>(
    async (attempt) => {
      const submit = async () => {
        if (input.refs && input.refs.length > 0) {
          const form = new FormData();
          form.set("model", model);
          form.set("prompt", prompt);
          form.set("size", size);
          for (let i = 0; i < input.refs!.length; i += 1) {
            const { blob, filename } = await refToBlob(input.refs![i]!, i + 1);
            form.append("image[]", blob, filename);
          }
          return fetch(`${BASE_URL}/images/edits`, {
            method: "POST",
            headers: { Authorization: `Bearer ${apiKey}` },
            body: form,
            signal: input.signal,
          });
        }

        return fetch(`${BASE_URL}/images/generations`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model,
            prompt,
            size,
            n: 1,
            response_format: "b64_json",
          }),
          signal: input.signal,
        });
      };

      const resp = await withConcurrency(ID, model, "image", submit);
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        const message = `${ID} ${endpoint} ${resp.status}: ${text.slice(0, 500)}`;
        if (resp.status >= 400 && resp.status < 500) {
          throw new TerminalProviderError(message);
        }
        throw new Error(message);
      }

      const json = (await resp.json()) as {
        data?: Array<{ b64_json?: string; url?: string }>;
      };
      const item = json.data?.[0];
      const url = item?.b64_json ? `data:image/png;base64,${item.b64_json}` : item?.url;
      if (!url) {
        throw new TransientPayloadError(
          `${ID} ${endpoint} response had no data[0].b64_json or url. Raw: ${JSON.stringify(json).slice(0, 500)}`,
        );
      }
      return { url, rawJson: json, _attempt: attempt };
    },
    {
      noRetry: input.noRetry,
      onTransientFailure: async (err, attempt) => {
        await logFailure(input, ID, model, "image", { prompt, size, refs: input.refs?.length ?? 0 }, err, t0, attempt);
      },
    },
  );

  const imgDest = assetPath(input.projectId, "images", `${input.slot}.png`);
  await protectExistingAsset(imgDest, input.overwrite);
  const localPath = await writeImageFromUrlOrDataUri(net.url, imgDest);

  const result: GenerateResult = {
    url: net.url,
    localPath,
    costUsd: IMAGE_PRICE_PER_GEN[model] ?? IMAGE_PRICE_FALLBACK,
    latencyMs: Date.now() - t0,
    model,
  };
  await logGeneration(input.projectId, {
    slot: input.slot,
    provider: ID,
    model,
    endpoint: `${ID}/${endpoint}`,
    kind: "image",
    input: { slot: input.slot, project: input.projectId, prompt: input.prompt, size, refs: input.refs ?? [] },
    output: { url: net.url.startsWith("data:") ? "[data-uri]" : net.url, local: localPath },
    status: "ok",
    latency_ms: result.latencyMs,
    cost_usd: result.costUsd,
    attempt: net._attempt,
    note: input.note ?? input.slot,
  });
  return result;
}

export const openaiConnector: RalphyConnector = {
  id: ID,
  label: LABEL,
  envVar: ENV_VAR,
  signupUrl: SIGNUP_URL,
  capabilities: ["text", "image"],
  available: () => Boolean(process.env[ENV_VAR]),
  callLLM,
  generateImage,
};
