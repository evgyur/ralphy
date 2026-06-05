// Capability registry — maps env vars to pipeline features.
//
// Codex OAuth covers the default text / image path. Direct OpenAI and
// OpenRouter remain optional fallbacks. ElevenLabs covers voice + music.
//
// Usage:
//   import { requireCapability, hasCapability, getCapabilityStatus } from "./capabilities.js";
//
//   requireCapability("voiceover-elevenlabs");          // throws clean error if missing
//   if (hasCapability("llm-openrouter")) { ... }        // optional path

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export type CapabilityId =
  | "voiceover-elevenlabs"
  | "llm-codex"
  | "llm-openai"
  | "llm-openrouter"
  | "audio-groq";

export type CapabilityCategory = "media" | "voice" | "music" | "llm";

export type Capability = {
  id: CapabilityId;
  label: string;
  description: string;
  /** Env var that unlocks it. */
  envVar: string;
  category: CapabilityCategory;
  /** Where to obtain the key — shown in setup wizard. */
  signupUrl: string;
  /** Required for the core text/image/audio pipeline. */
  required: boolean;
  /** False for keyless/local-login checks that setup should not prompt for. */
  configuredBySetup?: boolean;
};

export const CAPABILITIES: Capability[] = [
  {
    id: "llm-codex",
    label: "Codex OAuth",
    description:
      "Local Codex ChatGPT login — covers GPT-5.5 text/vision work and GPT Image 2 image generation without an OpenAI API key.",
    envVar: "CODEX_HOME",
    category: "llm",
    signupUrl: "https://developers.openai.com/codex/cli/",
    required: true,
    configuredBySetup: false,
  },
  {
    id: "llm-openai",
    label: "OpenAI",
    description:
      "Optional direct OpenAI key fallback — covers GPT-5.5 text/vision work and GPT Image 2 image generation.",
    envVar: "OPENAI_API_KEY",
    category: "llm",
    signupUrl: "https://platform.openai.com/api-keys",
    required: false,
  },
  {
    id: "llm-openrouter",
    label: "OpenRouter",
    description:
      "Optional fallback key — covers video generation (kling-v3.0-pro, veo-3.1, seedance-2.0), " +
      "Gemini/OpenRouter image models, fallback LLM/vision, and legacy whisper-1 transcription.",
    envVar: "OPENROUTER_API_KEY",
    category: "llm",
    signupUrl: "https://openrouter.ai/keys",
    required: false,
  },
  {
    id: "audio-groq",
    label: "Groq Whisper",
    description:
      "Optional fast Whisper transcription via Groq (`whisper-large-v3-turbo`) for the `groq` captions backend.",
    envVar: "GROQ_API_KEY",
    category: "voice",
    signupUrl: "https://console.groq.com/keys",
    required: false,
  },
  {
    id: "voiceover-elevenlabs",
    label: "ElevenLabs",
    description:
      "Voiceover (eleven_multilingual_v2 RU, eleven_v3 EN premium) and music " +
      "(ElevenLabs Music — instrumental beds via /v1/music endpoint).",
    envVar: "ELEVENLABS_API_KEY",
    category: "voice",
    signupUrl: "https://elevenlabs.io/app/settings/api-keys",
    required: true,
  },
];

export function hasCapability(id: CapabilityId): boolean {
  const cap = CAPABILITIES.find((c) => c.id === id);
  if (!cap) return false;
  if (id === "llm-codex") {
    try {
      const authPath = path.join(process.env.CODEX_HOME || path.join(os.homedir(), ".codex"), "auth.json");
      const raw = JSON.parse(fs.readFileSync(authPath, "utf8")) as {
        tokens?: { access_token?: string; account_id?: string };
      };
      return Boolean(raw.tokens?.access_token && raw.tokens?.account_id);
    } catch {
      return false;
    }
  }
  return Boolean(process.env[cap.envVar]);
}

export function requireCapability(id: CapabilityId): void {
  if (hasCapability(id)) return;
  const cap = CAPABILITIES.find((c) => c.id === id);
  if (!cap) throw new Error(`Unknown capability: ${id}`);
  throw new Error(
    `Capability "${cap.label}" is not configured.\n` +
      `  Required env var: ${cap.envVar}\n` +
      `  Get a key at: ${cap.signupUrl}\n` +
      `Run "ralphy setup" to configure interactively.`,
  );
}

export type CapabilityStatus = Capability & { enabled: boolean };

export function getCapabilityStatus(): CapabilityStatus[] {
  return CAPABILITIES.map((c) => ({ ...c, enabled: hasCapability(c.id) }));
}
