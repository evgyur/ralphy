// Provider connector registry — capability matrix + `--provider` resolution.
//
// First slice of notes/ideas/005 (pluggable provider spec). Locks the two
// bundled connectors (OpenAI + OpenRouter + ElevenLabs), the capability matrix, and the
// resolution rules: explicit `--provider` validated against the matrix +
// availability; otherwise first available connector that serves the capability.
//
// `resolveConnector` refuses via `raiseError`, which writes to stderr and calls
// `process.exit`. We stub both: exit throws a sentinel so control halts (the
// function is typed `never` past that point), and stderr.write is silenced.

import { describe, test, expect, spyOn, beforeEach, afterEach } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  listConnectors,
  connectorsFor,
  providerMatrix,
  resolveConnector,
} from "../../cli/lib/providers/registry.js";

const ENV_KEYS = [
  "CODEX_HOME",
  "OPENAI_API_KEY",
  "GROQ_API_KEY",
  "OPENROUTER_API_KEY",
  "ELEVENLABS_API_KEY",
  "RALPHY_DISABLE_HERMES_IMPORT",
] as const;
const saved: Record<string, string | undefined> = {};
let tmp: string;

beforeEach(() => {
  for (const k of ENV_KEYS) saved[k] = process.env[k];
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ralphy-provider-codex-"));
  process.env.CODEX_HOME = path.join(tmp, "missing-codex-home");
  process.env.RALPHY_DISABLE_HERMES_IMPORT = "1";
});
afterEach(() => {
  for (const k of ENV_KEYS) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
  fs.rmSync(tmp, { recursive: true, force: true });
});

function enableCodexOAuth(): void {
  const home = path.join(tmp, "codex-home");
  fs.mkdirSync(home, { recursive: true });
  fs.writeFileSync(
    path.join(home, "auth.json"),
    JSON.stringify({ tokens: { access_token: "codex-access", account_id: "codex-account" } }),
  );
  process.env.CODEX_HOME = home;
}

/** Run `fn`, capturing the `process.exit` code raiseError would use. */
function expectRefusal(fn: () => unknown): number {
  const exitSpy = spyOn(process, "exit").mockImplementation(((code?: number) => {
    throw new Error(`__exit__:${code ?? 0}`);
  }) as never);
  const errSpy = spyOn(process.stderr, "write").mockImplementation(() => true);
  try {
    fn();
    throw new Error("expected a refusal, got none");
  } catch (e) {
    const msg = (e as Error).message;
    const m = msg.match(/^__exit__:(\d+)$/);
    if (!m) throw e; // a non-exit error — re-surface it
    return Number(m[1]);
  } finally {
    exitSpy.mockRestore();
    errSpy.mockRestore();
  }
}

describe("provider registry — matrix", () => {
  test("bundled connectors in priority order: codex, openai, groq, openrouter, elevenlabs", () => {
    expect(listConnectors().map((c) => c.id)).toEqual([
      "codex",
      "openai",
      "groq",
      "openrouter",
      "elevenlabs",
    ]);
  });

  test("codex/openai serve text/image; groq serves transcribe; openrouter serves text/image/video/transcribe; elevenlabs serves voice/music/sfx/transcribe", () => {
    const byId = Object.fromEntries(providerMatrix().map((p) => [p.id, p.capabilities]));
    expect(byId.codex).toEqual(["text", "image"]);
    expect(byId.openai).toEqual(["text", "image"]);
    expect(byId.groq).toEqual(["transcribe"]);
    expect(byId.openrouter).toEqual(["text", "image", "video", "transcribe"]);
    expect(byId.elevenlabs).toEqual(["voice", "music", "sfx", "transcribe"]);
  });

  test("connectorsFor maps a capability to the providers that serve it", () => {
    expect(connectorsFor("image").map((c) => c.id)).toEqual(["codex", "openai", "openrouter"]);
    expect(connectorsFor("voice").map((c) => c.id)).toEqual(["elevenlabs"]);
    // transcribe is served by groq + legacy openrouter + elevenlabs
    expect(connectorsFor("transcribe").map((c) => c.id)).toEqual([
      "groq",
      "openrouter",
      "elevenlabs",
    ]);
  });
});

describe("provider registry — resolution (default = first available)", () => {
  test("image defaults to codex, voice to elevenlabs when all providers present", () => {
    enableCodexOAuth();
    process.env.OPENAI_API_KEY = "z";
    process.env.GROQ_API_KEY = "g";
    process.env.OPENROUTER_API_KEY = "x";
    process.env.ELEVENLABS_API_KEY = "y";
    expect(resolveConnector("image").id).toBe("codex");
    expect(resolveConnector("voice").id).toBe("elevenlabs");
  });

  test("explicit --provider that serves the capability and has its key wins", () => {
    enableCodexOAuth();
    process.env.OPENAI_API_KEY = "z";
    process.env.OPENROUTER_API_KEY = "x";
    expect(resolveConnector("image", "openrouter").id).toBe("openrouter");
  });

  test("image falls back to openai when codex oauth is absent", () => {
    process.env.OPENAI_API_KEY = "z";
    process.env.OPENROUTER_API_KEY = "x";
    expect(resolveConnector("image").id).toBe("openai");
  });

  test("image falls back to openrouter when codex and openai are absent", () => {
    delete process.env.OPENAI_API_KEY;
    process.env.OPENROUTER_API_KEY = "x";
    expect(resolveConnector("image").id).toBe("openrouter");
  });

  test("transcribe default picks groq when it is available", () => {
    process.env.GROQ_API_KEY = "g";
    process.env.OPENROUTER_API_KEY = "x";
    process.env.ELEVENLABS_API_KEY = "y";
    expect(resolveConnector("transcribe").id).toBe("groq");
  });

  test("transcribe falls back to openrouter when groq key is absent", () => {
    delete process.env.GROQ_API_KEY;
    process.env.OPENROUTER_API_KEY = "x";
    process.env.ELEVENLABS_API_KEY = "y";
    expect(resolveConnector("transcribe").id).toBe("openrouter");
  });

  test("transcribe falls back to elevenlabs when groq and openrouter keys are absent", () => {
    delete process.env.GROQ_API_KEY;
    delete process.env.OPENROUTER_API_KEY;
    process.env.ELEVENLABS_API_KEY = "y";
    expect(resolveConnector("transcribe").id).toBe("elevenlabs");
  });
});

describe("provider registry — refusals", () => {
  test("unknown provider id refuses (user error, exit 2)", () => {
    enableCodexOAuth();
    expect(expectRefusal(() => resolveConnector("image", "falai"))).toBe(2);
  });

  test("capability mismatch refuses (elevenlabs cannot do image, exit 2)", () => {
    enableCodexOAuth();
    process.env.ELEVENLABS_API_KEY = "y";
    expect(expectRefusal(() => resolveConnector("image", "elevenlabs"))).toBe(2);
  });

  test("explicit provider with missing key refuses (provider error, exit 3)", () => {
    process.env.CODEX_HOME = path.join(tmp, "missing-codex-home");
    expect(expectRefusal(() => resolveConnector("image", "codex"))).toBe(3);
  });

  test("default selection with no available provider refuses (exit 3)", () => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENROUTER_API_KEY;
    expect(expectRefusal(() => resolveConnector("image"))).toBe(3);
  });
});
