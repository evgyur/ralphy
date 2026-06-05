// Hermes integration helpers.
//
// Ralphy is often run from inside Hermes profiles where provider credentials
// already live in ~/.hermes/.env and OpenAI Codex OAuth lives in
// ~/.hermes/auth.json. This file imports those credentials without requiring
// users to duplicate keys into a Ralphy-local .env or ~/.codex/auth.json.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export type HermesCodexAuth = {
  accessToken: string;
  accountId?: string;
  path: string;
};

function hermesImportDisabled(): boolean {
  return process.env.RALPHY_DISABLE_HERMES_IMPORT === "1" || process.env.NODE_ENV === "test";
}

function candidateHermesHomes(): string[] {
  const homes = new Set<string>();
  const add = (value?: string) => {
    if (value) homes.add(path.resolve(value));
  };

  add(process.env.HERMES_HOME);
  add(path.join(os.homedir(), ".hermes"));

  // Hermes gateway sessions may run with HOME pointed at
  // ~/.hermes/profiles/<name>/home. Walk back to the profile and root homes.
  const parts = path.resolve(os.homedir()).split(path.sep);
  const profileIdx = parts.lastIndexOf("profiles");
  if (profileIdx > 0 && parts[profileIdx - 1] === ".hermes") {
    add(parts.slice(0, profileIdx + 2).join(path.sep));
    add(parts.slice(0, profileIdx).join(path.sep));
  }

  // Common service account fallback on Linux hosts.
  add("/home/hermes/.hermes");

  return [...homes];
}

function parseEnvLine(line: string): { key: string; value: string } | null {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) return null;
  const eq = trimmed.indexOf("=");
  if (eq === -1) return null;
  const key = trimmed.slice(0, eq).trim();
  if (!key) return null;
  const value = trimmed
    .slice(eq + 1)
    .trim()
    .replace(/^['"]|['"]$/g, "");
  return { key, value };
}

export function loadHermesEnv(): void {
  if (hermesImportDisabled()) return;
  for (const home of candidateHermesHomes()) {
    const envPath = path.join(home, ".env");
    let raw: string;
    try {
      raw = fs.readFileSync(envPath, "utf8");
    } catch {
      continue;
    }
    for (const line of raw.split("\n")) {
      const parsed = parseEnvLine(line);
      if (!parsed) continue;
      if (!process.env[parsed.key]) process.env[parsed.key] = parsed.value;
    }
  }
}

function authFromObject(raw: unknown, authPath: string): HermesCodexAuth | null {
  const root = raw as {
    providers?: { "openai-codex"?: { tokens?: Record<string, string> } };
    codex?: Record<string, string>;
    credential_pool?: { "openai-codex"?: Array<Record<string, string>> };
  };
  const candidates = [
    root.providers?.["openai-codex"]?.tokens,
    root.codex,
    ...(root.credential_pool?.["openai-codex"] ?? []),
  ];
  for (const c of candidates) {
    const accessToken = c?.access_token;
    if (!accessToken) continue;
    return {
      accessToken,
      accountId: c.account_id,
      path: authPath,
    };
  }
  return null;
}

export function loadHermesCodexAuth(): HermesCodexAuth | null {
  if (hermesImportDisabled()) return null;
  for (const home of candidateHermesHomes()) {
    const authPath = path.join(home, "auth.json");
    try {
      const raw = JSON.parse(fs.readFileSync(authPath, "utf8"));
      const auth = authFromObject(raw, authPath);
      if (auth) return auth;
    } catch {
      // Try the next Hermes home.
    }
  }
  return null;
}
