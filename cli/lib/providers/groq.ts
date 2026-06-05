import type { RalphyConnector } from "./types.js";

export const groqConnector: RalphyConnector = {
  id: "groq",
  label: "Groq",
  envVar: "GROQ_API_KEY",
  signupUrl: "https://console.groq.com/keys",
  capabilities: ["transcribe"],
  available() {
    return Boolean(process.env.GROQ_API_KEY);
  },
};
