import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { textFromContent } from "./content.js";
import { pruneHistory } from "./history.js";
import { registerMemoryCommand } from "./menu.js";
import { MemoryStore } from "./memory-store.js";

function formatRetrieved(records: Array<{ source: string; text: string }>, maxChars: number): string {
  let used = 0;
  const sections: string[] = [];
  for (const record of records) {
    const prefix = `[${record.source}] `;
    const available = maxChars - used - prefix.length;
    if (available <= 0) break;
    const text = record.text.slice(0, available);
    sections.push(prefix + text);
    used += prefix.length + text.length;
  }
  return sections.join("\n\n");
}

export default function (pi: ExtensionAPI) {
  const store = new MemoryStore();
  let currentUserPrompt = "";

  pi.on("session_start", async (_event, ctx) => {
    try {
      await store.initialize(ctx.cwd);
    } catch (error) {
      ctx.ui.notify(`Falha ao inicializar memória Ollama: ${error instanceof Error ? error.message : String(error)}`, "warning");
    }
  });

  pi.on("input", async (event) => {
    currentUserPrompt = event.text;
  });

  pi.on("before_agent_start", async (event, ctx) => {
    if (!store.config.enabled) return;
    try {
      const retrieved = await store.retrieve(event.prompt || currentUserPrompt);
      if (retrieved.length === 0) return;
      return { message: { customType: "ollama-memory", content: `Memórias relevantes recuperadas localmente:

${formatRetrieved(retrieved, store.config.maxRetrievedChars)}`, display: false } };
    } catch (error) {
      ctx.ui.notify(`Memória indisponível: ${error instanceof Error ? error.message : String(error)}`, "warning");
    }
  });

  pi.on("context", async (event) => {
    if (!store.config.enabled || !store.config.pruneHistory) return;
    return { messages: pruneHistory(event.messages, store.config) };
  });

  pi.on("message_end", async (event, ctx) => {
    if (!store.config.enabled || event.message.role !== "assistant") return;
    const assistantText = textFromContent(event.message.content);
    const combined = `Usuário:
${currentUserPrompt}

Assistente:
${assistantText}`.trim();
    try {
      await store.add(combined, "conversation", { sessionFile: ctx.sessionManager.getSessionFile() });
    } catch (error) {
      ctx.ui.notify(`Não foi possível indexar a conversa: ${error instanceof Error ? error.message : String(error)}`, "warning");
    }
  });

  pi.on("tool_result", async (event) => {
    if (!store.config.enabled || !store.config.indexToolResults || event.isError) return;
    const text = textFromContent(event.content);
    if (!text.trim()) return;
    try {
      await store.add(`Ferramenta: ${event.toolName}
${text}`, "tool", { toolName: event.toolName, toolCallId: event.toolCallId });
    } catch {
      // Falhas de indexação de tool não interrompem o agente.
    }
  });

  registerMemoryCommand(pi, store);
}
