import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { MemoryStore } from "./memory-store.js";

export function registerMemoryCommand(pi: ExtensionAPI, store: MemoryStore): void {
  pi.registerCommand("memory", {
    description: "Gerencia a memória semântica local via Ollama",
    getArgumentCompletions: (prefix: string) => {
      const commands = ["init", "status", "on", "off", "clear", "update"];
      const matches = commands.filter((command) => command.startsWith(prefix)).map((command) => ({ value: command, label: command }));
      return matches.length > 0 ? matches : null;
    },
    handler: async (args, ctx) => {
      const command = args.trim().toLowerCase();
      if (command !== "init" && !store.isInitialized) {
        ctx.ui.notify("Execute /memory init neste projeto antes de usar o pi-ollama-memory.", "warning");
        return;
      }
      if (command === "init") {
        await store.initialize(ctx.cwd);
        ctx.ui.notify("Configuração criada em .pi/ollama-memory.json.", "info");
      } else if (command === "on") {
        store.config.enabled = true;
        ctx.ui.notify("Memória Ollama ativada para esta sessão.", "info");
      } else if (command === "off") {
        store.config.enabled = false;
        ctx.ui.notify("Memória Ollama desativada para esta sessão.", "info");
      } else if (command === "clear") {
        ctx.ui.notify("Para limpar permanentemente, remova .pi/memory/index.jsonl e reinicie a sessão.", "warning");
      } else if (command === "update") {
        ctx.ui.notify("Atualização automática de embeddings removida. Use /memory clear e reindexe novas interações.", "warning");
      } else {
        ctx.ui.notify(`Memória: ${store.config.enabled ? "on" : "off"}; registros: ${store.records.length}; modelo: ${store.config.embeddingModel}.`, "info");
      }
    },
  });
}
