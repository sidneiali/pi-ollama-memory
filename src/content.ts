export function textFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((block) => {
    if (!block || typeof block !== "object") return "";
    const value = block as { type?: string; text?: string; name?: string; arguments?: unknown };
    if (typeof value.text === "string") return value.text;
    if (value.type === "toolCall") return `${value.name ?? "tool"} ${JSON.stringify(value.arguments ?? {})}`;
    return "";
  }).filter(Boolean).join("\n");
}
