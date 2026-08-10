import type { Config } from "./config.js";

function toolCallIds(message: any): string[] {
  const ids = new Set<string>();
  const add = (value: unknown) => {
    if (typeof value === "string" && value) ids.add(value);
  };
  const inspect = (value: unknown, blockType?: string): void => {
    if (!value || typeof value !== "object") return;
    if (Array.isArray(value)) {
      for (const item of value) inspect(item);
      return;
    }
    const object = value as Record<string, unknown>;
    if (blockType === "toolCall" || object.type === "toolCall") add(object.id);
    add(object.callId);
    add(object.toolCallId);
    add(object.tool_call_id);
    if (typeof object.type === "string") inspect(object.content, object.type);
    else inspect(object.content);
  };
  inspect(message);
  return [...ids];
}

function isToolResult(message: any): boolean {
  return message?.role === "toolResult" || message?.role === "tool_result";
}

function isAssistantToolCall(message: any): boolean {
  return message?.role === "assistant" && toolCallIds(message).length > 0;
}

export function pruneHistory(messages: any[], config: Config): any[] {
  const conversationRoles = new Set(["user", "assistant", "toolResult", "tool_result"]);
  const conversationIndexes = messages
    .map((message, index) => ({ message, index }))
    .filter(({ message }) => conversationRoles.has(message?.role))
    .map(({ index }) => index);
  const keep = new Set<number>();
  if (!config.pruneHistory) {
    for (const index of conversationIndexes) keep.add(index);
  } else {
    const cutoff = Math.max(0, conversationIndexes.length - config.keepRecentMessages);
    for (const index of conversationIndexes.slice(cutoff)) keep.add(index);
  }

  const indexesById = new Map<string, number[]>();
  messages.forEach((message, index) => {
    for (const id of toolCallIds(message)) {
      const indexes = indexesById.get(id) ?? [];
      indexes.push(index);
      indexesById.set(id, indexes);
    }
  });
  let changed = true;
  while (changed) {
    changed = false;
    for (const index of [...keep]) {
      for (const id of toolCallIds(messages[index])) {
        for (const relatedIndex of indexesById.get(id) ?? []) {
          if (!keep.has(relatedIndex)) {
            keep.add(relatedIndex);
            changed = true;
          }
        }
      }
    }
  }

  const keptCallIds = new Set<string>();
  for (const index of keep) {
    if (isAssistantToolCall(messages[index])) {
      for (const id of toolCallIds(messages[index])) keptCallIds.add(id);
    }
  }
  return messages.filter((message, index) => {
    if (!keep.has(index) && conversationRoles.has(message?.role)) return false;
    if (isToolResult(message)) {
      const ids = toolCallIds(message);
      return ids.length > 0 && ids.some((id) => keptCallIds.has(id));
    }
    if (isAssistantToolCall(message)) {
      return toolCallIds(message).every((id) => messages.some((candidate) =>
        isToolResult(candidate) && toolCallIds(candidate).includes(id)));
    }
    return true;
  });
}
