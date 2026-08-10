import { CONFIG_DIR_NAME } from "@earendil-works/pi-coding-agent";
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import type { Config } from "./config.js";
import { DEFAULT_CONFIG } from "./config.js";

export interface MemoryRecord {
  id: string;
  createdAt: string;
  source: string;
  text: string;
  vector: number[];
  metadata?: Record<string, unknown>;
}

export class MemoryStore {
  config: Config = { ...DEFAULT_CONFIG };
  cwd = "";
  records: MemoryRecord[] = [];
  private writeQueue: Promise<void> = Promise.resolve();

  private get configPath(): string { return join(this.cwd, CONFIG_DIR_NAME, "ollama-memory.json"); }
  private get storePath(): string { return join(this.cwd, CONFIG_DIR_NAME, "memory", "index.jsonl"); }

  async initialize(cwd: string): Promise<void> {
    this.cwd = cwd;
    const configDir = join(cwd, CONFIG_DIR_NAME);
    const memoryDir = join(configDir, "memory");
    await mkdir(memoryDir, { recursive: true });
    if (!existsSync(this.configPath)) {
      await writeFile(this.configPath, `${JSON.stringify(DEFAULT_CONFIG, null, 2)}\n`, "utf8");
    }
    this.config = await this.loadConfig();
    await this.loadRecords();
  }

  private async loadConfig(): Promise<Config> {
    try {
      const parsed = JSON.parse(await readFile(this.configPath, "utf8"));
      return { ...DEFAULT_CONFIG, ...parsed };
    } catch {
      return { ...DEFAULT_CONFIG };
    }
  }

  private async loadRecords(): Promise<void> {
    try {
      const raw = await readFile(this.storePath, "utf8");
      this.records = raw.split(/\r?\n+/).filter(Boolean).map((line) => JSON.parse(line) as MemoryRecord);
    } catch {
      this.records = [];
    }
  }

  private async embed(text: string): Promise<number[]> {
    const response = await fetch(`${this.config.ollamaUrl.replace(/\/$/, "")}/api/embed`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: this.config.embeddingModel, input: text }),
    });
    if (!response.ok) throw new Error(`Ollama respondeu ${response.status}: ${await response.text()}`);
    const payload = await response.json() as { embeddings?: number[][]; embedding?: number[] };
    const vector = payload.embeddings?.[0] ?? payload.embedding;
    if (!Array.isArray(vector) || vector.length === 0) throw new Error("Ollama não retornou um embedding válido");
    return vector;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const size = Math.min(a.length, b.length);
    if (size === 0) return 0;
    let dot = 0, normA = 0, normB = 0;
    for (let index = 0; index < size; index++) {
      dot += a[index] * b[index];
      normA += a[index] * a[index];
      normB += b[index] * b[index];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private enqueueWrite(record: MemoryRecord): Promise<void> {
    const write = async () => {
      await mkdir(join(this.cwd, CONFIG_DIR_NAME, "memory"), { recursive: true });
      await appendFile(this.storePath, `${JSON.stringify(record)}\n`, "utf8");
    };
    this.writeQueue = this.writeQueue.then(write, write);
    return this.writeQueue;
  }

  async add(text: string, source: string, metadata?: Record<string, unknown>): Promise<void> {
    const normalized = text.trim();
    if (!normalized || normalized.length < 20) return;
    const vector = await this.embed(normalized.slice(0, 12000));
    const record: MemoryRecord = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
      createdAt: new Date().toISOString(), source, text: normalized, vector, metadata,
    };
    this.records.push(record);
    await this.enqueueWrite(record);
  }

  async retrieve(query: string): Promise<MemoryRecord[]> {
    if (!query.trim() || this.records.length === 0) return [];
    const queryVector = await this.embed(query.slice(0, 12000));
    return this.records
      .map((record) => ({ record, score: this.cosineSimilarity(queryVector, record.vector) }))
      .filter((item) => item.score >= this.config.minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, this.config.topK)
      .map((item) => item.record);
  }
}
