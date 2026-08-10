export interface Config {
  enabled: boolean;
  ollamaUrl: string;
  embeddingModel: string;
  topK: number;
  maxRetrievedChars: number;
  minScore: number;
  indexToolResults: boolean;
  pruneHistory: boolean;
  keepRecentMessages: number;
}

export const DEFAULT_CONFIG: Config = {
  enabled: true,
  ollamaUrl: "http://localhost:11434",
  embeddingModel: "nomic-embed-text:latest",
  topK: 6,
  maxRetrievedChars: 12000,
  minScore: 0.25,
  indexToolResults: true,
  pruneHistory: false,
  keepRecentMessages: 8,
};
