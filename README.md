# pi-ollama-memory

Memória semântica por projeto para o Pi Coding Agent usando Ollama. O pacote
indexa prompts, respostas e resultados de ferramentas localmente, recupera os
trechos relevantes antes de cada chamada e evita reenviar histórico irrelevante.

> **Primeira utilização:** execute `/memory init` dentro do projeto. O comando cria automaticamente `.pi/ollama-memory.json` e os diretórios necessários; não é necessário criar arquivos ou diretórios manualmente.

## Requisitos

- Ollama em `http://localhost:11434`
- `nomic-embed-text` (ou outro modelo de embeddings)
- Pi Coding Agent >= 0.84

## Instalação

```bash
pi install ./pi-ollama-memory
# ou, após publicação:
pi install npm:pi-ollama-memory
```

## Configuração

O comando `/memory init` cria `.pi/ollama-memory.json` com os valores padrão. Edite o arquivo somente se quiser personalizar a configuração:

```json
{
  "enabled": true,
  "ollamaUrl": "http://localhost:11434",
  "embeddingModel": "nomic-embed-text:latest",
  "topK": 6,
  "maxRetrievedChars": 12000,
  "minScore": 0.25,
  "indexToolResults": true,
  "pruneHistory": false,
  "keepRecentMessages": 8
}
```

Os dados ficam em `.pi/ollama-memory/index.json` e não são enviados para a
nuvem. Adicione esse diretório ao `.gitignore` se o projeto for versionado.

## Comandos

- `/memory init` — inicializa a configuração e os diretórios do projeto
- `/memory status`
- `/memory search <texto>`
- `/memory on` e `/memory off`
- `/memory clear`
- `/memory update` — reprocessa as embeddings com o modelo configurado

A memória é uma camada de recuperação: o envelope da API continua sendo JSON.
Por padrão, mensagens antigas são removidas do contexto temporário e substituídas
por memórias semanticamente relevantes; a sessão original permanece intacta.
TOON pode ser aplicado por outra extensão aos dados densos recuperados.
