# DeepInfra provider for DeepSeek Harness

Use [DeepInfra](https://deepinfra.com) models from [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`).
The plugin registers a `deepinfra` provider route and streams over DeepInfra's OpenAI-compatible chat-completions API.

- **Live model catalog.** Models come from `GET https://api.deepinfra.com/v1/openai/models`, so new DeepInfra models appear without a plugin update. Only chat models are listed.
- **Real model facts.** Each model reports its own context window, image input (`vision`), and reasoning-effort levels. Compaction timing and the model picker use these values, so nothing has to be typed into settings.
- **Tool calls, reasoning, and images.** Parallel tool calls, `reasoning_content` (also sent back on later turns so thinking models can continue a tool loop), and image input on vision models.
- **Prompt caching.** The DSH session id goes out as `prompt_cache_key`, so repeated turns reuse their cached prefix.
- **No conflict with your own routes.** It adds a route beside `llm-pi-ai` and does not modify it.

## Install

Needs Node ≥ 22.15 and a DeepInfra API token (https://deepinfra.com/dash/api_keys).

```sh
dsh plugin --profile web add github:deepinfra/dsh-provider-deepinfra
export DEEPINFRA_API_KEY=...   # or store it from the web Models page
dsh web
```

Pick a DeepInfra model in the model picker, or make one the default:

```yaml
# $DSH_HOME/profiles/<profile>/cordis.patch.yml
- id: agent-default-model
  config:
    provider: deepinfra
    model: deepseek-ai/DeepSeek-V4-Flash
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `baseURL` | `https://api.deepinfra.com/v1/openai` | OpenAI-compatible API root |
| `apiKeyEnv` | `DEEPINFRA_API_KEY` | Credential reference resolved for each request through the DSH credential service, or from the environment when the composition has none |

```yaml
- id: provider-deepinfra
  config:
    apiKeyEnv: MY_DEEPINFRA_TOKEN
```

## Behavior

- The catalog is cached for 10 minutes. If the listing is unreachable, the last good listing is served, or a small built-in list if there was never one, so the route never comes up empty. A model id missing from the listing (for example, a private deployment) can still be requested; its context window is reported as unknown rather than guessed.
- Without an API key the picker lists no DeepInfra models, because every request would fail.
- Reasoning: models DeepInfra tags `reasoning_effort` offer `Off`, `Low`, `Medium`, `High`; models tagged only `reasoning` offer `Off`/`On` (`none`/`high` on the wire). The choice goes out as `reasoning_effort`, and leaving it unset keeps the model's own default. Some models, such as DeepSeek-V4.1-Flash, do not think unless you turn it on.
- HTTP failures map to DSH failure codes: `401` → `INVALID_CREDENTIAL`, `402` or quota text → `QUOTA`, `429` → `RATE_LIMIT` (with `Retry-After`), context-overflow `400` → `CONTEXT_WINDOW_EXCEEDED`, `5xx` → `SERVER`. `dsh-llm-retry` retries the transient ones.
- Images go inline as base64 data URLs, scaled to ~4 MP and 1 MiB each. When one request's images exceed 20 MiB, DSH is asked to offload the oldest ones.

## Development

```sh
npm install
npm test
```

The tests include a contract check: every `GenerateOptions` field declared by the installed `@deepseek-ai/dsh-llm` must have an entry in `REQUEST_FIELD_DISPOSITION` (`lib/request.js`). When DSH adds a request field, the test fails until the field is handled or deliberately ignored.

## License

MIT
