/**
 * DeepInfra model catalog: `GET {baseURL}/models` mapped to DSH model
 * metadata. Pure apart from the injected fetch, so it is testable without
 * the DSH runtime.
 */

export const DEFAULT_BASE_URL = 'https://api.deepinfra.com/v1/openai'

/** Wire values of DeepInfra's `reasoning_effort`; the id is sent verbatim. */
export const REASONING_EFFORTS = Object.freeze([
  { id: 'none', name: 'Off' },
  { id: 'low', name: 'Low' },
  { id: 'medium', name: 'Medium' },
  { id: 'high', name: 'High' },
])

/** Models tagged `reasoning` without graded effort only switch thinking on or off. */
export const REASONING_TOGGLE = Object.freeze([
  { id: 'none', name: 'Off' },
  { id: 'high', name: 'On' },
])

function reasoningOf(tags) {
  if (tags.includes('reasoning_effort')) return REASONING_EFFORTS
  if (tags.includes('reasoning')) return REASONING_TOGGLE
  return undefined
}

/**
 * Served while the live listing is unreachable, so the route never comes up
 * empty. Shaped like `/models` entries; the live listing replaces it whole.
 */
export const FALLBACK_MODELS = Object.freeze([
  entry('deepseek-ai/DeepSeek-V4-Pro', 1048576, ['chat', 'prompt_cache']),
  entry('deepseek-ai/DeepSeek-V4-Flash', 1048576, ['chat', 'prompt_cache', 'reasoning']),
  entry('moonshotai/Kimi-K3', 1048576, ['chat', 'vision', 'prompt_cache', 'reasoning']),
  entry('zai-org/GLM-5.3', 1048576, ['chat', 'prompt_cache', 'reasoning']),
  entry('Qwen/Qwen3.6-35B-A3B', 262144, ['chat', 'vision', 'prompt_cache', 'reasoning_effort', 'reasoning']),
])

function entry(id, contextLength, tags) {
  return Object.freeze({ id, metadata: Object.freeze({ context_length: contextLength, tags: Object.freeze(tags) }) })
}

function tagsOf(model) {
  const tags = model?.metadata?.tags
  return Array.isArray(tags) ? tags : []
}

function positiveInt(value) {
  return Number.isSafeInteger(value) && value > 0 ? value : undefined
}

/**
 * Chat-capable entries of one `/models` reply body, in listing order.
 * @param body - parsed JSON reply.
 * @returns entries tagged `chat`; anything else (embeddings, image, ASR) is dropped.
 */
export function chatModels(body) {
  const data = Array.isArray(body?.data) ? body.data : []
  return data.filter(model => typeof model?.id === 'string' && model.id.length > 0 && tagsOf(model).includes('chat'))
}

/** Advisory catalog entry for selectors. */
export function modelInfo(provider, model) {
  return {
    provider,
    id: model.id,
    name: model.id,
    inputModalities: tagsOf(model).includes('vision') ? ['text', 'image'] : ['text'],
  }
}

/**
 * Exact-model metadata. Unknown ids resolve to identity only: DSH treats
 * absent context and reasoning as unknown, which beats a guessed number.
 */
export function resolvedModelInfo(provider, id, model) {
  if (model === undefined) return { provider, id, name: id }
  const contextWindow = positiveInt(model.metadata?.context_length)
  const efforts = reasoningOf(tagsOf(model))
  return {
    ...modelInfo(provider, model),
    ...contextWindow === undefined ? {} : { context: { contextWindow } },
    ...efforts === undefined ? {} : { reasoning: { efforts: efforts.map(effort => ({ ...effort })) } },
  }
}

/**
 * Time-bounded cache over the live listing. Concurrent callers share one
 * request; a failed refresh keeps the last good listing, or the fallback
 * when there never was one.
 */
export class Catalog {
  #fetchModels
  #ttlMs
  #now
  #cached
  #fetchedAt = -Infinity
  #inflight

  /**
   * @param fetchModels - `(signal) => Promise<body>` for `GET /models`.
   * @param options - `ttlMs` cache lifetime, `now` clock for tests.
   */
  constructor(fetchModels, { ttlMs = 10 * 60_000, now = Date.now } = {}) {
    this.#fetchModels = fetchModels
    this.#ttlMs = ttlMs
    this.#now = now
  }

  async models(signal) {
    if (this.#cached !== undefined && this.#now() - this.#fetchedAt < this.#ttlMs) return this.#cached
    this.#inflight ??= this.#refresh().finally(() => { this.#inflight = undefined })
    return signal === undefined ? this.#inflight : abortable(this.#inflight, signal)
  }

  async find(id, signal) {
    return (await this.models(signal)).find(model => model.id === id)
  }

  async #refresh() {
    try {
      const models = chatModels(await this.#fetchModels())
      if (models.length > 0) {
        this.#cached = models
        this.#fetchedAt = this.#now()
        return models
      }
    } catch {
      // Discovery must never fail a request; the fallback below covers it.
    }
    return this.#cached ?? FALLBACK_MODELS
  }
}

function abortable(promise, signal) {
  signal.throwIfAborted()
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(signal.reason)
    signal.addEventListener('abort', onAbort, { once: true })
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort))
  })
}
