/**
 * DeepInfra provider for DeepSeek Harness: registers the `deepinfra` route
 * on `ctx.llm`, advertises DeepInfra's live chat-model listing, and streams
 * over its OpenAI-compatible chat-completions endpoint.
 */

import { requestImageDimensions } from '@deepseek-ai/dsh-attachment'
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment'
import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  IMAGE_OFFLOAD_REQUIRED_CODE,
  INVALID_CREDENTIAL_CODE,
  LlmAdapter,
  LlmError,
  QUOTA_EXCEEDED_CODE,
  assertUsableApiKey,
  attributionHeaders,
  contentHasImage,
  isContextWindowExceededError,
  isQuotaExceededError,
  offloadedImageText,
  projectOffloadedImages,
  requestImageHandleText,
  requiredImageOffload,
  resolveImageAttachmentAccess,
} from '@deepseek-ai/dsh-llm'
import z from '@deepseek-ai/schemastery'

import { Catalog, DEFAULT_BASE_URL, modelInfo, resolvedModelInfo } from './catalog.js'
import { buildRequestBody } from './request.js'
import { sseData, toStreamChunks } from './stream.js'

export { Catalog, DEFAULT_BASE_URL, FALLBACK_MODELS, REASONING_EFFORTS, REASONING_TOGGLE, chatModels } from './catalog.js'
export { REQUEST_FIELD_DISPOSITION, buildRequestBody, toChatMessages } from './request.js'
export { mapUsage, sseData, toStreamChunks } from './stream.js'

export const name = 'provider-deepinfra'
export const inject = ['llm']

export const PROVIDER = 'deepinfra'
const DISPLAY_NAME = 'DeepInfra'
const DEFAULT_API_KEY_ENV = 'DEEPINFRA_API_KEY'
const REQUEST_IMAGE_PIXEL_BUDGET = 4_194_304
const REQUEST_IMAGE_MAX_BYTES = 1_048_576
const MAX_REQUEST_IMAGE_BYTES = 20 * 1_048_576

export const Config = z.object({
  baseURL: z.string().default(DEFAULT_BASE_URL).description('DeepInfra OpenAI-compatible API root.').volatile(),
  apiKeyEnv: z.string().role('credential-ref').default(DEFAULT_API_KEY_ENV)
    .description('Credential reference holding the DeepInfra API token.').volatile(),
})

function stripSlash(url) {
  return String(url || DEFAULT_BASE_URL).replace(/\/+$/, '')
}

function retryAfterMs(header) {
  if (header === null) return undefined
  const seconds = Number(header)
  if (Number.isFinite(seconds) && seconds > 0) return seconds * 1000
  const at = Date.parse(header)
  return Number.isFinite(at) && at > Date.now() ? at - Date.now() : undefined
}

/** Classify a non-2xx DeepInfra reply. */
export function httpFailureCode(status, detail) {
  if (status === 401) return INVALID_CREDENTIAL_CODE
  if (status === 402 || isQuotaExceededError(detail)) return QUOTA_EXCEEDED_CODE
  if (status === 403) return 'AUTH'
  if (status === 404) return 'UNKNOWN_MODEL'
  if (status === 429) return 'RATE_LIMIT'
  if ((status === 400 || status === 413 || status === 422) && isContextWindowExceededError(detail)) {
    return CONTEXT_WINDOW_EXCEEDED_CODE
  }
  if (status >= 500) return 'SERVER'
  return 'INVALID_REQUEST'
}

async function httpError(res, model) {
  const raw = await res.text().catch(() => '')
  let detail = raw
  try {
    const body = JSON.parse(raw)
    detail = body?.error?.message ?? body?.detail?.error ?? (typeof body?.detail === 'string' ? body.detail : raw)
  } catch {
    // Not JSON: keep the raw text.
  }
  const requestId = res.headers.get('x-request-id') ?? undefined
  const retryAfter = retryAfterMs(res.headers.get('retry-after'))
  return new LlmError(
    `DeepInfra ${res.status} for model "${model}": ${String(detail || res.statusText).slice(0, 2000)}`,
    httpFailureCode(res.status, String(detail)),
    {
      status: res.status,
      ...requestId ? { requestId } : {},
      ...retryAfter === undefined ? {} : { providerRetryAfterMs: retryAfter },
    },
  )
}

export class DeepInfraAdapter extends LlmAdapter {
  #options
  #resolveApiKey
  #fetch
  #attachments
  #imageAccess
  #catalogs = new Map()

  /**
   * @param deps.options - `() => { baseURL, apiKeyEnv }`, read per operation.
   * @param deps.resolveApiKey - `(ref) => Promise<string>`; throws `MISSING_CREDENTIAL`.
   * @param deps.fetch - fetch implementation; defaults to the global.
   * @param deps.attachments - `() => AttachmentStore | undefined`.
   * @param deps.imageAccess - `(store, ref) => ImageAttachmentAccess | undefined`.
   */
  constructor({ options, resolveApiKey, fetch = globalThis.fetch, attachments = () => undefined, imageAccess = () => undefined }) {
    super()
    this.#options = options
    this.#resolveApiKey = resolveApiKey
    this.#fetch = fetch
    this.#attachments = attachments
    this.#imageAccess = imageAccess
  }

  #baseURL() {
    return stripSlash(this.#options().baseURL)
  }

  #catalog() {
    const baseURL = this.#baseURL()
    let catalog = this.#catalogs.get(baseURL)
    if (catalog === undefined) {
      catalog = new Catalog(async () => {
        const res = await this.#fetch(`${baseURL}/models`, { headers: attributionHeaders() })
        if (!res.ok) throw new Error(`GET /models returned ${res.status}`)
        return res.json()
      })
      this.#catalogs.set(baseURL, catalog)
    }
    return catalog
  }

  providerInfo(provider) {
    return { id: provider, name: DISPLAY_NAME }
  }

  imageRequestTarget(_provider, _model, image) {
    return { ...requestImageDimensions(image.width, image.height, REQUEST_IMAGE_PIXEL_BUDGET), maxBytes: REQUEST_IMAGE_MAX_BYTES }
  }

  async listModels(provider) {
    // Without a key every request would fail, so the picker offers nothing.
    try {
      await this.#resolveApiKey(this.#options().apiKeyEnv)
    } catch (error) {
      if (error instanceof LlmError && error.code === 'MISSING_CREDENTIAL') return []
      throw error
    }
    return (await this.#catalog().models()).map(model => modelInfo(provider, model))
  }

  async resolveModel(provider, model, signal) {
    return resolvedModelInfo(provider, model, await this.#catalog().find(model, signal))
  }

  async #prepareImages(options) {
    const images = new Map()
    if (!options.messages.some(message => contentHasImage(message.content))) return { messages: options.messages, images }
    const store = this.#attachments()
    if (store === undefined) throw new LlmError('DeepInfra image input requires the attachment service', 'UNSUPPORTED_CONTENT')

    const refs = new Map()
    for (const message of options.messages) {
      for (const block of message.content) {
        if (block.type === 'image' && block.offloaded !== true) refs.set(block.attachment.attachmentId, block.attachment)
      }
    }
    const versions = new Map()
    await Promise.all([...refs.values()].map(async (ref) => {
      versions.set(ref.attachmentId, await store.readImageRequest(ref, this.imageRequestTarget(PROVIDER, options.model, ref), options.signal))
    }))
    const offloadImages = requiredImageOffload(
      options.messages,
      { representation: 'base64', maxBytes: MAX_REQUEST_IMAGE_BYTES },
      block => versions.get(block.attachment.attachmentId).bytes,
    )
    if (offloadImages > 0) {
      throw new LlmError(
        `DeepInfra request images exceed ${MAX_REQUEST_IMAGE_BYTES} base64 bytes; ${offloadImages} more oldest occurrence(s) must be offloaded.`,
        IMAGE_OFFLOAD_REQUIRED_CODE,
        { offloadImages },
      )
    }
    for (const [id, version] of versions) {
      const ref = refs.get(id)
      images.set(id, {
        url: `data:${version.mediaType};base64,${Buffer.from(version.data).toString('base64')}`,
        handleText: requestImageHandleText(ref, version, this.#imageAccess(store, ref)),
      })
    }
    const messages = projectOffloadedImages(options.messages, ref => offloadedImageText(ref, this.#imageAccess(store, ref)))
    return { messages, images }
  }

  async *stream(options) {
    const { apiKeyEnv } = this.#options()
    const baseURL = this.#baseURL()
    const apiKey = await this.#resolveApiKey(apiKeyEnv)
    const { messages, images } = await this.#prepareImages(options)
    const body = JSON.stringify(buildRequestBody({ ...options, messages }, images))

    let res
    try {
      res = await this.#fetch(`${baseURL}/chat/completions`, {
        method: 'POST',
        headers: {
          ...attributionHeaders(),
          'content-type': 'application/json',
          accept: 'text/event-stream',
          authorization: `Bearer ${apiKey}`,
        },
        body,
        signal: options.signal,
      })
    } catch (cause) {
      if (options.signal?.aborted) throw new LlmError('DeepInfra request aborted by caller', 'ABORTED', { cause })
      throw new LlmError(`DeepInfra request failed: ${cause?.message ?? cause}`, 'TRANSPORT', { cause })
    }
    if (!res.ok) throw await httpError(res, options.model)
    if (res.body === null) throw new LlmError('DeepInfra returned an empty body', 'EMPTY_RESPONSE')

    try {
      yield* toStreamChunks(sseData(res.body), options.model)
    } catch (cause) {
      if (options.signal?.aborted) throw new LlmError('DeepInfra request aborted by caller', 'ABORTED', { cause })
      throw new LlmError(`DeepInfra stream failed: ${cause?.message ?? cause}`, 'TRANSPORT', { cause })
    }
  }
}

export function apply(ctx, config) {
  const options = () => ({ baseURL: config.baseURL.get(), apiKeyEnv: config.apiKeyEnv.get() })

  const resolveApiKey = async (ref) => {
    const credentials = ctx.get('credentials')
    const value = credentials !== undefined
      ? (await credentials.resolve(ref))?.value
      : launchEnvironmentOf(ctx).get(ref)?.value
    if (value !== undefined && value.length > 0) return assertUsableApiKey(value, name, ref)
    throw new LlmError(
      `${name}: no API key for provider route "${PROVIDER}"; store ${ref} through the credentials service`
      + ` (the web Models page writes it), or export ${ref} in the launching environment`,
      'MISSING_CREDENTIAL',
    )
  }

  const adapter = new DeepInfraAdapter({
    options,
    resolveApiKey,
    attachments: () => ctx.get('attachments'),
    imageAccess: (store, ref) => resolveImageAttachmentAccess(
      store,
      hostPath => ctx.get('fs')?.processPathFromHostPath(hostPath),
      ref,
    ),
  })

  ctx.llm.registerConfigurableProviders([{
    provider: PROVIDER,
    displayName: DISPLAY_NAME,
    settingsNs: ctx.fiber?.entry?.options.id ?? name,
    settingsPath: [],
    declared: false,
    defaultBaseURL: DEFAULT_BASE_URL,
  }])
  ctx.llm.registerAdapter([PROVIDER], adapter)
}
