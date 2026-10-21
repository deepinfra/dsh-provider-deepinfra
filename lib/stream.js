/**
 * DeepInfra SSE -> DSH `StreamChunk`s. Pure: failures end the stream as
 * plain `finish {kind:'error'}` chunks, which is one of the two error paths
 * the adapter contract sanctions.
 */

/**
 * Split an SSE byte stream into `data:` payloads.
 * @param body - a fetch `ReadableStream<Uint8Array>` or any async iterable of bytes.
 */
export async function* sseData(body) {
  const decoder = new TextDecoder()
  let buffer = ''
  for await (const bytes of body) {
    buffer += decoder.decode(bytes, { stream: true })
    let newline
    while ((newline = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, newline).replace(/\r$/, '')
      buffer = buffer.slice(newline + 1)
      if (line.startsWith('data:')) yield line.slice(5).trimStart()
    }
  }
  buffer += decoder.decode()
  if (buffer.startsWith('data:')) yield buffer.slice(5).trimStart()
}

/** OpenAI usage -> DSH's disjoint counts (cached input reported apart). */
export function mapUsage(usage) {
  const prompt = usage.prompt_tokens ?? 0
  const cached = usage.prompt_tokens_details?.cached_tokens ?? 0
  const reasoning = usage.completion_tokens_details?.reasoning_tokens
  return {
    inputTokens: Math.max(prompt - cached, 0),
    outputTokens: usage.completion_tokens ?? 0,
    ...Number.isSafeInteger(usage.total_tokens) ? { totalTokens: usage.total_tokens } : {},
    ...cached > 0 ? { cacheReadTokens: cached } : {},
    ...reasoning > 0 ? { reasoningTokens: reasoning } : {},
  }
}

function failure(message, code) {
  return { type: 'finish', reason: { kind: 'error', failure: { message, code } } }
}

/**
 * Translate chat-completion stream payloads into DSH chunks.
 * @param payloads - `data:` payload strings, `[DONE]` included.
 * @param model - requested model id, for diagnostics.
 */
export async function* toStreamChunks(payloads, model) {
  let nextIndex = 0
  let open
  const tools = new Map()
  let finishReason
  let usage
  let emitted = 0

  function* closeOpen() {
    if (open === undefined) return
    yield { type: 'block-end', index: open.index, block: { type: open.type, text: open.text } }
    open = undefined
  }

  function* appendText(type, text) {
    if (open?.type !== type) {
      yield* closeOpen()
      open = { type, index: nextIndex++, text: '' }
      emitted++
      yield { type: 'block-start', index: open.index, blockType: type }
    }
    open.text += text
    yield { type: type === 'text' ? 'text-delta' : 'reasoning-delta', index: open.index, text }
  }

  for await (const payload of payloads) {
    if (payload === '[DONE]') break
    let chunk
    try {
      chunk = JSON.parse(payload)
    } catch {
      yield failure(`DeepInfra sent an unparseable stream event for model "${model}"`, 'INVALID_RESPONSE')
      return
    }
    if (chunk.error !== undefined) {
      yield failure(String(chunk.error?.message ?? JSON.stringify(chunk.error)), 'PROVIDER_ERROR')
      return
    }
    if (chunk.usage) usage = chunk.usage
    const choice = chunk.choices?.[0]
    if (choice === undefined) continue
    const delta = choice.delta ?? {}
    const reasoning = delta.reasoning_content ?? delta.reasoning
    if (typeof reasoning === 'string' && reasoning.length > 0) yield* appendText('reasoning', reasoning)
    if (typeof delta.content === 'string' && delta.content.length > 0) yield* appendText('text', delta.content)
    for (const call of delta.tool_calls ?? []) {
      const key = call.index ?? 0
      let tool = tools.get(key)
      if (tool === undefined) {
        yield* closeOpen()
        tool = { index: nextIndex++, id: call.id || `call_${key}`, name: call.function?.name ?? '', arguments: '' }
        tools.set(key, tool)
        emitted++
        yield { type: 'block-start', index: tool.index, blockType: 'tool-call' }
      } else if (call.function?.name) {
        tool.name += call.function.name
      }
      const argumentsDelta = call.function?.arguments ?? ''
      tool.arguments += argumentsDelta
      yield {
        type: 'tool-call-delta',
        index: tool.index,
        id: tool.id,
        ...tool.name.length > 0 ? { name: tool.name } : {},
        argumentsDelta,
      }
    }
    if (choice.finish_reason) finishReason = choice.finish_reason
  }

  yield* closeOpen()
  for (const tool of tools.values()) {
    yield {
      type: 'block-end',
      index: tool.index,
      block: { type: 'tool-call', id: tool.id, name: tool.name, arguments: tool.arguments || '{}' },
    }
  }
  if (usage !== undefined) yield { type: 'usage', usage: mapUsage(usage) }

  switch (finishReason) {
    case 'stop':
    case 'tool_calls':
      if (emitted === 0) {
        yield failure(`model "${model}" returned a completed response with no content`, 'EMPTY_RESPONSE')
      } else {
        yield { type: 'finish', reason: { kind: tools.size > 0 ? 'tool-calls' : 'stop' } }
      }
      return
    case 'length':
      yield { type: 'finish', reason: { kind: 'max-tokens' } }
      return
    case 'content_filter':
      yield failure(`DeepInfra filtered the response of model "${model}"`, 'CONTENT_FILTER')
      return
    case undefined:
      yield failure(`DeepInfra stream for model "${model}" ended without a finish_reason`, 'TRANSPORT')
      return
    default:
      yield failure(`DeepInfra stream for model "${model}" finished with "${finishReason}"`, 'PROVIDER_ERROR')
  }
}
