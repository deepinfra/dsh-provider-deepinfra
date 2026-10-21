/**
 * DSH `GenerateOptions` -> DeepInfra chat-completions body. Pure: images
 * arrive pre-encoded from the caller, so this module needs no DSH runtime.
 */

/**
 * Every `GenerateOptions` field and what happens to it. The tests fail when
 * the installed `@deepseek-ai/dsh-llm` declares a field missing here, so a
 * new field is a decision rather than a silent drop.
 */
export const REQUEST_FIELD_DISPOSITION = Object.freeze({
  provider: 'routing only; never sent',
  model: 'model',
  reasoningEffort: 'reasoning_effort (ids are DeepInfra wire values)',
  messages: 'messages',
  system: 'leading system message, joined with leading system history',
  tools: 'tools (deferLoading declarations are withheld)',
  toolHistory: 'consumed by the runtime projection; never sent',
  temperature: 'temperature',
  maxTokens: 'max_tokens',
  stop: 'stop',
  signal: 'fetch cancellation',
  sessionId: 'prompt_cache_key, so one session reuses its cached prefix',
  purpose: 'not sent',
})

function textOf(content) {
  return content.filter(block => block.type === 'text').map(block => block.text).join('')
}

/**
 * User-side content as a plain string, or as parts when an image is present.
 * @param images - attachmentId -> `{ url, handleText }` for retained images.
 */
function partsOf(content, images) {
  const parts = []
  for (const block of content) {
    if (block.type === 'text') {
      if (block.text.length > 0) parts.push({ type: 'text', text: block.text })
    } else if (block.type === 'image') {
      const image = images.get(block.attachment.attachmentId)
      if (image === undefined) throw new Error(`image ${block.attachment.attachmentId} was not prepared for the request`)
      parts.push({ type: 'text', text: image.handleText })
      parts.push({ type: 'image_url', image_url: { url: image.url } })
    }
  }
  return parts.every(part => part.type === 'text') ? parts.map(part => part.text).join('') : parts
}

function assistantOf(message) {
  const text = textOf(message.content)
  const reasoning = message.content.filter(block => block.type === 'reasoning').map(block => block.text).join('')
  const calls = message.content.filter(block => block.type === 'tool-call')
  if (text.length === 0 && calls.length === 0) return undefined
  return {
    role: 'assistant',
    content: text.length > 0 ? text : null,
    // Thinking models continuing a tool loop need their own reasoning back.
    ...reasoning.length > 0 ? { reasoning_content: reasoning } : {},
    ...calls.length > 0
      ? {
          tool_calls: calls.map(call => ({
            id: call.id,
            type: 'function',
            function: { name: call.name, arguments: call.arguments || '{}' },
          })),
        }
      : {},
  }
}

/**
 * Convert request history to chat-completions messages.
 * @param options - the projected request.
 * @param images - prepared retained images keyed by attachment id.
 */
export function toChatMessages(options, images = new Map()) {
  const out = []
  const lead = [options.system]
  let start = 0
  while (options.messages[start]?.role === 'system') lead.push(textOf(options.messages[start++].content))
  const system = lead.filter(Boolean).join('\n\n')
  if (system.length > 0) out.push({ role: 'system', content: system })

  // A tool message cannot carry images; they follow the tool batch as user content.
  let toolImages = []
  const flushToolImages = () => {
    if (toolImages.length > 0) out.push({ role: 'user', content: toolImages })
    toolImages = []
  }

  for (const message of options.messages.slice(start)) {
    if (message.role !== 'tool') flushToolImages()
    switch (message.role) {
      case 'system':
      case 'developer': {
        const text = textOf(message.content)
        if (text.length > 0) out.push({ role: 'system', content: text })
        break
      }
      case 'user': {
        const content = partsOf(message.content, images)
        if (content.length > 0) out.push({ role: 'user', content })
        break
      }
      case 'assistant': {
        const assistant = assistantOf(message)
        if (assistant !== undefined) out.push(assistant)
        break
      }
      case 'tool': {
        const content = partsOf(message.content, images)
        const text = typeof content === 'string'
          ? content
          : content.filter(part => part.type === 'text').map(part => part.text).join('\n')
        if (typeof content !== 'string') toolImages.push(...content.filter(part => part.type === 'image_url'))
        out.push({ role: 'tool', tool_call_id: message.toolCallId, content: text || '(no output)' })
        break
      }
      default:
        break
    }
  }
  flushToolImages()
  return out
}

/** The complete streaming request body. */
export function buildRequestBody(options, images) {
  const tools = (options.tools ?? []).filter(tool => tool.deferLoading !== true)
  return {
    model: options.model,
    messages: toChatMessages(options, images),
    stream: true,
    stream_options: { include_usage: true },
    ...tools.length > 0
      ? {
          tools: tools.map(tool => ({
            type: 'function',
            function: { name: tool.name, description: tool.description, parameters: tool.parameters },
          })),
        }
      : {},
    ...options.reasoningEffort === undefined ? {} : { reasoning_effort: options.reasoningEffort },
    ...typeof options.temperature === 'number' ? { temperature: options.temperature } : {},
    ...options.maxTokens === undefined ? {} : { max_tokens: options.maxTokens },
    ...options.stop?.length ? { stop: options.stop } : {},
    ...options.sessionId === undefined ? {} : { prompt_cache_key: String(options.sessionId) },
  }
}
