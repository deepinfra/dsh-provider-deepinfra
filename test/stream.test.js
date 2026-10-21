import assert from 'node:assert/strict'
import { test } from 'node:test'

import { mapUsage, sseData, toStreamChunks } from '../lib/stream.js'

async function collect(iterable) {
  const out = []
  for await (const item of iterable) out.push(item)
  return out
}

async function* payloads(...events) {
  for (const event of events) yield typeof event === 'string' ? event : JSON.stringify(event)
}

const delta = (value, finish_reason = null) => ({ choices: [{ index: 0, delta: value, finish_reason }] })

test('sseData splits data lines across arbitrary byte boundaries', async () => {
  const wire = 'data: {"a":1}\r\n\r\n: keep-alive\n\ndata: [DONE]\n\n'
  const bytes = new TextEncoder().encode(wire)
  async function* chunks() {
    for (let i = 0; i < bytes.length; i += 3) yield bytes.slice(i, i + 3)
  }
  assert.deepEqual(await collect(sseData(chunks())), ['{"a":1}', '[DONE]'])
})

test('reasoning then text become two closed blocks with usage and stop', async () => {
  const chunks = await collect(toStreamChunks(payloads(
    delta({ role: 'assistant', reasoning_content: 'thin' }),
    delta({ reasoning_content: 'king' }),
    delta({ content: 'Hel' }),
    delta({ content: 'lo' }, 'stop'),
    { choices: [], usage: { prompt_tokens: 100, completion_tokens: 7, total_tokens: 107, prompt_tokens_details: { cached_tokens: 60 } } },
    '[DONE]',
  ), 'm'))
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'thin' },
    { type: 'reasoning-delta', index: 0, text: 'king' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'thinking' } },
    { type: 'block-start', index: 1, blockType: 'text' },
    { type: 'text-delta', index: 1, text: 'Hel' },
    { type: 'text-delta', index: 1, text: 'lo' },
    { type: 'block-end', index: 1, block: { type: 'text', text: 'Hello' } },
    { type: 'usage', usage: { inputTokens: 40, outputTokens: 7, totalTokens: 107, cacheReadTokens: 60 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('parallel tool calls stream as separate blocks and finish as tool-calls', async () => {
  const chunks = await collect(toStreamChunks(payloads(
    delta({ content: 'Let me check.' }),
    delta({ tool_calls: [{ index: 0, id: 'call_a', type: 'function', function: { name: 'read', arguments: '' } }] }),
    delta({ tool_calls: [{ index: 0, function: { arguments: '{"path":' } }] }),
    delta({ tool_calls: [{ index: 1, id: 'call_b', type: 'function', function: { name: 'ls', arguments: '{}' } }] }),
    delta({ tool_calls: [{ index: 0, function: { arguments: '"a"}' } }] }, 'tool_calls'),
    '[DONE]',
  ), 'm'))
  const ends = chunks.filter(chunk => chunk.type === 'block-end').map(chunk => chunk.block)
  assert.deepEqual(ends, [
    { type: 'text', text: 'Let me check.' },
    { type: 'tool-call', id: 'call_a', name: 'read', arguments: '{"path":"a"}' },
    { type: 'tool-call', id: 'call_b', name: 'ls', arguments: '{}' },
  ])
  assert.deepEqual(chunks.at(-1), { type: 'finish', reason: { kind: 'tool-calls' } })
  assert.ok(chunks.filter(chunk => chunk.type === 'tool-call-delta').every(chunk => chunk.id.startsWith('call_')))
})

test('finish reasons and degenerate streams map to the harness vocabulary', async () => {
  const finishOf = async (...events) => (await collect(toStreamChunks(payloads(...events), 'm'))).at(-1).reason

  assert.deepEqual(await finishOf(delta({ content: 'x' }, 'length')), { kind: 'max-tokens' })
  assert.equal((await finishOf(delta({}, 'stop'))).failure.code, 'EMPTY_RESPONSE')
  assert.equal((await finishOf(delta({ content: 'x' }))).failure.code, 'TRANSPORT')
  assert.equal((await finishOf(delta({ content: 'x' }, 'content_filter'))).failure.code, 'CONTENT_FILTER')
  assert.equal((await finishOf({ error: { message: 'engine crashed' } })).failure.code, 'PROVIDER_ERROR')
  assert.equal((await finishOf('{not json')).failure.code, 'INVALID_RESPONSE')
})

test('mapUsage reports cached input apart and keeps reasoning tokens', () => {
  assert.deepEqual(mapUsage({
    prompt_tokens: 10,
    completion_tokens: 5,
    total_tokens: 15,
    completion_tokens_details: { reasoning_tokens: 3 },
  }), { inputTokens: 10, outputTokens: 5, totalTokens: 15, reasoningTokens: 3 })
})
