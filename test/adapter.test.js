import assert from 'node:assert/strict'
import { test } from 'node:test'

import { BlockAssembler, LlmError } from '@deepseek-ai/dsh-llm'

import { DeepInfraAdapter } from '../lib/index.js'

const BASE = 'https://api.example.test/v1/openai'

function sse(...events) {
  const wire = events.map(event => `data: ${typeof event === 'string' ? event : JSON.stringify(event)}\n\n`).join('')
  return new Response(wire, { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

function adapter(fetch, { apiKey = 'di-key' } = {}) {
  return new DeepInfraAdapter({
    options: () => ({ baseURL: `${BASE}/`, apiKeyEnv: 'DEEPINFRA_API_KEY' }),
    resolveApiKey: async () => {
      if (apiKey === null) throw new LlmError('missing', 'MISSING_CREDENTIAL')
      return apiKey
    },
    fetch,
  })
}

const request = {
  provider: 'deepinfra',
  model: 'deepseek-ai/DeepSeek-V4-Flash',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
}

test('stream posts an attributed, authenticated request and assembles the reply', async () => {
  const calls = []
  const fetch = async (url, init) => {
    calls.push({ url, init })
    return sse(
      { choices: [{ index: 0, delta: { content: 'Hello' }, finish_reason: 'stop' }] },
      { choices: [], usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 } },
      '[DONE]',
    )
  }
  const assembler = new BlockAssembler()
  for await (const chunk of adapter(fetch).stream(request)) assembler.push(chunk)

  assert.equal(calls[0].url, `${BASE}/chat/completions`)
  assert.equal(calls[0].init.headers.authorization, 'Bearer di-key')
  assert.match(calls[0].init.headers['user-agent'], /^deepseek-harness\//)
  assert.equal(JSON.parse(calls[0].init.body).model, 'deepseek-ai/DeepSeek-V4-Flash')
  assert.deepEqual(assembler.blocks(), [{ type: 'text', text: 'Hello' }])
  assert.deepEqual(assembler.finish, { kind: 'stop' })
})

test('HTTP failures throw LlmError with routing facts', async () => {
  const fail = (status, body, headers = {}) => adapter(async () => new Response(JSON.stringify(body), { status, headers }))
  const errorOf = async (instance) => {
    try {
      for await (const _ of instance.stream(request)) { /* drain */ }
    } catch (error) {
      return error
    }
    assert.fail('expected a throw')
  }

  const limited = await errorOf(fail(429, { detail: { error: 'Model busy, retry later' } }, { 'retry-after': '2', 'x-request-id': 'req-1' }))
  assert.equal(limited.code, 'RATE_LIMIT')
  assert.equal(limited.failure.status, 429)
  assert.equal(limited.failure.providerRetryAfterMs, 2000)
  assert.equal(limited.failure.requestId, 'req-1')
  assert.match(limited.message, /Model busy/)

  assert.equal((await errorOf(fail(401, { detail: { error: 'invalid token' } }))).code, 'INVALID_CREDENTIAL')
  assert.equal((await errorOf(fail(402, { detail: { error: 'payment required' } }))).code, 'QUOTA')
  assert.equal((await errorOf(fail(400, { error: { message: "This model's maximum context length is 1000 tokens" } }))).code, 'CONTEXT_WINDOW_EXCEEDED')
  assert.equal((await errorOf(fail(400, { error: { message: 'bad tool schema' } }))).code, 'INVALID_REQUEST')
  assert.equal((await errorOf(fail(503, {}))).code, 'SERVER')
})

test('listModels offers nothing without a key and the live listing with one', async () => {
  const fetch = async () => Response.json({
    data: [{ id: 'zai-org/GLM-5.3', metadata: { context_length: 1048576, tags: ['chat'] } }],
  })
  assert.deepEqual(await adapter(fetch, { apiKey: null }).listModels('deepinfra'), [])
  assert.deepEqual(await adapter(fetch).listModels('deepinfra'), [
    { provider: 'deepinfra', id: 'zai-org/GLM-5.3', name: 'zai-org/GLM-5.3', inputModalities: ['text'] },
  ])
})

test('resolveModel answers context from the listing', async () => {
  const fetch = async () => Response.json({
    data: [{ id: 'zai-org/GLM-5.3', metadata: { context_length: 1048576, tags: ['chat'] } }],
  })
  const info = await adapter(fetch).resolveModel('deepinfra', 'zai-org/GLM-5.3')
  assert.deepEqual(info.context, { contextWindow: 1048576 })
})

test('caller abort surfaces as ABORTED', async () => {
  const controller = new AbortController()
  controller.abort()
  const fetch = async (_url, init) => {
    init.signal.throwIfAborted()
    return sse('[DONE]')
  }
  await assert.rejects(
    async () => { for await (const _ of adapter(fetch).stream({ ...request, signal: controller.signal })) { /* drain */ } },
    error => error.code === 'ABORTED',
  )
})
