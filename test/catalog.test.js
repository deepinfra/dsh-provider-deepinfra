import assert from 'node:assert/strict'
import { test } from 'node:test'

import { Catalog, FALLBACK_MODELS, chatModels, resolvedModelInfo } from '../lib/catalog.js'

const LISTING = {
  data: [
    { id: 'deepseek-ai/DeepSeek-V4-Flash', metadata: { context_length: 1048576, tags: ['chat', 'prompt_cache', 'reasoning'] } },
    { id: 'Qwen/Qwen3.6-27B', metadata: { context_length: 262144, tags: ['chat', 'vlm', 'vision', 'reasoning_effort', 'reasoning'] } },
    { id: 'BAAI/bge-m3', metadata: { context_length: 8192, tags: ['embed'] } },
    { id: 'ByteDance/Seedance-1.5-Pro', metadata: { context_length: null, tags: ['video-gen'] } },
  ],
}

test('chatModels keeps only chat-tagged entries in listing order', () => {
  assert.deepEqual(chatModels(LISTING).map(model => model.id), ['deepseek-ai/DeepSeek-V4-Flash', 'Qwen/Qwen3.6-27B'])
  assert.deepEqual(chatModels({}), [])
})

test('resolvedModelInfo maps context, vision, and reasoning levels', () => {
  const [flash, qwen] = chatModels(LISTING)
  assert.deepEqual(resolvedModelInfo('deepinfra', flash.id, flash), {
    provider: 'deepinfra',
    id: flash.id,
    name: flash.id,
    inputModalities: ['text'],
    context: { contextWindow: 1048576 },
    reasoning: { efforts: [{ id: 'none', name: 'Off' }, { id: 'high', name: 'On' }] },
  })
  const info = resolvedModelInfo('deepinfra', qwen.id, qwen)
  assert.deepEqual(info.inputModalities, ['text', 'image'])
  assert.deepEqual(info.reasoning.efforts.map(effort => effort.id), ['none', 'low', 'medium', 'high'])
  assert.equal(resolvedModelInfo('deepinfra', 'x', { id: 'x', metadata: { tags: ['chat'] } }).reasoning, undefined)
})

test('resolvedModelInfo leaves an unlisted model unknown rather than guessed', () => {
  assert.deepEqual(resolvedModelInfo('deepinfra', 'acme/private', undefined), { provider: 'deepinfra', id: 'acme/private', name: 'acme/private' })
})

test('Catalog serves the fallback when the listing is unreachable', async () => {
  const catalog = new Catalog(() => Promise.reject(new Error('offline')))
  assert.equal(await catalog.models(), FALLBACK_MODELS)
})

test('Catalog caches within the TTL and keeps the last good listing on failure', async () => {
  let now = 0
  let calls = 0
  let fail = false
  const catalog = new Catalog(() => {
    calls++
    return fail ? Promise.reject(new Error('down')) : Promise.resolve(LISTING)
  }, { ttlMs: 100, now: () => now })

  const first = await catalog.models()
  await catalog.models()
  assert.equal(calls, 1)

  now = 200
  fail = true
  assert.equal(await catalog.models(), first)
  assert.equal(calls, 2)
})

test('Catalog shares one in-flight refresh between concurrent callers', async () => {
  let calls = 0
  const catalog = new Catalog(async () => { calls++; return LISTING })
  await Promise.all([catalog.models(), catalog.models(), catalog.find('Qwen/Qwen3.6-27B')])
  assert.equal(calls, 1)
})
