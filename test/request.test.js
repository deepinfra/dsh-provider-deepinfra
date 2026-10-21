import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { test } from 'node:test'

import { REQUEST_FIELD_DISPOSITION, buildRequestBody, toChatMessages } from '../lib/request.js'

const text = value => ({ type: 'text', text: value })
const image = id => ({ type: 'image', attachment: { attachmentId: id } })

test('every GenerateOptions field of the installed dsh-llm has a disposition', () => {
  const require = createRequire(import.meta.url)
  const types = readFileSync(join(dirname(require.resolve('@deepseek-ai/dsh-llm/package.json')), 'lib/types/types.d.ts'), 'utf8')
  const body = types.slice(types.indexOf('export interface GenerateOptions {'))
  const declared = [...body.slice(0, body.indexOf('\n}')).matchAll(/^ {4}(\w+)\??:/gm)].map(match => match[1])
  assert.ok(declared.length > 5)
  assert.deepEqual(declared.filter(field => !(field in REQUEST_FIELD_DISPOSITION)), [])
})

test('leading system history joins the one-shot system text', () => {
  const messages = toChatMessages({
    system: 'one-shot',
    messages: [
      { role: 'system', content: [text('prompt')] },
      { role: 'user', content: [text('hi')] },
      { role: 'developer', content: [text('notice')] },
    ],
  })
  assert.deepEqual(messages, [
    { role: 'system', content: 'one-shot\n\nprompt' },
    { role: 'user', content: 'hi' },
    { role: 'system', content: 'notice' },
  ])
})

test('assistant history carries text, reasoning, and tool calls; tool results follow', () => {
  const messages = toChatMessages({
    messages: [
      {
        role: 'assistant',
        content: [
          { type: 'reasoning', text: 'think' },
          text('calling'),
          { type: 'tool-call', id: 'call_1', name: 'read', arguments: '{"path":"a"}' },
          { type: 'tool-call', id: 'call_2', name: 'ls', arguments: '' },
        ],
      },
      { role: 'tool', toolCallId: 'call_1', content: [text('file body')] },
      { role: 'tool', toolCallId: 'call_2', content: [] },
      { role: 'assistant', content: [{ type: 'reasoning', text: 'only thinking' }] },
    ],
  })
  assert.deepEqual(messages, [
    {
      role: 'assistant',
      content: 'calling',
      reasoning_content: 'think',
      tool_calls: [
        { id: 'call_1', type: 'function', function: { name: 'read', arguments: '{"path":"a"}' } },
        { id: 'call_2', type: 'function', function: { name: 'ls', arguments: '{}' } },
      ],
    },
    { role: 'tool', tool_call_id: 'call_1', content: 'file body' },
    { role: 'tool', tool_call_id: 'call_2', content: '(no output)' },
  ])
})

test('images become image_url parts; tool images follow the tool batch as user content', () => {
  const images = new Map([
    ['img-1', { url: 'data:image/png;base64,AAA', handleText: 'Image one.' }],
    ['img-2', { url: 'data:image/png;base64,BBB', handleText: 'Image two.' }],
  ])
  const messages = toChatMessages({
    messages: [
      { role: 'user', content: [text('look'), image('img-1')] },
      { role: 'tool', toolCallId: 'call_1', content: [text('shot'), image('img-2')] },
      { role: 'user', content: [text('next')] },
    ],
  }, images)
  assert.deepEqual(messages, [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'look' },
        { type: 'text', text: 'Image one.' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AAA' } },
      ],
    },
    { role: 'tool', tool_call_id: 'call_1', content: 'shot\nImage two.' },
    { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,BBB' } }] },
    { role: 'user', content: 'next' },
  ])
})

test('buildRequestBody maps controls and withholds deferred tools', () => {
  const body = buildRequestBody({
    provider: 'deepinfra',
    model: 'deepseek-ai/DeepSeek-V4-Flash',
    messages: [{ role: 'user', content: [text('hi')] }],
    tools: [
      { name: 'read', description: 'read a file', parameters: { type: 'object' } },
      { name: 'later', description: 'deferred', parameters: { type: 'object' }, deferLoading: true },
    ],
    reasoningEffort: 'low',
    temperature: 0,
    maxTokens: 512,
    stop: ['END'],
    sessionId: 'session-1',
  })
  assert.deepEqual(body, {
    model: 'deepseek-ai/DeepSeek-V4-Flash',
    messages: [{ role: 'user', content: 'hi' }],
    stream: true,
    stream_options: { include_usage: true },
    tools: [{ type: 'function', function: { name: 'read', description: 'read a file', parameters: { type: 'object' } } }],
    reasoning_effort: 'low',
    temperature: 0,
    max_tokens: 512,
    stop: ['END'],
    prompt_cache_key: 'session-1',
  })
})
