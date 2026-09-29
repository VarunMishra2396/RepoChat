/** Client for the repochat backend (http://localhost:8000). */

export interface Source {
  text: string
  source: string
  chunk_index: number
  score: number
}

interface SourcesEvent {
  type: 'sources'
  sources: Source[]
}

interface TokenEvent {
  type: 'token'
  text: string
}

interface DoneEvent {
  type: 'done'
}

type ServerEvent = SourcesEvent | TokenEvent | DoneEvent

const API = 'http://localhost:8000'

/**
 * Ask a question. Calls onSources once (with the retrieved chunks), then
 * onToken for every streamed answer token. Resolves when the stream ends.
 */
export async function askStream(
  question: string,
  onSources: (sources: Source[]) => void,
  onToken: (token: string) => void,
): Promise<void> {
  const res = await fetch(`${API}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok || !res.body) {
    throw new Error(`Backend error: ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const handleEvent = (evt: ServerEvent) => {
    if (evt.type === 'sources') onSources(evt.sources)
    else if (evt.type === 'token') onToken(evt.text)
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // SSE frames are separated by blank lines.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!line) continue
      handleEvent(JSON.parse(line.slice('data: '.length)) as ServerEvent)
    }
  }
}

export async function ingest(path: string): Promise<{ files: number; chunks: number }> {
  const res = await fetch(`${API}/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  if (!res.ok) throw new Error(`Ingest failed: ${res.status}`)
  return res.json()
}

export async function health(): Promise<{ status: string; chunks: number }> {
  const res = await fetch(`${API}/health`)
  return res.json()
}
