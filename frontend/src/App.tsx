import { useState } from 'react'
import { askStream, Source } from './api'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const send = async () => {
    const question = input.trim()
    if (!question || busy) return
    setInput('')
    setError(null)
    setBusy(true)

    const history: Message[] = [...messages, { role: 'user', content: question }]
    setMessages(history)

    // Placeholder assistant message that we fill in as tokens stream.
    const assistant: Message = { role: 'assistant', content: '', sources: [] }
    setMessages([...history, assistant])

    try {
      await askStream(
        question,
        (sources) => {
          assistant.sources = sources
          setMessages([...history, { ...assistant }])
        },
        (token) => {
          assistant.content += token
          setMessages([...history, { ...assistant }])
        },
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
      setMessages(history)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>repochat</h1>
        <p>Ask questions about your codebase. Answers cite their sources.</p>
      </header>

      {error && <div className="error">{error} (is the backend running on :8000?)</div>}

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty">
            Ingest a folder first (<code>python cli.py ingest /path/to/repo</code>),
            then ask something like “how does chunking work?”.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">
              {m.content || (busy && m.role === 'assistant' ? '…' : '')}
            </div>
            {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
              <details className="sources">
                <summary>{m.sources.length} sources</summary>
                <ul>
                  {m.sources.map((s, j) => (
                    <li key={j}>
                      <span className="src-path">
                        [{j + 1}] {s.source}
                      </span>{' '}
                      <span className="src-score">{(s.score * 100).toFixed(1)}%</span>
                      <pre className="src-text">{s.text.slice(0, 400)}</pre>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}
      </main>

      <footer className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="Ask about your codebase…"
          disabled={busy}
        />
        <button onClick={send} disabled={busy || !input.trim()}>
          {busy ? '…' : 'Ask'}
        </button>
      </footer>
    </div>
  )
}
