import { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import QueryInput from './components/QueryInput'

const API_BASE = import.meta.env.PROD ? 'https://rag-backend-963140776209.us-central1.run.app' : 'http://localhost:8000'

const SUGGESTIONS = [
    'Who won Super Bowl 50?',
    'When did Nikola Tesla die?',
    'What is the capital of Kenya?',
    'What group bought Cyprus after the Norman conquest?',
]

export default function App() {
    const [messages, setMessages] = useState([])
    const [isLoading, setIsLoading] = useState(false)

    function handleNewChat() {
        setMessages([])
    }

    async function handleSend(query) {
        if (!query.trim() || isLoading) return

        // Add user message
        const userMsg = { role: 'user', content: query }
        setMessages(prev => [...prev, userMsg])
        setIsLoading(true)

        // Add placeholder assistant message
        const assistantMsg = { role: 'assistant', content: '', sources: [], model: '', isStreaming: true }
        setMessages(prev => [...prev, assistantMsg])

        try {
            const res = await fetch(`${API_BASE}/query/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: 3 }),
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Query failed')
            }

            const reader = res.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop() || ''

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue
                    const jsonStr = line.slice(6).trim()
                    if (!jsonStr) continue

                    try {
                        const event = JSON.parse(jsonStr)

                        if (event.type === 'chunk') {
                            setMessages(prev => {
                                const updated = [...prev]
                                const last = updated[updated.length - 1]
                                updated[updated.length - 1] = { ...last, content: last.content + event.content }
                                return updated
                            })
                        } else if (event.type === 'done') {
                            setMessages(prev => {
                                const updated = [...prev]
                                const last = updated[updated.length - 1]
                                updated[updated.length - 1] = {
                                    ...last,
                                    isStreaming: false,
                                    sources: event.sources || [],
                                    model: event.model || '',
                                    queryTime: event.query_time_ms,
                                }
                                return updated
                            })
                        } else if (event.type === 'error') {
                            throw new Error(event.content)
                        }
                    } catch (parseErr) {
                        // Skip malformed JSON lines
                    }
                }
            }
        } catch (err) {
            setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1] = {
                    role: 'assistant',
                    content: `Error: ${err.message}`,
                    isStreaming: false,
                    sources: [],
                }
                return updated
            })
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="app-shell">
            <header className="topbar">
                <div className="brand-block">
                    <span className="brand-mark">A</span>
                    <span className="brand-name">Assistant</span>
                </div>

                <button
                    type="button"
                    className="new-chat-action"
                    onClick={handleNewChat}
                    disabled={messages.length === 0 || isLoading}
                >
                    New chat
                </button>
            </header>

            <main className="chat-main">

                <ChatInterface
                    messages={messages}
                    suggestions={SUGGESTIONS}
                    onSuggestionClick={handleSend}
                />

                <QueryInput onSend={handleSend} isLoading={isLoading} />
            </main>
        </div>
    )
}
