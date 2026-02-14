import { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import QueryInput from './components/QueryInput'

const API_BASE = import.meta.env.PROD ? 'https://rag-backend-963140776209.us-central1.run.app' : 'http://localhost:8000'
const PROGRESS_STEP_ORDER = ['understand', 'retrieve', 'draft', 'verify', 'finalize']
const PROGRESS_LABELS = {
    understand: 'Understanding question',
    retrieve: 'Finding relevant sources',
    draft: 'Drafting answer',
    verify: 'Verifying answer',
    finalize: 'Finalizing response',
}

const SUGGESTIONS = [
    'Which NFL team won Super Bowl 50?',
    'What color was used to emphasize the 50th anniversary of the Super Bowl?',
    'Who was the Super Bowl 50 MVP?',
    'Which group headlined the Super Bowl 50 halftime show?',
]

function createInitialProgressSteps() {
    return PROGRESS_STEP_ORDER.map((step) => ({
        step,
        label: PROGRESS_LABELS[step],
        state: 'pending',
    }))
}

function mergeProgressSteps(existingSteps, event) {
    const merged = new Map(
        PROGRESS_STEP_ORDER.map((step) => [
            step,
            { step, label: PROGRESS_LABELS[step], state: 'pending' },
        ])
    )

        ; (existingSteps || []).forEach((item) => {
            if (!item?.step || !merged.has(item.step)) return
            merged.set(item.step, { ...merged.get(item.step), ...item })
        })

    if (event?.step && merged.has(event.step)) {
        const current = merged.get(event.step)
        merged.set(event.step, {
            ...current,
            label: event.label || current.label,
            state: event.state || current.state,
            meta: event.meta || current.meta,
        })
    }

    return PROGRESS_STEP_ORDER.map((step) => merged.get(step))
}

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
        const assistantMsg = {
            role: 'assistant',
            content: '',
            sources: [],
            model: '',
            isStreaming: true,
            progressSteps: createInitialProgressSteps(),
            qualitySummary: null,
        }
        setMessages(prev => [...prev, assistantMsg])

        try {
            let sawStatusEvent = false

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

                    let event = null
                    try {
                        event = JSON.parse(jsonStr)
                    } catch (parseErr) {
                        // Skip malformed JSON lines
                        continue
                    }

                    if (event.type === 'status') {
                        sawStatusEvent = true
                        setMessages(prev => {
                            const updated = [...prev]
                            const last = updated[updated.length - 1]
                            updated[updated.length - 1] = {
                                ...last,
                                progressSteps: mergeProgressSteps(last.progressSteps, event),
                            }
                            return updated
                        })
                    } else if (event.type === 'chunk') {
                        setMessages(prev => {
                            const updated = [...prev]
                            const last = updated[updated.length - 1]
                            updated[updated.length - 1] = { ...last, content: last.content + event.content }
                            return updated
                        })
                    } else if (event.type === 'done') {
                        // If status/done arrive in one buffered chunk, React may batch updates and
                        // skip rendering the interim "thinking process". Yield one frame first.
                        await new Promise((resolve) => {
                            if (typeof window !== 'undefined' && window.requestAnimationFrame) {
                                window.requestAnimationFrame(() => resolve())
                                return
                            }
                            setTimeout(resolve, 0)
                        })

                        setMessages(prev => {
                            const updated = [...prev]
                            const last = updated[updated.length - 1]
                            const sources = event.sources || []
                            const qualitySummary = event.quality_summary || {
                                verification: sources.length > 0 ? 'verified' : 'none',
                                sources_used: sources.length,
                            }
                            const progressSteps = sawStatusEvent
                                ? last.progressSteps
                                : mergeProgressSteps(last.progressSteps, {
                                    step: 'finalize',
                                    state: 'completed',
                                    label: PROGRESS_LABELS.finalize,
                                })
                            updated[updated.length - 1] = {
                                ...last,
                                isStreaming: false,
                                sources,
                                model: event.model || '',
                                queryTime: event.query_time_ms,
                                qualitySummary,
                                progressSteps,
                            }
                            return updated
                        })
                    } else if (event.type === 'error') {
                        throw new Error(event.content || 'Query failed')
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
                    progressSteps: [],
                    qualitySummary: null,
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
                    <span className="brand-mark">C</span>
                    <span className="brand-name">Chat</span>
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
