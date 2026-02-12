import { useRef, useEffect } from 'react'
import Message from './Message'

export default function ChatInterface({ messages, suggestions, onSuggestionClick }) {
    const chatEndRef = useRef(null)

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    if (messages.length === 0) {
        return (
            <div className="chat-scroll-area">
                <div className="empty-state">
                    <p className="empty-state-kicker">Start a conversation</p>
                    <h1>How can I help today?</h1>

                    <div className="suggestions-grid">
                        {suggestions.map((s, i) => (
                            <button
                                key={i}
                                className="suggestion-card"
                                onClick={() => onSuggestionClick(s)}
                            >
                                <span className="suggestion-title">{s}</span>
                                <span className="suggestion-subtitle">RAG prompt idea</span>
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="chat-scroll-area">
            <div className="chat-container">
                {messages.map((msg, i) => (
                    <Message key={i} message={msg} />
                ))}
                <div ref={chatEndRef} />
            </div>
        </div>
    )
}
