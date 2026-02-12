import { useState, useRef, useEffect } from 'react'

export default function QueryInput({ onSend, isLoading }) {
    const [query, setQuery] = useState('')
    const textareaRef = useRef(null)

    // Auto-resize textarea
    useEffect(() => {
        const el = textareaRef.current
        if (el) {
            el.style.height = 'auto'
            el.style.height = Math.min(el.scrollHeight, 200) + 'px'
        }
    }, [query])

    function handleSubmit(e) {
        e.preventDefault()
        if (query.trim() && !isLoading) {
            onSend(query.trim())
            setQuery('')
            if (textareaRef.current) {
                textareaRef.current.style.height = 'auto'
            }
        }
    }

    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e)
        }
    }

    return (
        <div className="input-area-wrapper">
            <div className="input-container-centered">
                <form onSubmit={handleSubmit} className="input-box">
                    <textarea
                        ref={textareaRef}
                        className="input-textarea"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask anything..."
                        rows={1}
                        disabled={isLoading}
                    />
                    <div className="input-actions">
                        <button
                            type="submit"
                            className="send-button"
                            disabled={!query.trim() || isLoading}
                            title="Send message"
                        >
                            {/* Up arrow icon */}
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-white">
                                <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}
