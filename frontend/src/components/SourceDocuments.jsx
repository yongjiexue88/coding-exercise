import { useState } from 'react'

export default function SourceDocuments({ sources }) {
    const [isOpen, setIsOpen] = useState(false)

    if (!sources || sources.length === 0) return null

    return (
        <div className="sources-wrapper">
            <button className="source-toggle-btn" onClick={() => setIsOpen(!isOpen)}>
                {isOpen ? '▼' : '▶'} {sources.length} Sources
            </button>

            {isOpen && (
                <div className="sources-grid">
                    {sources.map((source, i) => (
                        <div key={i} className="source-item">
                            <h4>{source.source} ({(source.relevance_score * 100).toFixed(0)}%)</h4>
                            <p>{source.content}</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
