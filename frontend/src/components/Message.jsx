import SourceDocuments from './SourceDocuments'

export default function Message({ message }) {
    const isUser = message.role === 'user'

    return (
        <div className={`message-row ${isUser ? 'user-row' : 'assistant-row'}`}>
            <div className="message-container">
                <div className="message-avatar">
                    {isUser ? (
                        <div className="user-avatar-circle">You</div>
                    ) : (
                        <div className="assistant-avatar-circle">AI</div>
                    )}
                </div>

                <div className="message-content">
                    {/* User Name / Role Label */}
                    <div className="message-author">
                        {isUser ? 'You' : 'Assistant'}
                    </div>

                    <div className="message-text">
                        {message.content}
                        {message.isStreaming && <span className="streaming-cursor">●</span>}
                    </div>

                    {!isUser && !message.isStreaming && message.sources?.length > 0 && (
                        <div className="message-sources">
                            <SourceDocuments sources={message.sources} />
                        </div>
                    )}

                    {!isUser && !message.isStreaming && message.model && (
                        <div className="message-footer-meta">
                            <span className="model-badge">{message.model}</span>
                            {message.queryTime && <span className="time-badge">{Math.round(message.queryTime)}ms</span>}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
