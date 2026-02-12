import { useState } from 'react'

export default function Sidebar({ isOpen, onNewChat, toggleSidebar }) {
    const [history] = useState([
        { id: 1, title: 'Previous Chat History' },
        { id: 2, title: 'React Hooks Explanation' },
        { id: 3, title: 'CSS Grid Layouts' },
    ])

    return (
        <>
            <div className={`sidebar ${isOpen ? 'open' : ''}`}>
                <div className="sidebar-header">
                    <button className="new-chat-btn" onClick={onNewChat} title="New chat">
                        <span className="plus-icon">+</span>
                        <span className="new-chat-text">New chat</span>
                        <span className="edit-icon">✎</span>
                    </button>

                    <button className="close-sidebar-mobile" onClick={toggleSidebar}>
                        ✕
                    </button>
                </div>

                <div className="sidebar-content">
                    <div className="history-group">
                        <h3 className="history-label">Recent</h3>
                        <ul className="history-list">
                            {history.map((item) => (
                                <li key={item.id} className="history-item">
                                    <button className="history-btn">
                                        <span className="msg-icon">💬</span>
                                        <span className="history-title">{item.title}</span>
                                    </button>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                <div className="sidebar-footer">
                    <button className="user-profile-btn">
                        <div className="user-avatar">👤</div>
                        <div className="user-info">
                            <span className="user-name">User</span>
                            <span className="user-plan">Free Plan</span>
                        </div>
                    </button>
                </div>
            </div>

            {isOpen && <div className="sidebar-overlay" onClick={toggleSidebar} />}
        </>
    )
}
