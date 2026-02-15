import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import App from './App';

// Mock child components to avoid testing their implementation details
vi.mock('./components/ChatInterface', () => ({
    default: ({ suggestions }) => (
        <div data-testid="chat-interface">
            {suggestions.map(s => <button key={s}>{s}</button>)}
        </div>
    )
}));

vi.mock('./components/QueryInput', () => ({
    default: ({ onSend, isLoading }) => (
        <div data-testid="query-input">
            <button disabled={isLoading} onClick={() => onSend('test query')}>Send</button>
        </div>
    )
}));

describe('App', () => {
    it('renders the branding and new chat button', () => {
        render(<App />);
        expect(screen.getByText('Chat')).toBeInTheDocument();
        expect(screen.getByText('New chat')).toBeInTheDocument();
    });

    it('renders the main layout components', () => {
        render(<App />);
        expect(screen.getByTestId('chat-interface')).toBeInTheDocument();
        expect(screen.getByTestId('query-input')).toBeInTheDocument();
    });
});
