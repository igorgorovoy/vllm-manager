const api = {
    async get(url) {
        const resp = await fetch(url);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        return resp.json();
    },

    async post(url, data) {
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        return resp.json();
    },

    async del(url) {
        const resp = await fetch(url, { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        return resp.json();
    },

    sse(url, { onMessage, onEvent, onError }) {
        const es = new EventSource(url);
        if (onMessage) es.onmessage = (e) => onMessage(e.data);
        if (onEvent) {
            for (const [event, handler] of Object.entries(onEvent)) {
                es.addEventListener(event, (e) => handler(e.data));
            }
        }
        es.onerror = (e) => {
            if (onError) onError(e);
            es.close();
        };
        return es;
    },

    async streamChat(url, data, onChunk, onDone) {
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const chunk = line.slice(6).trim();
                    if (chunk === '[DONE]') {
                        if (onDone) onDone();
                        return;
                    }
                    try {
                        const parsed = JSON.parse(chunk);
                        const token = parsed.choices?.[0]?.delta?.content;
                        if (token) onChunk(token);
                    } catch {}
                }
            }
        }
        if (onDone) onDone();
    },
};
