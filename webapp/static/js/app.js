const _nonReactive = { charts: {}, gpuSSE: null, logSSE: null };

function app() {
    return {
        view: 'dashboard',
        navItems: [
            { id: 'dashboard', label: 'Dashboard', icon: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM13.5 6a2.25 2.25 0 0 1 2.25-2.25H18A2.25 2.25 0 0 1 20.25 6v2.25A2.25 2.25 0 0 1 18 10.5h-2.25a2.25 2.25 0 0 1-2.25-2.25V6ZM13.5 15.75a2.25 2.25 0 0 1 2.25-2.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-2.25a2.25 2.25 0 0 1-2.25-2.25v-2.25Z"/></svg>' },
            { id: 'models', label: 'Models', icon: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125"/></svg>' },
            { id: 'servers', label: 'Servers', icon: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M5.25 14.25h13.5m-13.5 0a3 3 0 0 1-3-3m3 3a3 3 0 1 0 0 6h13.5a3 3 0 1 0 0-6m-16.5-3a3 3 0 0 1 3-3h13.5a3 3 0 0 1 3 3m-19.5 0a4.5 4.5 0 0 1 .9-2.7L5.737 5.1a3.375 3.375 0 0 1 2.7-1.35h7.126c1.062 0 2.062.5 2.7 1.35l2.587 3.45a4.5 4.5 0 0 1 .9 2.7m0 0a3 3 0 0 1-3 3m0 3h.008v.008h-.008v-.008Zm0-6h.008v.008h-.008v-.008Z"/></svg>' },
            { id: 'logs', label: 'Logs', icon: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"/></svg>' },
            { id: 'chat', label: 'Chat', icon: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z"/></svg>' },
        ],

        // Data
        gpu: {},
        memory: { total_gb: 0, used_gb: 0, available_gb: 0, percent: 0 },
        disk: { total_gb: 0, used_gb: 0, free_gb: 0, hf_cache_gb: 0 },
        inference: null,
        models: [],
        servers: [],

        // Toast
        toast: { show: false, message: '', type: 'success' },

        // Models
        showAddModel: false,
        newModel: { name: '', repo: '', dtype: 'auto', max_model_len: 8192, tensor_parallel_size: 1, gpu_memory_utilization: 0.9 },
        downloadProgress: {},

        // Servers
        startServerForm: { model_name: '', port: 8000, gpu_memory_utilization: 0.9 },
        serverStarting: false,
        startupLog: [],

        // Logs
        logsServerId: '',
        logLines: [],

        // Chat
        chatPort: 8000,
        chatTemp: 0.7,
        chatMessages: [],
        chatInput: '',
        chatStreaming: false,

        // SSE & Charts (stored in _nonReactive to avoid Alpine proxy recursion on Chart.js circular refs)
        historyMax: 120,
        history: {
            labels: [],
            tps: [],
            gpuUtil: [],
            gpuTemp: [],
            power: [],
            reqRunning: [],
            reqWaiting: [],
            kvCache: [],
            ttft: [],
            tpot: [],
        },

        async init() {
            this.startGpuStream();
            await this.loadModels();
            await this.loadServers();
            await this.attachActiveDownloads();
            this.$watch('view', (val) => {
                if (val === 'dashboard') {
                    this.destroyCharts();
                    this.$nextTick(() => this.updateCharts());
                }
            });
        },

        // --- GPU Stream ---
        startGpuStream() {
            if (_nonReactive.gpuSSE) _nonReactive.gpuSSE.close();
            _nonReactive.gpuSSE = new EventSource('/api/gpu/stream');
            _nonReactive.gpuSSE.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    this.gpu = data.gpu || {};
                    this.memory = data.memory || this.memory;
                    this.disk = data.disk || this.disk;
                    this.inference = data.inference || null;
                    this.pushHistory(data);
                } catch {}
            };
            _nonReactive.gpuSSE.onerror = () => {
                _nonReactive.gpuSSE.close();
                setTimeout(() => this.startGpuStream(), 5000);
            };
        },

        pushHistory(data) {
            const h = this.history;
            const now = new Date();
            const label = now.getHours().toString().padStart(2,'0') + ':' +
                          now.getMinutes().toString().padStart(2,'0') + ':' +
                          now.getSeconds().toString().padStart(2,'0');
            h.labels.push(label);
            h.tps.push(data.inference?.tokens_per_sec ?? 0);
            h.gpuUtil.push(data.gpu?.utilization_gpu_pct ?? 0);
            h.gpuTemp.push(data.gpu?.temperature_c ?? 0);
            h.power.push(data.gpu?.power_draw_w ?? 0);
            h.reqRunning.push(data.inference?.requests_running ?? 0);
            h.reqWaiting.push(data.inference?.requests_waiting ?? 0);
            h.kvCache.push(data.inference?.kv_cache_usage_pct ?? 0);
            h.ttft.push(data.inference?.avg_ttft_ms ?? 0);
            h.tpot.push(data.inference?.avg_tpot_ms ?? 0);

            // Trim to max
            for (const key of Object.keys(h)) {
                if (h[key].length > this.historyMax) h[key].shift();
            }

            this.$nextTick(() => this.updateCharts());
        },

        destroyCharts() {
            for (const key of Object.keys(_nonReactive.charts)) {
                if (_nonReactive.charts[key]) { _nonReactive.charts[key].destroy(); _nonReactive.charts[key] = null; }
            }
        },

        _createChart(id, datasets) {
            const el = document.getElementById(id);
            if (!el) return null;
            return new Chart(el.getContext('2d'), {
                type: 'line',
                data: { labels: [...this.history.labels], datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: {
                        legend: { position: 'top', labels: { color: '#9ca3af', font: { size: 10 }, boxWidth: 8, padding: 8, usePointStyle: true } },
                        tooltip: { backgroundColor: '#1a1d2e', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, titleColor: '#e5e7eb', bodyColor: '#9ca3af', padding: 8, bodyFont: { size: 11 } },
                    },
                    scales: {
                        x: { display: false },
                        y: { ticks: { color: '#6b7280', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true },
                    },
                    elements: { point: { radius: 0, hoverRadius: 3 }, line: { tension: 0.3, borderWidth: 1.5 } },
                },
            });
        },

        _ds(label, data, color) {
            return { label, data: [...data], borderColor: color, backgroundColor: color + '18', fill: true };
        },

        updateCharts() {
            if (typeof Chart === 'undefined' || this.view !== 'dashboard') return;
            const h = this.history;
            if (h.labels.length < 2) return;

            // Create or update each chart
            const chartDefs = {
                tps:  { id: 'chartTps',      sets: [this._ds('tok/s', h.tps, '#30a2ff')] },
                gpu:  { id: 'chartGpu',       sets: [this._ds('Util %', h.gpuUtil, '#10b981'), this._ds('Temp °C', h.gpuTemp, '#ef4444'), this._ds('Power W', h.power, '#fdb517')] },
                req:  { id: 'chartRequests',  sets: [this._ds('Active', h.reqRunning, '#10b981'), this._ds('Queue', h.reqWaiting, '#f59e0b'), this._ds('KV Cache %', h.kvCache, '#a855f7')] },
                lat:  { id: 'chartLatency',   sets: [this._ds('TTFT ms', h.ttft, '#06b6d4'), this._ds('TPOT ms', h.tpot, '#f97316')] },
            };

            for (const [key, def] of Object.entries(chartDefs)) {
                if (!_nonReactive.charts[key]) {
                    _nonReactive.charts[key] = this._createChart(def.id, def.sets);
                } else {
                    const chart = _nonReactive.charts[key];
                    chart.data.labels = [...h.labels];
                    def.sets.forEach((ds, i) => { chart.data.datasets[i].data = ds.data; });
                    chart.update('none');
                }
            }
        },

        // --- Models ---
        async loadModels() {
            try { this.models = await api.get('/api/profiles'); } catch (e) { this.showToast(e.message, 'error'); }
        },

        async addModel() {
            try {
                await api.post('/api/profiles', this.newModel);
                this.showAddModel = false;
                this.newModel = { name: '', repo: '', dtype: 'auto', max_model_len: 8192, tensor_parallel_size: 1, gpu_memory_utilization: 0.9 };
                await this.loadModels();
                this.showToast('Model added');
            } catch (e) { this.showToast(e.message, 'error'); }
        },

        async deleteModel(name) {
            if (!confirm(`Delete model "${name}"?`)) return;
            try {
                await api.del(`/api/profiles/${name}`);
                await this.loadModels();
                this.showToast('Model deleted');
            } catch (e) { this.showToast(e.message, 'error'); }
        },

        async attachActiveDownloads() {
            try {
                const r = await api.get('/api/downloads/active');
                for (const name of (r.downloads || [])) {
                    if (this.models.some(m => m.name === name)) this.downloadModel(name);
                }
            } catch {}
        },

        async downloadModel(name) {
            try {
                try {
                    await api.post(`/api/downloads/${name}/start`);
                } catch (e) {
                    // 409 = already in progress — just attach to existing stream
                    if (!/already in progress/i.test(e.message)) throw e;
                }
                this.downloadProgress[name] = 'Starting...';
                const es = new EventSource(`/api/downloads/${name}/stream`);
                es.addEventListener('progress', (e) => {
                    this.downloadProgress[name] = e.data;
                });
                es.addEventListener('complete', (e) => {
                    delete this.downloadProgress[name];
                    es.close();
                    this.loadModels();
                    this.showToast(e.data);
                });
                es.addEventListener('error', (e) => {
                    delete this.downloadProgress[name];
                    es.close();
                    this.showToast(e.data || 'Download failed', 'error');
                });
                es.onerror = () => {
                    delete this.downloadProgress[name];
                    es.close();
                };
            } catch (e) { this.showToast(e.message, 'error'); }
        },

        // --- Servers ---
        async loadServers() {
            try {
                this.servers = await api.get('/api/servers');
                if (this.servers.length > 0) {
                    this.chatPort = this.servers[0].port;
                }
            } catch (e) { this.showToast(e.message, 'error'); }
        },

        async startServer() {
            this.serverStarting = true;
            this.startupLog = [];
            try {
                const srv = await api.post('/api/servers', this.startServerForm);
                await this.loadServers();
                this.showToast(`Server starting: ${srv.model_name} on port ${srv.port}`);

                // Switch to logs view to show progress
                this.logsServerId = srv.server_id;
                this.logLines = [];
                this.view = 'logs';

                // Stream startup logs into both startupLog and logLines
                const es = new EventSource(`/api/servers/${srv.server_id}/startup-stream`);
                es.addEventListener('log', (e) => {
                    this.startupLog.push(e.data);
                    this.logLines.push(e.data);
                    if (this.logLines.length > 2000) this.logLines.splice(0, 500);
                    this.$nextTick(() => {
                        if (this.$refs.logsEl) this.$refs.logsEl.scrollTop = this.$refs.logsEl.scrollHeight;
                    });
                });
                es.addEventListener('ready', (e) => {
                    this.showToast(e.data);
                    this.serverStarting = false;
                    es.close();
                    this.loadServers();
                });
                es.addEventListener('error', (e) => {
                    this.showToast(e.data || 'Startup error', 'error');
                    this.serverStarting = false;
                    es.close();
                    this.loadServers();
                });
                es.onerror = () => {
                    this.serverStarting = false;
                    es.close();
                };
            } catch (e) {
                this.showToast(e.message, 'error');
                this.serverStarting = false;
            }
        },

        async stopServer(serverId) {
            try {
                await api.del(`/api/servers/${serverId}`);
                await this.loadServers();
                this.showToast('Server stopped');
            } catch (e) { this.showToast(e.message, 'error'); }
        },

        // --- Logs ---
        viewLogs(serverId) {
            this.view = 'logs';
            this.logsServerId = serverId || (this.servers.length > 0 ? this.servers[0].server_id : '');
            this.loadLogs();
        },

        async loadLogs() {
            if (_nonReactive.logSSE) { _nonReactive.logSSE.close(); _nonReactive.logSSE = null; }
            if (!this.logsServerId) { this.logLines = []; return; }

            try {
                const data = await api.get(`/api/logs/${this.logsServerId}`);
                this.logLines = data.lines || [];
                this.$nextTick(() => {
                    if (this.$refs.logsEl) this.$refs.logsEl.scrollTop = this.$refs.logsEl.scrollHeight;
                });

                // Start streaming
                _nonReactive.logSSE = new EventSource(`/api/logs/${this.logsServerId}/stream`);
                _nonReactive.logSSE.onmessage = (e) => {
                    this.logLines.push(e.data);
                    if (this.logLines.length > 2000) this.logLines.splice(0, 500);
                    this.$nextTick(() => {
                        if (this.$refs.logsEl) this.$refs.logsEl.scrollTop = this.$refs.logsEl.scrollHeight;
                    });
                };
                _nonReactive.logSSE.onerror = () => { _nonReactive.logSSE.close(); _nonReactive.logSSE = null; };
            } catch (e) { this.showToast(e.message, 'error'); }
        },

        // --- Chat ---
        async sendChat() {
            const text = this.chatInput.trim();
            if (!text || this.chatStreaming) return;

            this.chatMessages.push({ role: 'user', content: text });
            this.chatInput = '';
            this.chatStreaming = true;
            this.chatMessages.push({ role: 'assistant', content: '' });

            const msgIdx = this.chatMessages.length - 1;

            this.$nextTick(() => {
                if (this.$refs.chatEl) this.$refs.chatEl.scrollTop = this.$refs.chatEl.scrollHeight;
            });

            try {
                await api.streamChat(
                    '/api/chat/completions',
                    {
                        messages: this.chatMessages.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
                        port: this.chatPort,
                        temperature: this.chatTemp,
                        max_tokens: 4096,
                        stream: true,
                    },
                    (token) => {
                        this.chatMessages[msgIdx].content += token;
                        this.$nextTick(() => {
                            if (this.$refs.chatEl) this.$refs.chatEl.scrollTop = this.$refs.chatEl.scrollHeight;
                        });
                    },
                    () => { this.chatStreaming = false; },
                );
            } catch (e) {
                this.chatMessages[msgIdx].content = `Error: ${e.message}`;
                this.chatStreaming = false;
            }
        },

        renderMarkdown(text) {
            if (!text) return '';
            try { return marked.parse(text); } catch { return text; }
        },

        // --- Toast ---
        showToast(message, type = 'success') {
            this.toast = { show: true, message, type };
            setTimeout(() => { this.toast.show = false; }, 4000);
        },
    };
}
