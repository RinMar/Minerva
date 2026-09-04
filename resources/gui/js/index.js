var modelReady = false;

// ── QWebChannel bridge ──
if (typeof qt !== 'undefined') {
    new QWebChannel(qt.webChannelTransport, function (channel) {
        window.pyBridge = channel.objects.pyBridge;
        // Expose to child frames periodically if they load later
        setInterval(() => {
            let cFrame = document.getElementById('chat-frame');
            let gFrame = document.getElementById('graph-frame');
            if(cFrame && cFrame.contentWindow && !cFrame.contentWindow.pyBridge) {
                cFrame.contentWindow.pyBridge = window.pyBridge;
            }
        }, 100);
    });
}

// ── Dispatch calls to Chat Iframe ──
function startAssistantMessage() {
    var win = document.getElementById('chat-frame').contentWindow;
    if (win && win.startAssistantMessage) win.startAssistantMessage();
}
function appendAssistantToken(token) {
    var win = document.getElementById('chat-frame').contentWindow;
    if (win && win.appendAssistantToken) win.appendAssistantToken(token);
}
function setAssistantState(state) {
    var win = document.getElementById('chat-frame').contentWindow;
    if (win && win.setAssistantState) win.setAssistantState(state);
}
function appendMessage(role, text) {
    var win = document.getElementById('chat-frame').contentWindow;
    if (win && win.appendMessage) win.appendMessage(role, text);
}
function clearChat() {
    var win = document.getElementById('chat-frame').contentWindow;
    if (win && win.clearChat) win.clearChat();
}

// ── Dispatch calls to Graph Iframe ──
function updateGraph(dataUpdate) {
    var win = document.getElementById('graph-frame').contentWindow;
    if (win && win.updateGraph) win.updateGraph(dataUpdate);
}
function clearGraph() {
    var win = document.getElementById('graph-frame').contentWindow;
    if (win && win.clearGraph) win.clearGraph();
}

// ── Model Loader Controls (Called by Python) ──
function showModelLoader(text) {
    var win = document.getElementById('chat-frame').contentWindow;
    if (win && win.showModelLoader) win.showModelLoader(text);
}

function hideModelLoader() {
    var win = document.getElementById('chat-frame').contentWindow;
    if (win && win.hideModelLoader) win.hideModelLoader();
}

// ── Settings Loader Controls (Called by Python or Chat UI) ──
let vramData = null;

function showSettings() {
    const modal = document.getElementById('settings-modal');
    
    if (window.pyBridge) {
        window.pyBridge.get_vram_info(function(vramJson) {
            vramData = JSON.parse(vramJson);
            window.pyBridge.get_model_settings(function(settingsJson) {
                const settings = JSON.parse(settingsJson);
                initSettingsUI(settings);
                modal.style.display = 'flex';
            });
        });
    } else {
        // Fallback for standalone browser dev testing
        vramData = {
            has_gpu: true,
            has_cuda: true,
            vram_monitored: true,
            total_mb: 8192,
            free_mb: 5200,
            total_layers: 65,
            per_layer_mb: 80.0,
            kv_per_token_per_layer_mb: 0.00012,
            overhead_mb: 300,
            ctx_min: 2048,
            ctx_max: 40960
        };
        initSettingsUI({ n_gpu_layers: 0, n_ctx: 8192 });
        modal.style.display = 'flex';
    }
}

function initSettingsUI(settings) {
    const gpuSlider = document.getElementById('gpu-layer-slider');
    const ctxSlider = document.getElementById('ctx-slider');

    gpuSlider.max = vramData.total_layers;
    ctxSlider.min = vramData.ctx_min;
    ctxSlider.max = vramData.ctx_max;

    gpuSlider.value = settings.n_gpu_layers;
    ctxSlider.value = settings.n_ctx;

    // Attach listeners
    gpuSlider.oninput = () => recalcLimits('gpu');
    ctxSlider.oninput = () => recalcLimits('ctx');

    recalcLimits();
}

function recalcLimits(triggeredBy) {
    const gpuSlider = document.getElementById('gpu-layer-slider');
    const ctxSlider = document.getElementById('ctx-slider');
    const gpuMarker = document.getElementById('gpu-limit-marker');
    const ctxMarker = document.getElementById('ctx-limit-marker');

    if (!vramData || !vramData.has_gpu) {
        gpuSlider.value = 0;
        gpuSlider.disabled = true;
        ctxSlider.disabled = false;
        if (gpuMarker) gpuMarker.style.display = 'none';
        if (ctxMarker) ctxMarker.style.display = 'none';
        updateLabels(0, parseInt(ctxSlider.value));
        return;
    }

    gpuSlider.disabled = false;
    ctxSlider.disabled = false;

    let n = parseInt(gpuSlider.value);
    let C = parseInt(ctxSlider.value);

    // If VRAM is not real-time monitored or free_mb is not set, allow full range
    if (!vramData.vram_monitored || !vramData.free_mb) {
        if (gpuMarker) gpuMarker.style.display = 'none';
        if (ctxMarker) ctxMarker.style.display = 'none';
        updateLabels(n, C);
        return;
    }

    // Monitored VRAM clamping logic
    const freeMb = vramData.free_mb;
    const overheadMb = vramData.overhead_mb || 300;
    const perLayerMb = vramData.per_layer_mb || 80.0;
    const kvPerTokenPerLayerMb = vramData.kv_per_token_per_layer_mb || 0.00012;
    const totalLayers = vramData.total_layers || 65;
    const ctxMin = vramData.ctx_min || 2048;
    const ctxMax = vramData.ctx_max || 40960;

    let maxLayers = totalLayers;
    let maxCtx = ctxMax;

    if (triggeredBy === 'ctx') {
        // User is moving Context slider. Clamp C based on fixed n.
        if (n > 0) {
            const remainingForKv = freeMb - overheadMb - (n * perLayerMb);
            if (remainingForKv > 0) {
                maxCtx = Math.floor(remainingForKv / (n * kvPerTokenPerLayerMb));
            } else {
                maxCtx = ctxMin;
            }
            maxCtx = Math.max(ctxMin, Math.min(maxCtx, ctxMax));
        }
        if (C > maxCtx) {
            C = maxCtx;
            ctxSlider.value = C;
        }
        // Calculate limit for n given this C
        const costPerLayer = perLayerMb + (C * kvPerTokenPerLayerMb);
        maxLayers = Math.floor((freeMb - overheadMb) / costPerLayer);
        maxLayers = Math.max(0, Math.min(maxLayers, totalLayers));
        
    } else if (triggeredBy === 'gpu') {
        // User is moving GPU slider. Clamp n based on fixed C.
        const costPerLayer = perLayerMb + (C * kvPerTokenPerLayerMb);
        maxLayers = Math.floor((freeMb - overheadMb) / costPerLayer);
        maxLayers = Math.max(0, Math.min(maxLayers, totalLayers));
        
        if (n > maxLayers) {
            n = maxLayers;
            gpuSlider.value = n;
        }
        // Calculate limit for C given this n
        if (n > 0) {
            const remainingForKv = freeMb - overheadMb - (n * perLayerMb);
            if (remainingForKv > 0) {
                maxCtx = Math.floor(remainingForKv / (n * kvPerTokenPerLayerMb));
            } else {
                maxCtx = ctxMin;
            }
            maxCtx = Math.max(ctxMin, Math.min(maxCtx, ctxMax));
        }
        
    } else {
        // Initialization: Clamp n first, then clamp C
        const costPerLayer = perLayerMb + (C * kvPerTokenPerLayerMb);
        maxLayers = Math.floor((freeMb - overheadMb) / costPerLayer);
        maxLayers = Math.max(0, Math.min(maxLayers, totalLayers));
        if (n > maxLayers) {
            n = maxLayers;
            gpuSlider.value = n;
        }
        
        if (n > 0) {
            const remainingForKv = freeMb - overheadMb - (n * perLayerMb);
            if (remainingForKv > 0) {
                maxCtx = Math.floor(remainingForKv / (n * kvPerTokenPerLayerMb));
            } else {
                maxCtx = ctxMin;
            }
            maxCtx = Math.max(ctxMin, Math.min(maxCtx, ctxMax));
        }
        if (C > maxCtx) {
            C = maxCtx;
            ctxSlider.value = C;
        }
    }

    // Position limit markers on tracks (%)
    if (gpuMarker) {
        const gpuPct = (maxLayers / totalLayers) * 100;
        gpuMarker.style.left = gpuPct + '%';
        gpuMarker.style.display = 'block';
    }

    if (ctxMarker) {
        const ctxPct = ((maxCtx - ctxMin) / (ctxMax - ctxMin)) * 100;
        ctxMarker.style.left = ctxPct + '%';
        ctxMarker.style.display = 'block';
    }

    updateLabels(n, C);
}

function updateLabels(n, C) {
    document.getElementById('layer-info').textContent = `${n} / ${vramData ? vramData.total_layers : 65} layers on GPU`;
    document.getElementById('ctx-info').textContent = `${C.toLocaleString()} tokens`;
}

function closeSettings() {
    document.getElementById('settings-modal').style.display = 'none';
}

function closeSettingsIfOutside(event) {
    if (event.target.id === 'settings-modal') closeSettings();
}

function saveSettings() {
    closeSettings();
    const nLayers = parseInt(document.getElementById('gpu-layer-slider').value);
    const nCtx = parseInt(document.getElementById('ctx-slider').value);
    if (window.pyBridge) {
        window.pyBridge.update_model_settings(nLayers, nCtx);
    }
}

