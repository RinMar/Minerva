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
let selectedPerfMode = 'low';

function showSettings() {
    const modal = document.getElementById('settings-modal');
    if (window.pyBridge) {
        window.pyBridge.get_performance_mode(function(mode) {
            selectedPerfMode = mode;
            updatePerfUI();
            modal.style.display = 'flex';
        });
    } else {
        updatePerfUI();
        modal.style.display = 'flex';
    }
}

function closeSettings() {
    document.getElementById('settings-modal').style.display = 'none';
}

function closeSettingsIfOutside(event) {
    if (event.target.id === 'settings-modal') closeSettings();
}

function selectPerf(mode) {
    selectedPerfMode = mode;
    updatePerfUI();
}

function updatePerfUI() {
    document.querySelectorAll('.perf-opt').forEach(opt => opt.classList.remove('active'));
    document.getElementById('perf-' + selectedPerfMode).classList.add('active');
}

function saveSettings() {
    closeSettings();
    if (window.pyBridge) {
        window.pyBridge.update_performance_mode(selectedPerfMode);
    }
}
