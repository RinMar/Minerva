// Use window.pyBridge if available, otherwise try window.parent.pyBridge
function getBridge() {
    return window.pyBridge || (window.parent && window.parent.pyBridge); // It may be injected by index.html script
}

function openSettings() {
    // Tell parent window to show settings modal
    if(window.parent && window.parent.showSettings) {
        window.parent.showSettings();
    } else if (getBridge()) {
        getBridge().handle_settings_click();
    }
}

// ── Chat Functions Exported to Parent ──
window.appendMessage = function(role, text) {
    var container = document.getElementById('chat-messages');
    var div = document.createElement('div');
    div.className = 'msg ' + role;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
};

window.clearChat = function() {
    var container = document.getElementById('chat-messages');
    container.innerHTML = '<div class="msg system">Connected to Minerva. Your knowledge graph is loading…</div>';
};

var currentMessageDiv = null;
var currentTextDiv = null;
var thinkingIcon = null;

window.startAssistantMessage = function() {
    var container = document.getElementById('chat-messages');
    currentMessageDiv = document.createElement('div');
    currentMessageDiv.className = 'msg assistant';
    
    thinkingIcon = document.createElement('div');
    thinkingIcon.className = 'thinking-icon';
    thinkingIcon.innerHTML = '🤔 <i>Thinking...</i>';
    thinkingIcon.style.display = 'inline-flex';
    thinkingIcon.style.color = '#a78bfa';
    thinkingIcon.style.fontSize = '12px';
    thinkingIcon.style.marginBottom = '6px';
    currentMessageDiv.appendChild(thinkingIcon);
    
    currentTextDiv = document.createElement('div');
    currentMessageDiv.appendChild(currentTextDiv);
    
    container.appendChild(currentMessageDiv);
    container.scrollTop = container.scrollHeight;
};

window.appendAssistantToken = function(token) {
    if (currentTextDiv) {
        currentTextDiv.appendChild(document.createTextNode(token));
        var container = document.getElementById('chat-messages');
        container.scrollTop = container.scrollHeight;
    }
};

window.setAssistantState = function(state) {
    if (state === 'think_start') {
        if (thinkingIcon) {
            thinkingIcon.innerHTML = '🤔 <i>Thinking...</i>';
            thinkingIcon.style.display = 'inline-flex';
        }
    } else if (state.startsWith('action_start_')) {
        if (thinkingIcon) {
            thinkingIcon.innerHTML = '⚙️ <i>Calling tools...</i>';
            thinkingIcon.style.display = 'inline-flex';
        }
    } else if (state === 'think_end' || state === 'action_end' || state === 'done') {
        if (thinkingIcon) {
            thinkingIcon.style.display = 'none';
        }
        if (state === 'done') {
            currentMessageDiv = null;
            currentTextDiv = null;
            thinkingIcon = null;
        }
    }
    var container = document.getElementById('chat-messages');
    container.scrollTop = container.scrollHeight;
};

var modelReady = false;

window.setModelReady = function(ready) {
    modelReady = ready;
    if (ready) {
        document.getElementById('send-btn').classList.remove('disabled');
    } else {
        document.getElementById('send-btn').classList.add('disabled');
    }
};

window.showModelLoader = function(text) {
    window.setModelReady(false);
    const loader = document.getElementById('model-loader');
    const status = document.getElementById('loader-status');
    if (loader && status) {
        if (text) status.textContent = text;
        loader.style.display = 'flex';
    }
};

window.hideModelLoader = function() {
    window.setModelReady(true);
    const loader = document.getElementById('model-loader');
    if (loader) {
        loader.style.display = 'none';
    }
};

// ── Input Handling ──
function sendMessage() {
    if (!modelReady) return; // Block sending while models load

    var input = document.getElementById('chat-input');
    var text = input.value.trim();
    if (!text) return;

    window.appendMessage('user', text);
    input.value = '';
    input.style.height = 'auto'; // reset height

    let bridge = getBridge();
    if (bridge) {
        bridge.send_message(text);
    }
}

var chatInput = document.getElementById('chat-input');
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

document.getElementById('send-btn').addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevents adding a new line
        sendMessage();
    }
});

// ── User Dropdown ──
function toggleProfileDropdown() {
    const dropdown = document.getElementById('profile-dropdown');
    const isVisible = dropdown.style.display === 'flex';
    
    if (!isVisible) {
        let bridge = getBridge();
        if(!bridge) return;
        
        bridge.get_user_list(function(users) {
            const list = document.getElementById('user-list');
            list.innerHTML = '';
            users.forEach(u => {
                const item = document.createElement('div');
                item.className = 'dropdown-item';
                item.id = 'user-item-' + u;
                
                const info = document.createElement('div');
                info.className = 'user-info';
                info.innerHTML = '👤 <span class="user-name-text">' + u + '</span>';
                item.appendChild(info);

                const edit = document.createElement('div');
                edit.className = 'edit-icon';
                edit.innerHTML = '✏️';
                edit.title = 'Rename Profile';
                edit.onclick = (e) => startRename(e, u);
                item.appendChild(edit);

                item.onclick = function(e) {
                    if (e.target.closest('.edit-icon') || e.target.closest('.rename-input')) return;
                    getBridge().switch_profile(u);
                    toggleProfileDropdown();
                };
                list.appendChild(item);
            });
            dropdown.style.display = 'flex';
        });
    } else {
        dropdown.style.display = 'none';
        document.getElementById('add-profile-input-area').style.display = 'none';
    }
}

function startRename(event, oldName) {
    event.stopPropagation();
    const item = document.getElementById('user-item-' + oldName);
    const info = item.querySelector('.user-info');
    const originalHTML = info.innerHTML;
    
    info.innerHTML = '👤 ';
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'rename-input';
    input.value = oldName;
    
    input.onclick = (e) => e.stopPropagation();
    input.onkeydown = (e) => {
        if (e.key === 'Enter') {
            const newName = input.value.trim();
            if (newName && newName !== oldName) {
                getBridge().rename_user(oldName, newName);
                info.innerHTML = '👤 <span class="user-name-text">' + newName + '</span>';
                item.id = 'user-item-' + newName;
            } else {
                info.innerHTML = originalHTML;
            }
        } else if (e.key === 'Escape') {
            info.innerHTML = originalHTML;
        }
    };
    
    info.appendChild(input);
    input.focus();
    input.select();
}

function showAddProfileInput(event) {
    event.stopPropagation();
    document.getElementById('add-profile-input-area').style.display = 'block';
    document.getElementById('new-profile-name').focus();
}

function handleNewProfileKeyDown(event) {
    if (event.key === 'Enter') {
        const name = event.target.value.trim();
        if (name) {
            let bridge = getBridge();
            if (bridge) bridge.create_profile(name);
            toggleProfileDropdown();
        }
    } else if (event.key === 'Escape') {
        document.getElementById('add-profile-input-area').style.display = 'none';
    }
}

window.addEventListener('click', function(e) {
    const dropdown = document.getElementById('profile-dropdown');
    const userBtn = document.getElementById('user-btn');
    if (dropdown && dropdown.style.display === 'flex' && !dropdown.contains(e.target) && e.target !== userBtn) {
        dropdown.style.display = 'none';
        document.getElementById('add-profile-input-area').style.display = 'none';
    }
});
