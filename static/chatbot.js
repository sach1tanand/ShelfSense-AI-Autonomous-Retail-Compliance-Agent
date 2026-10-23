(() => {
  const root = document.getElementById('shelf-chat');
  if (!root) return;

  const launcher = document.getElementById('shelf-chat-launcher');
  const panel = document.getElementById('shelf-chat-panel');
  const closeButton = document.getElementById('shelf-chat-close');
  const form = document.getElementById('shelf-chat-form');
  const input = document.getElementById('shelf-chat-input');
  const sendButton = document.getElementById('shelf-chat-send');
  const messagesElement = document.getElementById('shelf-chat-messages');
  const loading = document.getElementById('shelf-chat-loading');
  let reportId = root.dataset.reportId || null;
  let configured = false;
  let history = [];

  const storageKey = () => `shelfsense-chat-${reportId || 'general'}`;
  const escapeHtml = value => value.replace(/[&<>\"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', "'": '&#39;'
  })[character]);
  const renderAssistantMarkdown = content => {
    const inline = value => escapeHtml(value)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>');
    const lines = content.replace(/\r/g, '').split('\n');
    const output = [];
    let listType = null;
    const closeList = () => {
      if (listType) output.push(listType === 'ul' ? '</ul>' : '</ol>');
      listType = null;
    };
    for (const line of lines) {
      const trimmed = line.trim();
      const heading = trimmed.match(/^#{1,3}\s+(.+)$/);
      const bullet = trimmed.match(/^(?:[-+*])\s+(.+)$/);
      const numbered = trimmed.match(/^\d+[.)]\s+(.+)$/);
      if (!trimmed) { closeList(); continue; }
      if (heading) {
        closeList();
        output.push(`<strong class="shelf-chat-heading">${inline(heading[1])}</strong>`);
      } else if (bullet || numbered) {
        const type = bullet ? 'ul' : 'ol';
        if (listType !== type) { closeList(); output.push(`<${type}>`); listType = type; }
        output.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      } else {
        closeList();
        output.push(`<p>${inline(trimmed)}</p>`);
      }
    }
    closeList();
    return output.join('');
  };
  const persist = () => {
    try { sessionStorage.setItem(storageKey(), JSON.stringify(history.slice(-16))); } catch (_) { /* optional */ }
  };
  const appendMessage = (role, content, isError = false) => {
    messagesElement.querySelector('.shelf-chat-intro')?.remove();
    const bubble = document.createElement('div');
    bubble.className = `shelf-chat-message ${role}${isError ? ' error' : ''}`;
    if (role === 'assistant' && !isError) bubble.innerHTML = renderAssistantMarkdown(content);
    else bubble.textContent = content;
    messagesElement.appendChild(bubble);
    messagesElement.scrollTop = messagesElement.scrollHeight;
  };
  const renderHistory = () => {
    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey()) || '[]');
      if (!Array.isArray(saved)) return;
      history = saved.filter(item => item && ['user', 'assistant'].includes(item.role) && typeof item.content === 'string').slice(-16);
      history.forEach(item => appendMessage(item.role, item.content));
    } catch (_) { history = []; }
  };
  const setOpen = open => {
    panel.hidden = !open;
    launcher.setAttribute('aria-expanded', String(open));
    if (open) {
      if (!messagesElement.querySelector('.shelf-chat-message') && !configured) {
        appendMessage('assistant', 'AI chat is not configured. Set SHELFSENSE_CHAT_API_KEY on the server and restart ShelfSense AI.', true);
      } else if (!messagesElement.querySelector('.shelf-chat-message') && configured) {
        appendMessage('assistant', reportId ? 'I can answer questions using the selected audit report. Try “Summarize this audit.”' : 'I can explain ShelfSense AI features and workflow. Run an audit to ask about its results.');
      }
      input.focus();
    }
  };

  const refreshStatus = async () => {
    try {
      const response = await fetch('/api/chat', { headers: { Accept: 'application/json' } });
      const data = await response.json();
      configured = response.ok && Boolean(data.configured);
      if (!reportId && data.report_id) reportId = data.report_id;
      renderHistory();
    } catch (_) {
      appendMessage('assistant', 'Chat service is temporarily unavailable. Please try again later.', true);
    }
  };

  const sendMessage = async text => {
    const message = text.trim();
    if (!message || sendButton.disabled) return;
    const previousHistory = history.slice(-8);
    appendMessage('user', message);
    history.push({ role: 'user', content: message });
    persist();
    input.value = '';
    sendButton.disabled = true;
    loading.hidden = false;
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ message, report_id: reportId, history: previousHistory })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'The chatbot could not answer right now.');
      if (!reportId && data.report_id) reportId = data.report_id;
      appendMessage('assistant', data.reply);
      history.push({ role: 'assistant', content: data.reply });
      persist();
    } catch (error) {
      appendMessage('assistant', error.message || 'The chatbot could not answer right now.', true);
    } finally {
      loading.hidden = true;
      sendButton.disabled = false;
      input.focus();
    }
  };

  launcher.addEventListener('click', () => setOpen(panel.hidden));
  closeButton.addEventListener('click', () => setOpen(false));
  form.addEventListener('submit', event => { event.preventDefault(); sendMessage(input.value); });
  document.getElementById('shelf-chat-summarize').addEventListener('click', () => sendMessage('Summarize this audit.'));
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
  });
  refreshStatus();
})();
