// =============================================================================
// AUTH NAV
// =============================================================================
async function pulseNovaInitAuthNav() {
  try {
    const res  = await fetch('/me');
    const data = await res.json();
    const authed = !!data?.authenticated;

    // FIX: Set isAuthenticated directly on the app instance BEFORE any
    // downstream code reads it. Previously this was only set on window.app
    // inside a conditional, causing a race where loadUserHistory() would
    // see undefined and silently bail out.
    if (window.app) {
      window.app.isAuthenticated = authed;
      // Also cache the profile so logout / dashboard can read it
      window.app._authProfile = data?.profile || null;
      window.app._authPrefs   = data?.prefs   || null;
    }

    // 1. Top nav bar & mobile menu
    const signin       = document.getElementById('nav-signin');
    const acct         = document.getElementById('nav-account');
    const mobileSignin = document.getElementById('mobile-nav-signin');
    const mobileAcct   = document.getElementById('mobile-nav-account');

    if (signin)       signin.style.display       = authed ? 'none'        : 'inline-flex';
    if (acct)         acct.style.display         = authed ? 'inline-flex' : 'none';
    if (mobileSignin) mobileSignin.style.display = authed ? 'none'        : 'flex';
    if (mobileAcct)   mobileAcct.style.display   = authed ? 'flex'        : 'none';

    // 2. Homepage hero buttons
    const btnSignin      = document.getElementById('btn-hero-signin');
    const btnGuest       = document.getElementById('btn-hero-guest');
    const btnTriageAuth  = document.getElementById('btn-hero-triage-auth');
    const btnDashboard   = document.getElementById('btn-hero-dashboard');

    if (authed) {
      if (btnSignin)     btnSignin.classList.add('hidden');
      if (btnGuest)      btnGuest.classList.add('hidden');
      if (btnTriageAuth) { btnTriageAuth.classList.remove('hidden'); btnTriageAuth.classList.add('flex'); }
      if (btnDashboard)  { btnDashboard.classList.remove('hidden');  btnDashboard.classList.add('flex');  }

      // 3. Populate inline dashboard
      const dashName       = document.getElementById('dash-name');
      const dashEmail      = document.getElementById('dash-email');
      const dashStatus     = document.getElementById('dash-status');
      const dashLoginBtn   = document.getElementById('dash-login-btn');
      const dashLogoutBtn  = document.getElementById('dash-logout-btn');
      const dashGuestBanner = document.getElementById('dash-guest-banner');
      const histAuthChip   = document.getElementById('history-auth-chip');
      const prefStore      = document.getElementById('pref-store-history');
      const prefRetention  = document.getElementById('pref-retention');

      if (dashName)        dashName.textContent  = data.profile?.name  || data.profile?.sub || 'User';
      if (dashEmail)       dashEmail.textContent = data.profile?.email || '';
      if (dashStatus)      { dashStatus.textContent = 'Signed In'; dashStatus.className = 'chip chip-green'; }
      if (dashLoginBtn)    dashLoginBtn.classList.add('hidden');
      if (dashLogoutBtn)   dashLogoutBtn.classList.remove('hidden');
      if (dashGuestBanner) dashGuestBanner.classList.add('hidden');
      if (histAuthChip)    { histAuthChip.textContent = 'Signed In'; histAuthChip.className = 'chip chip-green text-[10px]'; }

      // Pre-fill preference toggles from server prefs
      if (prefStore     && data.prefs) prefStore.checked    = !!data.prefs.consent_store_history;
      if (prefRetention && data.prefs) prefRetention.value  = data.prefs.data_retention_days ?? 30;

    } else {
      if (btnSignin)     { btnSignin.classList.remove('hidden');     btnSignin.classList.add('flex'); }
      if (btnGuest)      { btnGuest.classList.remove('hidden');      btnGuest.classList.add('flex');  }
      if (btnTriageAuth) { btnTriageAuth.classList.add('hidden');    btnTriageAuth.classList.remove('flex'); }
      if (btnDashboard)  { btnDashboard.classList.add('hidden');     btnDashboard.classList.remove('flex');  }

      // Show guest banner & history notice
      const dashGuestBanner = document.getElementById('dash-guest-banner');
      const histGuestMsg    = document.getElementById('history-guest-msg');
      if (dashGuestBanner) dashGuestBanner.classList.remove('hidden');
      if (histGuestMsg)    histGuestMsg.classList.remove('hidden');
    }
  } catch (e) {
    console.error('Auth check failed:', e);
  }
}

// =============================================================================
// IOS TTS UNLOCK
// =============================================================================
let _iosTTSUnlocked = false;
function _unlockIOSTTSGlobal() {
  if (_iosTTSUnlocked) return;
  _iosTTSUnlocked = true;
  try {
    const u = new SpeechSynthesisUtterance('');
    u.volume = 0;
    window.speechSynthesis?.speak(u);
    window.speechSynthesis?.cancel();
    window.speechSynthesis?.getVoices();
  } catch (_) {}
}
document.addEventListener('touchstart', _unlockIOSTTSGlobal, { once: true, passive: true });
document.addEventListener('click',      _unlockIOSTTSGlobal, { once: true });
if (window.speechSynthesis) {
  window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
}

// =============================================================================
// PULSENOVA APP
// =============================================================================
class PulseNovaApp {
  constructor() {
    this.greetings = {
      'en-US': "Hello. I'm PulseNova. Please describe your symptoms in detail, or send me a photo if relevant.",
      'en-GB': "Hello. I'm PulseNova. Please describe your symptoms in detail, or send me a photo if relevant.",
      'en-AU': "Hello. I'm PulseNova. Please describe your symptoms in detail, or send me a photo if relevant.",
      'en-IN': "Hello. I'm PulseNova. Please describe your symptoms in detail, or send me a photo if relevant.",
      'fr-FR': "Bonjour. Je suis PulseNova. Veuillez décrire vos symptômes en détail, ou envoyez-moi une photo si pertinent.",
      'de-DE': "Hallo. Ich bin PulseNova. Bitte beschreiben Sie Ihre Symptome im Detail oder senden Sie mir ein Foto.",
      'es-US': "Hola. Soy PulseNova. Por favor, describa sus síntomas en detalle, o envíeme una foto si es relevante.",
      'it-IT': "Ciao. Sono PulseNova. Per favore, descrivi i tuoi sintomi in dettaglio, o inviami una foto se pertinente.",
      'pt-BR': "Olá. Eu sou o PulseNova. Por favor, descreva seus sintomas em detalhes, ou me envie uma foto se for relevante.",
      'hi-IN': "नमस्ते। मैं पल्स-नोवा हूँ। कृपया अपने लक्षणों का विस्तार से वर्णन करें, या यदि प्रासंगिक हो तो मुझे एक फोटो भेजें।",
    };

    this.selectedLanguage = localStorage.getItem('pulseNova_lang')      || 'en-US';
    this.xrayLanguage     = localStorage.getItem('pulseNova_xray_lang') || this.selectedLanguage;
    this.labLanguage      = localStorage.getItem('pulseNova_lab_lang')  || this.selectedLanguage;
    this.speakOutput      = localStorage.getItem('pulseNova_speak')     === '1';

    this.messages  = [{ role: 'assistant', text: this.greetings[this.selectedLanguage] || this.greetings['en-US'] }];
    this.chatImage = null;
    this.chatBusy  = false;
    this.xrayImage = null;
    this.labImage  = null;
    this.xrayHistory   = [];
    this.labHistory    = [];
    this.triageHistory = [];

    // FIX: currentChatId is used as the stable identifier sent to the server
    // so the server can upsert the same session row instead of creating
    // duplicates on every message.
    this.currentChatId = this._newChatId();

    // FIX: isAuthenticated starts as false (not undefined).
    // pulseNovaInitAuthNav() sets the real value after /me resolves.
    this.isAuthenticated = false;
    this._authProfile    = null;
    this._authPrefs      = null;

    this.supportedLanguages = [
      { code: 'en-US', label: 'English (US)'        },
      { code: 'en-GB', label: 'English (UK)'         },
      { code: 'en-AU', label: 'English (Australia)'  },
      { code: 'en-IN', label: 'English (India)'      },
      { code: 'fr-FR', label: 'Français (France)'    },
      { code: 'de-DE', label: 'Deutsch (Germany)'    },
      { code: 'es-US', label: 'Español (US)'         },
      { code: 'it-IT', label: 'Italiano (Italy)'     },
      { code: 'pt-BR', label: 'Português (Brazil)'   },
      { code: 'hi-IN', label: 'हिन्दी (India)'       },
    ];

    // Voice
    this.voiceState         = 'idle';
    this.voiceOverlayOpen   = false;
    this.voiceLoopActive    = false;
    this.recognition        = null;
    this.currentUtterance   = null;
    this.voiceBarsAnimId    = null;
    this._pendingTranscript = '';
    this._iosResumeTimer    = null;
    this.showEnglishPanel   = true;

    // Vitals
    this.vitals = {
      stream: null, video: null, canvas: null, ctx: null,
      processCanvas: document.createElement('canvas'), processCtx: null, animId: null,
      redValues: [], smoothed: [], bpmHistory: [],
      stableBpmCount: 0, lastPeakTime: 0,
      isMonitoring: false, timeoutId: null,
      torchTrack: null, torchOn: false,
      signalQuality: 0, context: 'resting',
      lastStableBpm: null, summaryReady: false,
    };
    this.vitals.processCanvas.width  = 20;
    this.vitals.processCanvas.height = 20;
    this.vitals.processCtx = this.vitals.processCanvas.getContext('2d', { willReadFrequently: true });
    this.processVitalsFrame = this.processVitalsFrame.bind(this);

    // Care
    this.googleMapsReady   = false;
    this.googleMapsLoading = null;
    this.gmap              = null;
    this.placesService     = null;
    this.geocoder          = null;
    this.providers         = [];
    this.care              = { userLat: null, userLon: null, centerLabel: null };

    // Init
    this.initSpeech();
    this.initCareSearch();
    this.initDropzones();
    this.initLanguageControls();
    this.initChatInputAutoResize();
    this.syncSpeechToggleIcon();

    this.renderChat();
    this.renderProviders();
    this.setVitalsContext('resting');
    this.initRx();
    // FIX: Do NOT call resetVitalsSummary() in the constructor.
    // The summary card starts hidden in the HTML and should only
    // become visible once the user actually starts a reading.
    // Calling it here was causing the card to flash visible on page load.
  }

  // ---------------------------------------------------------------------------
  // HELPERS
  // ---------------------------------------------------------------------------
  _newChatId() {
    // Generates a stable string ID suitable for both local use and the DB
    return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  }

  /* -------------------- TOAST -------------------- */
  toast(msg, icon = 'info') {
    const t   = document.getElementById('toast');
    const txt = document.getElementById('toast-text');
    const ico = document.getElementById('toast-ico');
    if (!t || !txt || !ico) return;
    txt.textContent = msg;
    ico.setAttribute('data-lucide', icon);
    t.classList.add('show');
    if (window.lucide) lucide.createIcons();
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => t.classList.remove('show'), 2400);
  }

  /* -------------------- NAV -------------------- */
  navigate(pageId) {
    // NEW: Intercept 'dashboard' clicks and route to the dedicated accounts page
    if (pageId === 'dashboard') {
      window.location.href = '/account';
      return;
    }

    if (pageId !== 'vitals' && this.vitals.isMonitoring) this.stopVitals(false);
    document.querySelectorAll('.page-section').forEach(el => el.classList.add('hidden'));
    const page = document.getElementById(`page-${pageId}`);
    if (page) page.classList.remove('hidden');

    document.querySelectorAll('.nav-btn').forEach(el => {
      el.classList.toggle('active', el.dataset.page === pageId);
    });
    document.getElementById('mobile-menu')?.classList.add('hidden');

    if (pageId !== 'triage')  { this.closeVoiceMode(); try { window.speechSynthesis.cancel(); } catch (_) {} }
    if (pageId === 'triage')  setTimeout(() => document.getElementById('chat-input')?.focus(), 100);
    if (pageId === 'xray' || pageId === 'labs') setTimeout(() => this._initVisionLangSelects(), 50);
  }

  toggleMobileMenu() { document.getElementById('mobile-menu')?.classList.toggle('hidden'); }
  /* -------------------- LOGOUT -------------------- */
  // FIX: logout() method was missing from the class entirely.
  // The dashboard "Sign out" button called app.logout() which threw
  // TypeError: app.logout is not a function and silently did nothing.
  async logout() {
    try {
      await fetch('/auth/logout', { method: 'POST' });
    } catch (_) {}
    this.isAuthenticated = false;
    this._authProfile    = null;
    this._authPrefs      = null;
    window.location.href = '/';
  }

  /* -------------------- SAVE PREFS -------------------- */
  async savePrefs() {
    const consent   = document.getElementById('pref-store-history')?.checked ?? false;
    const retention = parseInt(document.getElementById('pref-retention')?.value || '30', 10);
    try {
      const res = await fetch('/prefs', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          consent_store_history: consent,
          data_retention_days:   retention,
        }),
      });
      if (res.ok) this.toast('Preferences saved', 'check');
      else        this.toast('Failed to save preferences', 'alert-triangle');
    } catch (_) {
      this.toast('Network error saving preferences', 'alert-triangle');
    }
  }

  /* -------------------- REFRESH DASHBOARD -------------------- */
  async refreshDashboard() {
    await pulseNovaInitAuthNav();
    if (this.isAuthenticated) await this.loadUserHistory();
    this.toast('Dashboard refreshed', 'refresh-cw');
  }

  /* -------------------- LANGUAGE -------------------- */
  initLanguageControls() {
    const triageHeader = document.querySelector('#page-triage .border-b');
    if (triageHeader && !document.getElementById('triage-lang-select')) {
      const right = triageHeader.querySelector('.flex.items-center.gap-3');
      if (right) {
        const wrap = document.createElement('div');
        wrap.className = 'hidden sm:block';
        wrap.innerHTML = `<select id="triage-lang-select" class="text-xs border border-slate-200 rounded-lg px-2 py-1 bg-white text-slate-600 outline-none">${this.supportedLanguages.map(l => `<option value="${l.code}">${l.label}</option>`).join('')}</select>`;
        right.prepend(wrap);
        const sel = wrap.querySelector('select');
        sel.value = this.selectedLanguage;
        sel.addEventListener('change', e => this.setLanguage(e.target.value));
      }
    }

    const mobileMenu = document.getElementById('mobile-menu');
    if (mobileMenu && !document.getElementById('mobile-lang-select')) {
      const wrap = document.createElement('div');
      wrap.className = 'px-4 py-3 border-b border-slate-100 bg-slate-50 flex items-center justify-between';
      wrap.innerHTML = `<span class="text-sm font-medium text-slate-600">Language</span><select id="mobile-lang-select" class="text-sm border border-slate-200 rounded-lg px-2 py-1 bg-white text-slate-600 outline-none">${this.supportedLanguages.map(l => `<option value="${l.code}">${l.label}</option>`).join('')}</select>`;
      mobileMenu.prepend(wrap);
      const sel = wrap.querySelector('select');
      sel.value = this.selectedLanguage;
      sel.addEventListener('change', e => { this.setLanguage(e.target.value); this.toggleMobileMenu(); });
    }

    const subtitle = document.getElementById('voice-subtitle');
    if (subtitle && !document.getElementById('voice-lang-wrap')) {
      const wrap = document.createElement('div');
      wrap.id        = 'voice-lang-wrap';
      wrap.className = 'mt-3';
      wrap.innerHTML = `<label class="text-xs text-slate-400 mr-2">Language</label><select id="voice-lang-select" class="text-xs border border-white/10 bg-white/5 text-white rounded-lg px-2 py-1 outline-none">${this.supportedLanguages.map(l => `<option value="${l.code}">${l.label}</option>`).join('')}</select>`;
      subtitle.insertAdjacentElement('afterend', wrap);
      const sel = wrap.querySelector('select');
      sel.value = this.selectedLanguage;
      sel.addEventListener('change', e => this.setLanguage(e.target.value));
    }
  }

  setGlobalLanguage(code) { this.setLanguage(code); }

  setLanguage(code) {
    this.selectedLanguage = code;
    localStorage.setItem('pulseNova_lang', code);
    if (this.recognition) this.recognition.lang = code;
    ['triage-lang-select', 'voice-lang-select', 'mobile-lang-select', 'global-lang-select', 'mobile-lang-select'].forEach(id => {
      const el = document.getElementById(id);
      if (el && el.value !== code) el.value = code;
    });
    this.resetChat();
    this.toast('Language updated', 'globe');
  }

  /* -------------------- CHAT UX -------------------- */
  initChatInputAutoResize() {
    const input = document.getElementById('chat-input');
    if (!input) return;
    const resize = () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 128) + 'px'; };
    input.addEventListener('input', resize);
    resize();
  }

  focusChatInput() {
    const input = document.getElementById('chat-input');
    if (!input) return;
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }

  /* -------------------- TTS TOGGLE -------------------- */
  toggleSpeechOutput() {
    this.speakOutput = !this.speakOutput;
    localStorage.setItem('pulseNova_speak', this.speakOutput ? '1' : '0');
    this.syncSpeechToggleIcon();
    this.toast(this.speakOutput ? 'Spoken replies: ON' : 'Spoken replies: OFF', this.speakOutput ? 'volume-2' : 'volume-x');
    if (!this.speakOutput) { try { window.speechSynthesis.cancel(); } catch (_) {} }
  }

  syncSpeechToggleIcon() {
    const btn = document.getElementById('btn-speech-toggle');
    if (!btn) return;
    btn.innerHTML = this.speakOutput
      ? `<i data-lucide="volume-2" class="w-4 h-4"></i>`
      : `<i data-lucide="volume-x"  class="w-4 h-4"></i>`;
    if (window.lucide) lucide.createIcons();
  }

  speakText(text) {
    if (!this.speakOutput) return;
    try { window.speechSynthesis.cancel(); } catch (_) {}
    const cleaned = (text || '').replace(/\*+/g, '').replace(/#{1,3}\s*/g, '').replace(/\[TRIGGER_[^\]]+\]/g, '').trim();
    if (!cleaned) return;
    const utter = new SpeechSynthesisUtterance(cleaned.slice(0, 1200));
    utter.lang = this.selectedLanguage || 'en-US';
    try { window.speechSynthesis.speak(utter); } catch (_) {}
  }

  /* -------------------- VOICE UI -------------------- */
  toggleEnglishPanel() {
    this.showEnglishPanel = !this.showEnglishPanel;
    document.getElementById('voice-en-wrap')?.classList.toggle('hidden', !this.showEnglishPanel);
  }

  _showTranslationPanel({ detectedLabel, orig, en }) {
    const panel = document.getElementById('voice-translation-panel');
    const d     = document.getElementById('voice-lang-detected');
    const o     = document.getElementById('voice-orig');
    const e     = document.getElementById('voice-en');
    const wrap  = document.getElementById('voice-en-wrap');
    if (panel) panel.classList.remove('hidden');
    if (d)     d.textContent = detectedLabel || '--';
    if (o)     o.textContent = orig          || '';
    if (e)     e.textContent = en            || '';
    if (wrap)  wrap.classList.toggle('hidden', !this.showEnglishPanel);
  }

  setVoiceState(state) {
    this.voiceState = state;
    const orb      = document.getElementById('voice-orb');
    const title    = document.getElementById('voice-title');
    const subtitle = document.getElementById('voice-subtitle');
    const dot      = document.getElementById('voice-indicator-dot');
    const btn      = document.getElementById('voice-main-btn');
    const bars     = document.getElementById('voice-bars');
    const preview  = document.getElementById('voice-transcript-preview');

    if (orb)  { orb.className = 'voice-orb'; orb.classList.add(`state-${state}`); }
    if (bars) { bars.className = 'voice-bars'; if (state === 'speaking') bars.classList.add('speaking'); if (state === 'thinking') bars.classList.add('thinking'); }

    switch (state) {
      case 'idle':
        if (title)   title.textContent   = 'Voice Mode';
        if (subtitle) subtitle.innerHTML = 'Tap <strong class="text-white">Start Listening</strong> to begin.';
        if (dot)     dot.className       = 'w-2 h-2 rounded-full bg-slate-600 inline-block transition-colors duration-300';
        if (btn)     { btn.innerHTML = '<i data-lucide="mic" class="w-4 h-4"></i><span id="voice-btn-label">Start Listening</span>'; btn.className = 'w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3.5 rounded-2xl flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20 transition-all text-sm'; }
        if (preview) preview.textContent = '';
        this.voiceLoopActive = false; this.stopBarAnimation(); this.resetBars(); break;
      case 'listening':
        if (title)   title.textContent   = 'Listening…';
        if (subtitle) subtitle.innerHTML = "Speak naturally. I'm listening.";
        if (dot)     dot.className       = 'w-2 h-2 rounded-full bg-blue-400 inline-block transition-colors duration-300 status-pulse';
        if (btn)     { btn.innerHTML = '<i data-lucide="mic-off" class="w-4 h-4"></i><span>Stop</span>'; btn.className = 'w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-3.5 rounded-2xl flex items-center justify-center gap-2 shadow-lg transition-all text-sm'; }
        if (preview) preview.textContent = '';
        this.startBarAnimation('listening'); break;
      case 'thinking':
        if (title)   title.textContent   = 'Thinking…';
        if (subtitle) subtitle.innerHTML = 'PulseNova is processing your message.';
        if (dot)     dot.className       = 'w-2 h-2 rounded-full bg-purple-400 inline-block';
        if (btn)     { btn.innerHTML = '<i data-lucide="loader" class="w-4 h-4 animate-spin"></i><span>Processing…</span>'; btn.className = 'w-full bg-slate-700 text-white font-semibold py-3.5 rounded-2xl flex items-center justify-center gap-2 opacity-70 cursor-not-allowed text-sm'; }
        this.stopBarAnimation(); break;
      case 'speaking':
        if (title)   title.textContent   = 'Speaking…';
        if (subtitle) subtitle.innerHTML = 'Speak to interrupt.';
        if (dot)     dot.className       = 'w-2 h-2 rounded-full bg-emerald-400 inline-block status-pulse';
        if (btn)     { btn.innerHTML = '<i data-lucide="mic" class="w-4 h-4"></i><span>Interrupt</span>'; btn.className = 'w-full bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-3.5 rounded-2xl flex items-center justify-center gap-2 shadow-lg transition-all text-sm'; }
        this.startBarAnimation('speaking'); break;
    }
    this.initLanguageControls();
    if (window.lucide) lucide.createIcons();
  }

  startBarAnimation(mode) {
    this.stopBarAnimation();
    const ids   = ['vb1', 'vb2', 'vb3', 'vb4', 'vb5', 'vb6', 'vb7'];
    let   frame = 0;
    const animate = () => {
      frame++;
      ids.forEach((id, i) => {
        const el = document.getElementById(id);
        if (!el) return;
        const h = mode === 'listening'
          ? 6 + Math.abs(Math.sin(frame * 0.12 + i * 0.9))  * 30
          : mode === 'speaking'
            ? 6 + Math.abs(Math.sin(frame * 0.18 + i * 1.2)) * 34
            : 6;
        el.style.height = h + 'px';
      });
      this.voiceBarsAnimId = requestAnimationFrame(animate);
    };
    this.voiceBarsAnimId = requestAnimationFrame(animate);
  }
  stopBarAnimation() { if (this.voiceBarsAnimId) { cancelAnimationFrame(this.voiceBarsAnimId); this.voiceBarsAnimId = null; } }
  resetBars()        { ['vb1','vb2','vb3','vb4','vb5','vb6','vb7'].forEach(id => { const el = document.getElementById(id); if (el) el.style.height = '6px'; }); }

  /* -------------------- SPEECH -------------------- */
  initSpeech() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    this.recognition               = new SR();
    this.recognition.continuous     = false;
    this.recognition.interimResults = true;
    this.recognition.lang           = this.selectedLanguage;

    this.recognition.onstart  = () => {
      try { window.speechSynthesis.cancel(); } catch (_) {}
      this.currentUtterance = null;
      this.setVoiceState('listening');
    };
    this.recognition.onresult = (event) => {
      let interim = '', final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) final  += event.results[i][0].transcript;
        else                          interim += event.results[i][0].transcript;
      }
      const p = document.getElementById('voice-transcript-preview');
      if (p) p.textContent = `"${final || interim}"`;
      if (final.trim()) this._pendingTranscript = final.trim();
    };
    this.recognition.onend    = () => {
      if (!this.voiceOverlayOpen) return;
      const transcript        = this._pendingTranscript || '';
      this._pendingTranscript = '';
      if (transcript && this.voiceLoopActive) this._voiceSendAndReply(transcript);
      else if (this.voiceLoopActive) {
        this.setVoiceState('listening');
        setTimeout(() => { if (this.voiceState === 'listening' && this.voiceLoopActive) { try { this.recognition.start(); } catch (_) {} } }, 300);
      } else this.setVoiceState('idle');
    };
    this.recognition.onerror  = (e) => {
      if (e.error === 'aborted') return;
      if (this.voiceLoopActive && this.voiceOverlayOpen) {
        setTimeout(() => { if (this.voiceLoopActive && this.voiceOverlayOpen) { this.setVoiceState('listening'); try { this.recognition.start(); } catch (_) {} } }, 600);
      } else this.setVoiceState('idle');
    };
  }

  openVoiceOverlay() {
    if (window.speechSynthesis) {
      const u = new SpeechSynthesisUtterance(''); u.volume = 0;
      window.speechSynthesis.speak(u); window.speechSynthesis.cancel(); window.speechSynthesis.getVoices();
    }
    this.voiceOverlayOpen = true;
    document.getElementById('voice-overlay')?.classList.add('show');
    this.setVoiceState('idle');
    this.initLanguageControls();
    document.getElementById('voice-translation-panel')?.classList.add('hidden');
    if (window.lucide) lucide.createIcons();
  }

  closeVoiceMode() {
    this.voiceLoopActive    = false;
    this.voiceOverlayOpen   = false;
    this._pendingTranscript = '';
    if (this.recognition) { try { this.recognition.abort(); } catch (_) {} }
    try { window.speechSynthesis.cancel(); } catch (_) {}
    this.currentUtterance = null;
    if (this._iosResumeTimer) { clearInterval(this._iosResumeTimer); this._iosResumeTimer = null; }
    this.stopBarAnimation(); this.resetBars();
    document.getElementById('voice-overlay')?.classList.remove('show');
  }

  voicePrimaryAction() {
    if      (this.voiceState === 'idle')      { this.voiceLoopActive = true; this._startListening(); }
    else if (this.voiceState === 'listening') { this.voiceLoopActive = false; try { this.recognition.stop(); } catch (_) {} this.setVoiceState('idle'); }
    else if (this.voiceState === 'speaking')  { try { window.speechSynthesis.cancel(); } catch (_) {} this.setVoiceState('listening'); setTimeout(() => { try { this.recognition.start(); } catch (_) {} }, 200); }
  }

  _startListening() {
    if (!this.recognition) { alert('Speech recognition not supported. Try Chrome.'); return; }
    this.recognition.lang   = this.selectedLanguage;
    this._pendingTranscript = '';
    this.setVoiceState('listening');
    try { this.recognition.start(); } catch (_) { setTimeout(() => { try { this.recognition.start(); } catch (__) {} }, 500); }
  }

  /* -------------------- VOICE FLOW -------------------- */
  async _voiceSendAndReply(text) {
    this.setVoiceState('thinking');
    const p = document.getElementById('voice-transcript-preview');
    if (p) p.textContent = `You said: "${text}"`;
    this._showTranslationPanel({ detectedLabel: 'detecting…', orig: text, en: '' });

    this.messages.push({ role: 'user', text });
    this.renderChat();

    const history = this.messages.filter(m => m.text).slice(0, -1).map(m => ({ role: m.role, text: m.text }));
    const vt      = await this.callVoiceTurn(history, text);

    // FIX: was referencing undefined `cleanResponse` — use `assistantText` consistently
    let assistantText      = (vt && vt.reply_local) ? vt.reply_local : "I didn't get a response.";
    let triggerFirstAid    = false;
    let triggerCareFinder  = false;

    if (assistantText.includes('[TRIGGER_FIRST_AID]'))   { triggerFirstAid   = true; assistantText = assistantText.replace('[TRIGGER_FIRST_AID]',   '').trim(); }
    if (assistantText.includes('[TRIGGER_CARE_FINDER]')) { triggerCareFinder = true; assistantText = assistantText.replace('[TRIGGER_CARE_FINDER]', '').trim(); }

    this.messages.push({ role: 'assistant', text: assistantText });
    if (triggerCareFinder) this.messages.push({ role: 'assistant', type: 'care_finder' });
    this.renderChat();

    // FIX: was `this.speakText(cleanResponse)` — cleanResponse didn't exist here
    this.speakText(assistantText);

    await this._saveCurrentTriageChat();

    if (vt) {
      this._showTranslationPanel({
        detectedLabel: `${vt.source_language_name || vt.source_language || '--'} (${vt.source_language || '--'})`,
        orig: vt.transcript_original || text,
        en:   vt.transcript_en       || '',
      });
    }

    this._voiceSpeak(assistantText);

    if (triggerFirstAid) {
      const guideData = await this.callFirstAidGuide(text);
      if (guideData?.steps) {
        this.messages.push({ role: 'assistant', type: 'first_aid', steps: guideData.steps });
        this.renderChat();
        setTimeout(() => { const c = document.getElementById('chat-messages'); if (c) c.scrollTop = c.scrollHeight; }, 100);
      }
    }
  }

  async callVoiceTurn(history, transcript) {
    try {
      let cleanHistory = (history || []).map(t => ({ role: t.role, text: (t.text || '').trim() })).filter(t => t.text);
      while (cleanHistory.length && cleanHistory[0].role !== 'user') cleanHistory.shift();
      const res = await fetch('/api/voice-turn', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          transcript:             transcript || '',
          detected_lang:          this.selectedLanguage,
          history:                cleanHistory,
          include_english_reply:  true,
          max_tokens:             350,
          temperature:            0.3,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) { console.error(e); return null; }
  }

  _voiceSpeak(text) {
    try { window.speechSynthesis.cancel(); } catch (_) {}
    this.setVoiceState('speaking');
    const cleaned = (text || '').replace(/\*+/g, '').replace(/#{1,3}\s*/g, '').trim().slice(0, 1500);
    const utter   = new SpeechSynthesisUtterance(cleaned);
    utter.rate = 1.0; utter.pitch = 1.0; utter.volume = 1.0;
    utter.lang = this.selectedLanguage || 'en-US';

    const voices   = window.speechSynthesis.getVoices() || [];
    const exact    = voices.find(v => v.lang === utter.lang);
    const fallback = voices.find(v => v.lang.startsWith((utter.lang || 'en').split('-')[0])) || voices.find(v => v.lang.startsWith('en'));
    if (exact || fallback) utter.voice = exact || fallback;

    const onDone = () => {
      if (this._iosResumeTimer) { clearInterval(this._iosResumeTimer); this._iosResumeTimer = null; }
      if (!this.voiceOverlayOpen) return;
      if (this.voiceLoopActive) setTimeout(() => { if (this.voiceLoopActive && this.voiceOverlayOpen) this._startListening(); }, 500);
      else this.setVoiceState('idle');
    };
    utter.onend   = onDone;
    utter.onerror = e => { if (e.error === 'interrupted' || e.error === 'canceled') return; onDone(); };
    this.currentUtterance = utter;

    this._iosResumeTimer = setInterval(() => {
      if (window.speechSynthesis.speaking) { window.speechSynthesis.pause(); window.speechSynthesis.resume(); }
      else { clearInterval(this._iosResumeTimer); this._iosResumeTimer = null; }
    }, 10000);

    window.speechSynthesis.speak(utter);
  }

  /* -------------------- CHAT -------------------- */
  renderChat() {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    container.innerHTML = '';

    const userMsgs   = this.messages.filter(m => m.role === 'user').length;
    const countBadge = document.getElementById('chat-msg-count');
    const countNum   = document.getElementById('chat-msg-count-num');
    if (countBadge && countNum) {
      if (userMsgs > 0) { countBadge.classList.remove('hidden'); countBadge.classList.add('flex'); countNum.textContent = userMsgs; }
      else              { countBadge.classList.add('hidden');    countBadge.classList.remove('flex'); }
    }

    this.messages.forEach(msg => {
      const isUser  = msg.role === 'user';
      const div     = document.createElement('div');
      div.className = `flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`;
      const wrapper = document.createElement('div');
      wrapper.className = 'max-w-[85%] space-y-1.5';

      if (msg.image) {
        const imgWrap = document.createElement('div');
        imgWrap.className = `p-1.5 rounded-2xl ${isUser ? 'bg-blue-600 rounded-tr-none' : 'bg-white border'}`;
        imgWrap.innerHTML = `<img src="data:image/png;base64,${msg.image}" class="max-h-40 rounded-lg object-cover" alt="Uploaded image">`;
        wrapper.appendChild(imgWrap);
      }

      if (msg.text) {
        const bubble = document.createElement('div');
        bubble.className = `rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${isUser ? 'bg-blue-600 text-white rounded-tr-none' : 'bg-white border border-slate-200 text-slate-800 rounded-tl-none shadow-sm'}`;
        bubble.innerHTML = this.formatText(msg.text);
        wrapper.appendChild(bubble);
      }

      if (msg.type === 'care_finder') {
        const c = document.createElement('div');
        c.className = 'w-full mt-2 animate-fade-in';
        c.innerHTML = `<div class="bg-blue-50 border border-blue-200 rounded-xl p-4 text-center space-y-2 shadow-sm">
          <div class="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center mx-auto shadow-md"><i data-lucide="map-pin" class="w-5 h-5 text-white"></i></div>
          <p class="text-sm font-semibold text-blue-900">Find Medical Care</p>
          <p class="text-xs text-blue-700 pb-1">I can help you locate nearby providers.</p>
          <button onclick="app.navigate('care'); setTimeout(() => app.locateUser(), 300);" class="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold py-2.5 rounded-lg transition-colors flex items-center justify-center gap-2">
            <i data-lucide="navigation" class="w-4 h-4"></i> Show Care Near Me
          </button></div>`;
        wrapper.appendChild(c);
      }

      if (msg.type === 'first_aid' && msg.steps) {
        const gc     = document.createElement('div');
        gc.className = 'w-full mt-2 space-y-3';
        const header = document.createElement('div');
        header.className = 'flex items-center gap-2 mb-3 text-blue-700';
        header.innerHTML = '<i data-lucide="sparkles" class="w-5 h-5"></i><span class="font-bold text-sm">Visual First Aid Guide</span>';
        gc.appendChild(header);
        msg.steps.forEach((step, idx) => {
          const sc = document.createElement('div');
          sc.className = 'bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm flex flex-col';
          const imgHtml = step.image_base64
            ? `<div class="bg-slate-100 flex items-center justify-center p-4 border-b border-slate-100"><img src="data:image/png;base64,${step.image_base64}" alt="Step ${idx + 1}" class="max-h-48 object-contain rounded-lg shadow-sm"></div>`
            : '';
          sc.innerHTML = `${imgHtml}<div class="p-3"><div class="flex items-start gap-2"><span class="bg-blue-100 text-blue-700 min-w-[20px] h-5 rounded-full flex items-center justify-center text-[10px] font-bold mt-0.5">${idx + 1}</span><p class="text-sm text-slate-700 leading-relaxed flex-1">${this.formatText(step.step_text)}</p></div></div>`;
          gc.appendChild(sc);
        });
        wrapper.appendChild(gc);
      }

      div.appendChild(wrapper);
      container.appendChild(div);
    });

    container.scrollTop = container.scrollHeight;
    setTimeout(() => window.lucide?.createIcons(), 10);
  }

  async sendMessage() {
    if (this.chatBusy) return;
    const input = document.getElementById('chat-input');
    const text  = input?.value.trim() || '';
    const image = this.chatImage;
    if (!text && !image) return;

    this.chatBusy = true;
    const btn = document.getElementById('btn-send');
    if (btn) { btn.disabled = true; btn.classList.add('opacity-60'); }
    try { window.speechSynthesis.cancel(); } catch (_) {}

    this.messages.push({ role: 'user', text, image });
    this.renderChat();
    if (input) { input.value = ''; input.style.height = 'auto'; }
    this.clearChatImage();
    this.showLoading();

    const history      = this.messages.filter(m => m.text).slice(0, -1).map(m => ({ role: m.role, text: m.text }));
    const rawResponse  = await this.callNovaTriage(history, text, image);

    let cleanResponse     = rawResponse || "I didn't get a response.";
    let triggerFirstAid   = false;
    let triggerCareFinder = false;

    if (cleanResponse.includes('[TRIGGER_FIRST_AID]'))   { triggerFirstAid   = true; cleanResponse = cleanResponse.replace('[TRIGGER_FIRST_AID]',   '').trim(); }
    if (cleanResponse.includes('[TRIGGER_CARE_FINDER]')) { triggerCareFinder = true; cleanResponse = cleanResponse.replace('[TRIGGER_CARE_FINDER]', '').trim(); }

    this.removeLoading();
    this.messages.push({ role: 'assistant', text: cleanResponse });
    if (triggerCareFinder) this.messages.push({ role: 'assistant', type: 'care_finder' });
    this.renderChat();
    this.speakText(cleanResponse);

    // Save after every assistant reply
    await this._saveCurrentTriageChat();

    if (triggerFirstAid) {
      this.showLoading('Generating visual guide...');
      const guideData = await this.callFirstAidGuide(text);
      this.removeLoading();
      if (guideData?.steps) {
        this.messages.push({ role: 'assistant', type: 'first_aid', steps: guideData.steps });
        this.renderChat();
        setTimeout(() => { const c = document.getElementById('chat-messages'); if (c) c.scrollTop = c.scrollHeight; }, 100);
      }
    }

    this.chatBusy = false;
    if (btn) { btn.disabled = false; btn.classList.remove('opacity-60'); }
    setTimeout(() => this.focusChatInput(), 0);
  }

  showLoading(customText = null) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    const div = document.createElement('div');
    div.id        = 'chat-loading';
    div.className = 'flex justify-start';
    div.innerHTML = customText
      ? `<div class="bg-blue-50 border border-blue-100 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm flex gap-2 items-center"><div class="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div><span class="text-xs text-blue-700 font-semibold">${this.escapeHtml(customText)}</span></div>`
      : `<div class="bg-white border border-slate-200 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm flex gap-1 items-center h-9"><div class="w-1.5 h-1.5 bg-slate-400 rounded-full typing-dot"></div><div class="w-1.5 h-1.5 bg-slate-400 rounded-full typing-dot"></div><div class="w-1.5 h-1.5 bg-slate-400 rounded-full typing-dot"></div></div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }
  removeLoading() { document.getElementById('chat-loading')?.remove(); }

  handleChatImageSelect(input) {
    const file = input.files?.[0]; if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      this.chatImage = reader.result.split(',')[1];
      document.getElementById('chat-image-preview').src = reader.result;
      document.getElementById('chat-image-preview-container').classList.remove('hidden');
      this.toast('Image attached', 'image');
    };
    reader.readAsDataURL(file);
  }

  clearChatImage() {
    this.chatImage = null;
    const img  = document.getElementById('chat-image-preview');
    const box  = document.getElementById('chat-image-preview-container');
    const file = document.getElementById('chat-file-input');
    if (img)  img.src  = '';
    if (box)  box.classList.add('hidden');
    if (file) file.value = '';
  }

  resetChat() {
    try { window.speechSynthesis.cancel(); } catch (_) {}
    this.messages = [{ role: 'assistant', text: this.greetings[this.selectedLanguage] || this.greetings['en-US'] }];
    this.renderChat();
    this.focusChatInput();
    this._renderTriageHistory();
  }

  async callNovaTriage(history, userMessage, base64Image) {
    try {
      let cleanHistory = (history || []).map(t => ({ role: t.role, text: (t.text || '').trim() })).filter(t => t.text);
      while (cleanHistory.length && cleanHistory[0].role !== 'user') cleanHistory.shift();
      const res = await fetch('/api/triage', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          message:      userMessage    || '',
          history:      cleanHistory,
          image_base64: base64Image    || null,
          language:     this.selectedLanguage,
          // FIX: Pass the stable chat_id so the server can upsert instead of
          // inserting a duplicate row on every single message.
          chat_id:      this.currentChatId,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return (await res.json()).text || "I didn't get a response.";
    } catch (e) {
      console.error(e);
      return "I'm having trouble connecting. If this is an emergency, please call local emergency services.";
    }
  }

  async callFirstAidGuide(description) {
    try {
      const res = await fetch('/api/first-aid-guide', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ injury_description: description, language: this.selectedLanguage }),
      });
      if (!res.ok) throw new Error('Failed');
      return await res.json();
    } catch (e) { console.error(e); return null; }
  }

  async callNovaVision(base64Data, prompt) {
    try {
      const res = await fetch('/api/vision', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ image_base64: base64Data, prompt, language: this.selectedLanguage, max_tokens: 2000 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return (await res.json()).text || "I didn't get a response.";
    } catch (e) { console.error(e); return 'Analysis failed. Please try a clearer image.'; }
  }

  /* -------------------- DROPZONES -------------------- */
  initDropzones() {
    this.setupDropzone('xray-dropzone', 'xray-input', f => this.handleDroppedFile(f, 'xray'));
    this.setupDropzone('lab-dropzone',  'lab-input',  f => this.handleDroppedFile(f, 'lab'));
  }
  setupDropzone(dropId, inputId, onFile) {
    const zone = document.getElementById(dropId); if (!zone) return;
    ['dragenter', 'dragover'].forEach(evt => zone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); zone.classList.add(dropId.includes('lab') ? 'drag-active-purple' : 'drag-active'); }));
    ['dragleave', 'drop'].forEach(evt => zone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); zone.classList.remove('drag-active', 'drag-active-purple'); }));
    zone.addEventListener('drop', e => { const f = e.dataTransfer?.files?.[0]; if (f) onFile(f); });
  }
  handleDroppedFile(file, type) { if (type === 'xray') this.handleXrayUpload({ files: [file] }); else this.handleLabUpload({ files: [file] }); }
  handleXrayUpload(input) { this.handleFileUpload(input, 'xray'); }
  handleLabUpload(input)  { this.handleFileUpload(input, 'lab');  }

  handleFileUpload(input, type) {
    const file = input.files?.[0]; if (!file) return;
    if (!file.type.startsWith('image/')) { alert('Please upload an image file.'); return; }
    const reader = new FileReader();
    reader.onloadend = () => {
      const fullData = reader.result, base64 = fullData.split(',')[1];
      if (type === 'xray') {
        this.xrayImage = base64;
        const img = document.getElementById('xray-preview-img');
        img.src = fullData; img.classList.remove('hidden');
        document.getElementById('xray-placeholder').classList.add('hidden');
        document.getElementById('btn-analyze-xray').disabled = false;
        this.toast('Scan uploaded', 'scan-line');
      } else {
        this.labImage = base64;
        const img = document.getElementById('lab-preview-img');
        img.src = fullData; img.classList.remove('hidden');
        document.getElementById('lab-placeholder').classList.add('hidden');
        document.getElementById('btn-analyze-lab').disabled = false;
        this.toast('Lab uploaded', 'microscope');
      }
    };
    reader.readAsDataURL(file);
  }

  /* -------------------- DATABASE SYNC -------------------- */
  async loadUserHistory() {
    // FIX: Guard uses the correctly-initialised this.isAuthenticated (false by
    // default) instead of relying on an undefined value set externally.
    if (!this.isAuthenticated) return;

    try {
      const response = await fetch('/api/history');
      // FIX: Previously a non-ok response was silently swallowed.
      // Now we log it so it's visible in the browser console.
      if (!response.ok) {
        console.warn('loadUserHistory: /api/history returned', response.status);
        return;
      }

      const data = await response.json();

      if (data.triage && data.triage.length > 0) {
        this.triageHistory = data.triage
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
          .map(chat => ({
            id:       chat.id,
            title:    chat.title,
            messages: chat.messages,
            time:     new Date(chat.created_at).toLocaleDateString(),
          }));
        this._renderTriageHistory();
        this._syncHistoryOverlay();
      }

      if (data.documents && data.documents.length > 0) {
        this.xrayHistory = [];
        this.labHistory  = [];
        data.documents
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
          .forEach(doc => {
            const entry = {
              id:         doc.id,
              thumb:      doc.image_b64,
              reportHtml: doc.report_html,
              time:       new Date(doc.created_at).toLocaleDateString(),
            };
            if (doc.doc_type === 'xray') this.xrayHistory.push(entry);
            if (doc.doc_type === 'lab')  this.labHistory.push(entry);
          });
        this._renderXrayHistory();
        this._renderLabHistory();
      }
    } catch (error) {
      console.error('Failed to fetch history from database:', error);
    }
  }

  /* -------------------- HISTORY OVERLAY -------------------- */
  // FIX: openHistoryOverlay() and closeHistoryOverlay() were called from the
  // HTML (nav History button, mobile menu) but never defined on the class,
  // causing TypeError: app.openHistoryOverlay is not a function.

  openHistoryOverlay() {
    this._syncHistoryOverlay();
    document.getElementById('history-overlay')?.classList.remove('hidden');
    if (window.lucide) lucide.createIcons();
  }

  closeHistoryOverlay() {
    document.getElementById('history-overlay')?.classList.add('hidden');
  }

  // Keeps the mobile history overlay list in sync with triageHistory
  _syncHistoryOverlay() {
    const list  = document.getElementById('history-list-mobile');
    const empty = document.querySelector('#history-list-mobile + p, #history-list-mobile p');
    if (!list) return;

    // Remove all dynamic items (keep static empty-state paragraph if present)
    Array.from(list.children).forEach(child => {
      if (!child.id?.includes('empty')) child.remove();
    });

    if (this.triageHistory.length === 0) return;

    this.triageHistory.forEach(chat => {
      const btn       = document.createElement('div');
      const isActive  = chat.id === this.currentChatId;
      btn.className   = `group flex items-center justify-between rounded-xl px-3 py-2.5 transition-colors cursor-pointer border ${isActive ? 'bg-blue-50 border-blue-200 text-blue-800 shadow-sm' : 'bg-white hover:bg-slate-50 text-slate-600 border-transparent hover:border-slate-200'}`;
      btn.innerHTML   = `
        <div class="flex flex-col min-w-0 flex-1" onclick="app.loadTriageChat('${chat.id}'); app.closeHistoryOverlay();">
          <span class="text-sm font-medium truncate">${this.escapeHtml(chat.title)}</span>
          <span class="text-[10px] ${isActive ? 'text-blue-500' : 'text-slate-400'} mt-0.5">${chat.time}</span>
        </div>
        <button onclick="app.deleteTriageChat('${chat.id}', event)" class="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-500 transition-opacity ml-2 shrink-0 p-1">
          <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
        </button>`;
      list.appendChild(btn);
    });
    if (window.lucide) lucide.createIcons();
  }

  /* -------------------- SESSION HISTORY -------------------- */
  startNewTriageChat() {
    this._saveCurrentTriageChat();
    this.currentChatId = this._newChatId();
    this.resetChat();
  }

  async _saveCurrentTriageChat() {
    if (this.messages.length <= 1) return;

    const firstUserMsg = this.messages.find(m => m.role === 'user');
    const title        = firstUserMsg
      ? firstUserMsg.text.slice(0, 32) + (firstUserMsg.text.length > 32 ? '...' : '')
      : 'Triage Session';

    const chatData = {
      id:       this.currentChatId,
      title,
      messages: [...this.messages],
      time:     new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // 1. Local UI update
    const existingIdx = this.triageHistory.findIndex(h => h.id === this.currentChatId);
    if (existingIdx > -1) this.triageHistory[existingIdx] = chatData;
    else                  this.triageHistory.unshift(chatData);
    this._renderTriageHistory();

    // 2. Persist to PostgreSQL (only if authenticated)
    if (this.isAuthenticated) {
      try {
        await fetch('/api/history/triage', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(chatData),
        });
      } catch (err) {
        console.error('Failed to sync chat to database', err);
      }
    }
  }

  _renderTriageHistory() {
    const list  = document.getElementById('triage-history-list');
    const empty = document.getElementById('triage-empty-state');
    if (!list) return;

    Array.from(list.children).forEach(child => { if (child.id !== 'triage-empty-state') child.remove(); });

    if (this.triageHistory.length === 0) {
      if (empty) empty.classList.remove('hidden');
      return;
    }
    if (empty) empty.classList.add('hidden');

    this.triageHistory.forEach(chat => {
      const btn      = document.createElement('div');
      const isActive = chat.id === this.currentChatId;
      btn.className  = `group flex items-center justify-between rounded-xl px-3 py-2.5 transition-colors cursor-pointer border ${isActive ? 'bg-blue-50 border-blue-200 text-blue-800 shadow-sm' : 'bg-white hover:bg-slate-50 text-slate-600 border-transparent hover:border-slate-200'}`;
      btn.innerHTML  = `
        <div class="flex flex-col min-w-0 flex-1" onclick="app.loadTriageChat('${chat.id}')">
          <span class="text-sm font-medium truncate">${this.escapeHtml(chat.title)}</span>
          <span class="text-[10px] ${isActive ? 'text-blue-500' : 'text-slate-400'} mt-0.5">${chat.time}</span>
        </div>
        <button onclick="app.deleteTriageChat('${chat.id}', event)" class="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-500 transition-opacity ml-2 shrink-0 p-1">
          <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
        </button>`;
      list.appendChild(btn);
    });
    if (window.lucide) lucide.createIcons();
  }

  loadTriageChat(id) {
    this._saveCurrentTriageChat();
    const chat = this.triageHistory.find(h => h.id === id);
    if (chat) {
      this.currentChatId = chat.id;
      this.messages      = [...chat.messages];
      this.renderChat();
      this._renderTriageHistory();
    }
  }

  deleteTriageChat(id, event) {
    if (event) event.stopPropagation();
    this.triageHistory = this.triageHistory.filter(h => h.id !== id);
    if (this.currentChatId === id) {
      this.currentChatId = this._newChatId();
      this.resetChat();
    } else {
      this._renderTriageHistory();
    }
    this.toast('Chat deleted', 'trash');
  }

  /* -------------------- SCAN / LAB HISTORY -------------------- */
  _saveXrayHistory(imageB64, reportHtml) {
    const entry = { id: Date.now(), thumb: imageB64, reportHtml, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    this.xrayHistory.unshift(entry);
    if (this.xrayHistory.length > 5) this.xrayHistory.pop();
    this._renderXrayHistory();
  }
  _renderXrayHistory() {
    const panel = document.getElementById('xray-history-panel');
    const list  = document.getElementById('xray-history-list');
    if (!panel || !list) return;
    if (!this.xrayHistory.length) { panel.classList.add('hidden'); return; }
    panel.classList.remove('hidden');
    list.innerHTML = '';
    this.xrayHistory.forEach((entry, idx) => {
      const card     = document.createElement('button');
      card.className = 'w-full flex items-center gap-3 p-2 rounded-xl border border-slate-100 hover:bg-blue-50 hover:border-blue-200 transition-all text-left group';
      card.onclick   = () => this._restoreXrayEntry(entry);
      card.innerHTML = `<img src="data:image/png;base64,${entry.thumb}" class="w-12 h-12 rounded-lg object-cover bg-slate-900 shrink-0 border border-slate-200"><div class="flex-1 min-w-0"><p class="text-xs font-semibold text-slate-700 group-hover:text-blue-700">Scan ${this.xrayHistory.length - idx}</p><p class="text-[10px] text-slate-400 mt-0.5">${entry.time}</p></div><i data-lucide="chevron-right" class="w-3.5 h-3.5 text-slate-300 group-hover:text-blue-500 shrink-0"></i>`;
      list.appendChild(card);
    });
    if (window.lucide) lucide.createIcons();
  }
  _restoreXrayEntry(entry) {
    const img = document.getElementById('xray-preview-img');
    if (img) { img.src = `data:image/png;base64,${entry.thumb}`; img.classList.remove('hidden'); }
    document.getElementById('xray-placeholder')?.classList.add('hidden');
    const btn = document.getElementById('btn-analyze-xray'); if (btn) btn.disabled = false;
    this.xrayImage = entry.thumb;
    const result = document.getElementById('xray-result'); if (result) result.innerHTML = entry.reportHtml;
    if (window.lucide) lucide.createIcons();
    this.toast('Scan restored from session', 'clock');
  }
  _saveLabHistory(imageB64, reportHtml) {
    const entry = { id: Date.now(), thumb: imageB64, reportHtml, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    this.labHistory.unshift(entry);
    if (this.labHistory.length > 5) this.labHistory.pop();
    this._renderLabHistory();
  }
  _renderLabHistory() {
    const panel = document.getElementById('lab-history-panel');
    const list  = document.getElementById('lab-history-list');
    if (!panel || !list) return;
    if (!this.labHistory.length) { panel.classList.add('hidden'); return; }
    panel.classList.remove('hidden');
    list.innerHTML = '';
    this.labHistory.forEach((entry, idx) => {
      const card     = document.createElement('button');
      card.className = 'w-full flex items-center gap-3 p-2 rounded-xl border border-slate-100 hover:bg-purple-50 hover:border-purple-200 transition-all text-left group';
      card.onclick   = () => this._restoreLabEntry(entry);
      card.innerHTML = `<img src="data:image/png;base64,${entry.thumb}" class="w-12 h-12 rounded-lg object-cover bg-slate-100 shrink-0 border border-slate-200"><div class="flex-1 min-w-0"><p class="text-xs font-semibold text-slate-700 group-hover:text-purple-700">Report ${this.labHistory.length - idx}</p><p class="text-[10px] text-slate-400 mt-0.5">${entry.time}</p></div><i data-lucide="chevron-right" class="w-3.5 h-3.5 text-slate-300 group-hover:text-purple-500 shrink-0"></i>`;
      list.appendChild(card);
    });
    if (window.lucide) lucide.createIcons();
  }
  _restoreLabEntry(entry) {
    const img = document.getElementById('lab-preview-img');
    if (img) { img.src = `data:image/png;base64,${entry.thumb}`; img.classList.remove('hidden'); }
    document.getElementById('lab-placeholder')?.classList.add('hidden');
    const btn = document.getElementById('btn-analyze-lab'); if (btn) btn.disabled = false;
    this.labImage = entry.thumb;
    const result = document.getElementById('lab-result'); if (result) result.innerHTML = entry.reportHtml;
    if (window.lucide) lucide.createIcons();
    this.toast('Report restored from session', 'clock');
  }

  setXrayLanguage(code) { this.xrayLanguage = code; localStorage.setItem('pulseNova_xray_lang', code); }
  setLabLanguage(code)  { this.labLanguage  = code; localStorage.setItem('pulseNova_lab_lang',  code); }

  _initVisionLangSelects() {
    const opts = this.supportedLanguages.map(l => `<option value="${l.code}">${l.label}</option>`).join('');
    const xsel = document.getElementById('xray-lang-select');
    const lsel = document.getElementById('lab-lang-select');
    if (xsel && !xsel.children.length) { xsel.innerHTML = opts; xsel.value = this.xrayLanguage || this.selectedLanguage; }
    if (lsel && !lsel.children.length) { lsel.innerHTML = opts; lsel.value = this.labLanguage  || this.selectedLanguage; }
  }

  async analyzeXray() {
    if (!this.xrayImage) return;
    this._initVisionLangSelects();
    const btn      = document.getElementById('btn-analyze-xray');
    btn.disabled   = true;
    btn.innerHTML  = `<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Analyzing…`;
    const langCode = this.xrayLanguage || this.selectedLanguage;
    const langObj  = this.supportedLanguages.find(l => l.code === langCode);
    const langName = langObj ? langObj.label.split(' (')[0] : 'English';
    const prompt   = `You are an expert radiologist assistant. Analyze this X-Ray, MRI, or CT scan image in thorough detail.
CRITICAL: Write this ENTIRE report — every heading, label, and sentence — exclusively in ${langName}. Do NOT use English for any part.

Use exactly this structure:

### 🏥 Overview
Describe the scan type (X-Ray / MRI / CT), the body region shown, and the patient position (e.g. AP view, lateral view). State clearly which bone(s) or structure(s) are visible.

### 🦴 Bone & Structural Findings
- **Bone(s) Identified:** Name every bone or structure visible in the image.
- **Fracture / Injury:** State whether a fracture or abnormality is present. If yes, specify: location on the bone (e.g. mid-shaft, distal end, neck), fracture type if visible (e.g. transverse, oblique, comminuted, hairline), and which side of the body (left / right / not determinable).
- **Alignment:** Is the bone aligned normally or displaced/angulated? Describe the degree if visible.
- **Soft Tissue:** Note any visible swelling, calcification, or soft-tissue abnormality.
- **Bone Density & Quality:** Comment on bone density — normal, reduced (possible osteoporosis), or irregular.
- **Surrounding Structures:** Note any joint involvement, adjacent bone changes, or hardware (screws, plates) if present.

### ⚠️ Key Concerns
List any urgent or notable findings in plain language. If this appears to be an emergency finding, state that clearly.

### 🩺 Next Steps & Which Doctor to See
- **Recommended Specialist:** Name the exact type of doctor to consult and explain why.
- **Urgency:** State whether this needs emergency care, urgent care within 24–48 hours, or a scheduled outpatient appointment.
- **Likely Treatment Options:** Briefly describe what treatment may involve.
- **What to Tell the Doctor:** List 2–3 specific things the patient should mention.
- **What to Avoid:** Note any activities or movements to avoid until evaluated.

### 📋 Questions to Ask Your Doctor
List 3–4 questions the patient should ask their specialist.

---
*⚕️ IMPORTANT: This is an AI-assisted observation only and is NOT a medical diagnosis. Always consult a qualified medical professional for evaluation and treatment.*`;
    const text       = await this.callNovaVision(this.xrayImage, prompt);
    const reportHtml = this.formatMarkdown(text);
    document.getElementById('xray-result').innerHTML = reportHtml;
    this._saveXrayHistory(this.xrayImage, reportHtml);
    btn.disabled  = false;
    btn.innerHTML = `<i data-lucide="activity" class="w-4 h-4"></i> Analyze Scan`;
    if (window.lucide) lucide.createIcons();
  }

  async analyzeLabs() {
    if (!this.labImage) return;
    this._initVisionLangSelects();
    const btn      = document.getElementById('btn-analyze-lab');
    btn.disabled   = true;
    btn.innerHTML  = `<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Translating…`;
    const langCode = this.labLanguage || this.selectedLanguage;
    const langObj  = this.supportedLanguages.find(l => l.code === langCode);
    const langName = langObj ? langObj.label.split(' (')[0] : 'English';
    const prompt   = `You are an expert medical lab analyst. Carefully read this lab report image and produce a comprehensive patient-friendly interpretation.
CRITICAL: Write this ENTIRE report — every heading, label, and sentence — exclusively in ${langName}. Do NOT use English for any part.

Use exactly this structure:

### 🩸 Overview
Identify the type of lab panel(s) present. State the date of the report if visible.

### 🔬 Detailed Test Results
For EVERY test result visible in the report:
- **[Test Name] ([Abbreviation]):** Reported value and unit — Normal range — Status (Normal / High / Low / Critical) — Plain-language meaning in 1–2 sentences.

### ⚠️ Out-of-Range Flags
List every result outside the normal range:
- **[Test Name]:** Value found vs. normal range — What this could mean — How serious it may be (Minor / Moderate / Requires prompt attention).

### 🩺 Next Steps & Which Doctor to See
- **Recommended Specialist:** Name the specific type of doctor to consult based on the flagged results.
- **Urgency:** Emergency care / prompt appointment within a week / routine follow-up.
- **Lifestyle Adjustments:** Immediate diet, activity, or lifestyle changes recommended.
- **Possible Follow-up Tests:** Additional tests a doctor may order.
- **What to Tell the Doctor:** 3 specific things to mention.

### 📋 Questions to Ask Your Doctor
List 4 questions the patient should ask about these results.

---
*⚕️ IMPORTANT: This is an AI-assisted interpretation only and is NOT a medical diagnosis. Lab results must always be reviewed by a qualified healthcare provider.*`;
    const text       = await this.callNovaVision(this.labImage, prompt);
    const reportHtml = this.formatMarkdown(text);
    document.getElementById('lab-result').innerHTML = reportHtml;
    this._saveLabHistory(this.labImage, reportHtml);
    btn.disabled  = false;
    btn.innerHTML = `<i data-lucide="activity" class="w-4 h-4"></i> Translate Report`;
    if (window.lucide) lucide.createIcons();
  }

  /* -------------------- VITALS -------------------- */
  setVitalsContext(ctx) {
    this.vitals.context = ctx;
    document.querySelectorAll('.vitals-context-btn').forEach(btn => {
      const active  = btn.dataset.context === ctx;
      btn.className = active
        ? 'vitals-context-btn bg-blue-50 border border-blue-200 text-blue-700 rounded-xl px-3 py-2 text-xs font-semibold'
        : 'vitals-context-btn bg-white border border-slate-200 text-slate-600 rounded-xl px-3 py-2 text-xs font-semibold';
    });
    const labelMap = { resting: 'Resting', after_exercise: 'After exercise', feeling_unwell: 'Feeling unwell' };
    this._setText('sum-context', labelMap[ctx] || 'Resting');
  }

  resetVitalsSummary() {
    this.vitals.lastStableBpm = null; this.vitals.summaryReady = false;
    // FIX: Only show the summary card when a reading is in progress or
    // complete. Remove 'hidden' is intentional here (called from startVitals).
    document.getElementById('vitals-summary-card')?.classList.remove('hidden');
    this._setSummaryBadge('Waiting', 'chip-slate');
    this._setText('sum-bpm',        '--');
    this._setText('sum-confidence', '--');
    this._setText('sum-signal',     'Signal quality: --');
    this._setText('sum-status',     'No reading yet');
    this._setText('sum-summary',    'Take a reading to generate a pulse summary.');
    this._setText('sum-guidance',   'Stay still, place your fingertip fully over the camera, and tap Start.');
    this._setText('sum-warning',    'This is a camera-based estimate and not a diagnosis. If you have chest pain, trouble breathing, fainting, or severe symptoms, seek urgent care.');
    this.setVitalsContext(this.vitals.context || 'resting');
  }

  _setText(id, text)          { const el = document.getElementById(id); if (el) el.textContent = text; }
  _setSummaryBadge(text, cls) { const badge = document.getElementById('vitals-summary-badge'); if (!badge) return; badge.className = `chip ${cls}`; badge.textContent = text; }
  _qualityLabel(q)            { return q >= 60 ? 'Good' : q >= 25 ? 'Fair' : 'Poor'; }
  _confidenceLabel({ signalQuality = 0, bpmHistory = [], stableBpmCount = 0 }) {
    const variance = bpmHistory.length > 1 ? Math.max(...bpmHistory) - Math.min(...bpmHistory) : 999;
    if (signalQuality >= 60 && bpmHistory.length >= 5 && variance <= 12 && stableBpmCount >= 1) return 'High';
    if (signalQuality >= 30 && bpmHistory.length >= 3) return 'Moderate';
    return 'Low';
  }

  _buildVitalsInterpretation(bpm) {
    const ctx = this.vitals.context;
    let status = '', badge = ['chip-slate', 'Measured'], summary = '', guidance = '', warning = '';
    if (ctx === 'after_exercise') {
      if (bpm < 60)        { status = 'Lower than expected after activity'; badge = ['chip-amber', 'Check again'];  summary = 'Your pulse estimate is on the lower side for a post-exercise reading.'; guidance = 'Sit and retake the reading in 1–2 minutes. Make sure your finger fully covers the camera lens.'; }
      else if (bpm <= 140) { status = 'Expected after activity';            badge = ['chip-green', 'Expected'];     summary = 'This pulse reading can be normal shortly after exercise.';                          guidance = 'Rest for a few minutes, hydrate, and retake the reading to see if it trends down.'; }
      else                 { status = 'High after activity';                badge = ['chip-amber', 'Elevated'];     summary = 'Your pulse is fairly high for a post-exercise reading.';                            guidance = 'Stop activity, rest, and retake after 3–5 minutes. If it stays high or you feel unwell, seek care.'; }
    } else {
      if (bpm < 60)        { status = 'Below typical resting range';  badge = ['chip-amber', 'Below Range']; summary = 'A resting pulse under 60 can be normal for some people (especially athletes), but can also happen with fatigue or medications.'; guidance = 'If you feel fine, retake once while seated and relaxed. If you feel dizzy, weak, or faint, seek medical care.'; }
      else if (bpm <= 100) { status = 'Within typical resting range'; badge = ['chip-green', 'Normal Range']; summary = 'This reading is within a common resting heart-rate range for many adults.';                                                       guidance = 'If symptoms continue, use Triage to describe how you feel and get next-step guidance.'; }
      else if (bpm <= 120) { status = 'Mildly elevated';              badge = ['chip-amber', 'Elevated'];     summary = 'Your pulse is mildly elevated. This can happen with stress, anxiety, caffeine, dehydration, fever, or movement.';             guidance = 'Rest for 2–3 minutes, drink water, and retake. If you feel unwell, use Triage or seek urgent care.'; }
      else                 { status = 'High pulse reading';           badge = ['chip-red',   'High'];         summary = 'Your pulse reading is high and may need attention, especially if you are resting.';                                          guidance = 'Sit down and retake after 2–3 minutes. If it stays high or you have symptoms (chest pain, trouble breathing, fainting), seek urgent care.'; }
    }
    if (ctx === 'feeling_unwell') warning = 'Because you selected "Feeling unwell," take this reading seriously if you also have chest pain, shortness of breath, dizziness, or fainting. Consider urgent evaluation.';
    else                          warning = 'This is a camera-based estimate and not a diagnosis. For severe symptoms, call emergency services or seek urgent care.';
    return { status, badge, summary, guidance, warning };
  }

  updateVitalsSummary(final = false) {
    const bpm       = this.vitals.lastStableBpm || (this.vitals.bpmHistory.length ? Math.round(this.vitals.bpmHistory.reduce((a, b) => a + b, 0) / this.vitals.bpmHistory.length) : null);
    const signalQ   = Math.round(this.vitals.signalQuality || 0);
    const confidence = this._confidenceLabel(this.vitals);
    document.getElementById('vitals-summary-card')?.classList.remove('hidden');
    this._setText('sum-bpm',        bpm ? String(Math.round(bpm)) : '--');
    this._setText('sum-confidence', confidence);
    this._setText('sum-signal',     `Signal quality: ${this._qualityLabel(signalQ)} (${signalQ}%)`);
    if (!bpm) {
      this._setSummaryBadge('Measuring', 'chip-blue');
      this._setText('sum-status',  'Collecting pulse data');
      this._setText('sum-summary', 'PulseNova is still collecting a stable signal. Keep your finger still on the camera lens.');
      this._setText('sum-guidance','Avoid moving your finger and keep steady pressure on the lens for a few more seconds.');
      return;
    }
    const interp = this._buildVitalsInterpretation(Math.round(bpm));
    this._setText('sum-status',   interp.status);
    this._setText('sum-summary',  interp.summary);
    this._setText('sum-guidance', interp.guidance);
    this._setText('sum-warning',  interp.warning);
    this._setSummaryBadge(final ? interp.badge[1] : 'Live', final ? interp.badge[0] : 'chip-blue');
  }

  useVitalsInTriage() {
    const bpm = this.vitals.lastStableBpm;
    if (!bpm) { this.toast('Take a pulse reading first.', 'alert-triangle'); return; }
    const q          = Math.round(this.vitals.signalQuality || 0);
    const confidence = this._confidenceLabel(this.vitals);
    const ctxLabel   = { resting: 'resting', after_exercise: 'after exercise', feeling_unwell: 'feeling unwell' }[this.vitals.context] || 'resting';
    const draft      = `My pulse reading was ${Math.round(bpm)} BPM (${confidence.toLowerCase()} confidence, ${q}% signal quality) while ${ctxLabel}. Please help me interpret this with my symptoms.`;
    this.navigate('triage');
    const input = document.getElementById('chat-input');
    if (input) { input.value = draft; input.focus(); input.setSelectionRange(draft.length, draft.length); input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 128) + 'px'; }
  }

  async startVitals() {
    if (this.vitals.isMonitoring) return;
    this.vitals.video  = document.getElementById('vitals-video');
    this.vitals.canvas = document.getElementById('ppg-graph');
    this.vitals.ctx    = this.vitals.canvas.getContext('2d');
    this.vitals.canvas.width  = this.vitals.canvas.offsetWidth  || 400;
    this.vitals.canvas.height = this.vitals.canvas.offsetHeight || 160;
    const statusEl = document.getElementById('vitals-status');
    const qualWrap = document.getElementById('signal-quality-wrap');
    // resetVitalsSummary() is now only called here (on explicit Start),
    // not in the constructor, so the card doesn't flash on page load.
    this.resetVitalsSummary();
    try {
      statusEl.innerText = 'Requesting camera…';
      this.vitals.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 320 }, height: { ideal: 240 } },
      });
      this.vitals.video.srcObject = this.vitals.stream;
      await this.vitals.video.play();
      this.vitals.torchTrack = this.vitals.stream.getVideoTracks()[0];
      await this._setTorch(true);
      document.getElementById('btn-start-vitals').classList.add('hidden');
      document.getElementById('btn-stop-vitals').classList.remove('hidden');
      if (qualWrap) qualWrap.classList.remove('hidden');
      this.vitals.isMonitoring  = true;
      this.vitals.redValues     = []; this.vitals.smoothed = []; this.vitals.bpmHistory = [];
      this.vitals.stableBpmCount = 0;  this.vitals.lastPeakTime = 0;
      this.vitals.signalQuality  = 0;  this.vitals.lastStableBpm = null;
      statusEl.innerText = 'Cover lens with fingertip…';
      this._setSummaryBadge('Measuring', 'chip-blue');
      this._setText('sum-status',  'Position finger');
      this._setText('sum-summary', 'Place your fingertip over the camera lens and keep still while PulseNova detects your pulse.');
      this.vitals.timeoutId = setTimeout(() => {
        if (this.vitals.isMonitoring) {
          statusEl.innerText = 'No stable signal. Cover lens fully.';
          this._setSummaryBadge('Retake', 'chip-amber');
          this._setText('sum-status',  'No stable signal');
          this._setText('sum-summary', 'PulseNova could not detect a stable pulse signal.');
          this._setText('sum-guidance','Try brighter lighting, keep your finger steady, and cover the lens completely.');
          this.stopVitals(false);
        }
      }, 20000);
      this.processVitalsFrame();
      this.toast('Pulse reading started', 'activity');
    } catch (err) {
      console.error(err);
      statusEl.innerText = 'Camera access denied.';
      this._setSummaryBadge('Blocked', 'chip-red');
      this._setText('sum-status',  'Camera access required');
      this._setText('sum-summary', 'PulseNova needs camera access to estimate your pulse.');
      this._setText('sum-guidance','Allow camera permissions in your browser settings and try again.');
      alert('Unable to access camera.');
    }
  }

  async _setTorch(on) {
    const track = this.vitals.torchTrack; if (!track) return;
    try { const caps = track.getCapabilities?.() || {}; if (caps.torch) { await track.applyConstraints({ advanced: [{ torch: on }] }); this.vitals.torchOn = on; } } catch (_) {}
  }

  stopVitals(autoStop = false) {
    this.vitals.isMonitoring = false;
    if (this.vitals.animId)    cancelAnimationFrame(this.vitals.animId);
    if (this.vitals.timeoutId) clearTimeout(this.vitals.timeoutId);
    this._setTorch(false).catch(() => {});
    if (this.vitals.stream) { this.vitals.stream.getTracks().forEach(t => t.stop()); this.vitals.stream = null; this.vitals.torchTrack = null; }
    document.getElementById('btn-stop-vitals').classList.add('hidden');
    document.getElementById('btn-start-vitals').classList.remove('hidden');
    document.getElementById('signal-quality-wrap')?.classList.add('hidden');
    const statusEl     = document.getElementById('vitals-status');
    statusEl.className = '';
    statusEl.innerText = autoStop ? '✓ Reading captured' : 'Stopped';
    if (autoStop || this.vitals.lastStableBpm) this.updateVitalsSummary(true);
    else                                       this.updateVitalsSummary(false);
    this.toast(autoStop ? 'Reading captured' : 'Reading stopped', autoStop ? 'check' : 'square');
  }

  processVitalsFrame() {
    if (!this.vitals.isMonitoring) return;
    const now = performance.now();
    this.vitals.processCtx.drawImage(this.vitals.video, 0, 0, 20, 20);
    const imageData = this.vitals.processCtx.getImageData(0, 0, 20, 20).data;
    let redSum = 0, greenSum = 0;
    const pixels = imageData.length / 4;
    for (let i = 0; i < imageData.length; i += 4) { redSum += imageData[i]; greenSum += imageData[i + 1]; }
    const avgRed   = redSum   / pixels;
    const avgGreen = greenSum / pixels;
    const quality  = Math.min(100, Math.max(0, ((avgRed - avgGreen) / 80) * 100));
    this.vitals.signalQuality = quality;
    this._updateSignalQuality(quality);
    this.vitals.redValues.push({ time: now, val: avgRed });
    if (this.vitals.redValues.length > 240) this.vitals.redValues.shift();
    if (this.vitals.redValues.length >= 5) {
      const last5    = this.vitals.redValues.slice(-5).map(d => d.val);
      const smoothVal = last5.reduce((a, b) => a + b, 0) / 5;
      this.vitals.smoothed.push({ time: now, val: smoothVal });
      if (this.vitals.smoothed.length > 240) this.vitals.smoothed.shift();
    }
    if (quality > 30 && this.vitals.smoothed.length > 30) this.detectHeartbeat(now);
    if (!this.vitals.lastStableBpm && this.vitals.smoothed.length > 20 && Math.floor(now) % 800 < 20) this.updateVitalsSummary(false);
    this.drawPPGGraph();
    this.vitals.animId = requestAnimationFrame(this.processVitalsFrame);
  }

  _updateSignalQuality(quality) {
    const bar  = document.getElementById('signal-quality-bar');
    const text = document.getElementById('signal-quality-text');
    if (!bar || !text) return;
    bar.style.width = quality + '%';
    if      (quality < 25) { bar.className = 'h-full bg-red-400 rounded-full transition-all duration-500';    text.textContent = 'Poor — cover lens more'; }
    else if (quality < 60) { bar.className = 'h-full bg-yellow-400 rounded-full transition-all duration-500'; text.textContent = 'Fair — hold still'; }
    else                   { bar.className = 'h-full bg-green-400 rounded-full transition-all duration-500';  text.textContent = 'Good signal'; }
  }

  detectHeartbeat(now) {
    const data = this.vitals.smoothed;
    if (data.length < 40) return;
    const vals = data.slice(-60).map(d => d.val);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std  = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
    const threshold = mean + std * 0.6;
    const len  = data.length;
    const cur  = data[len - 1].val, prev = data[len - 2].val, prev2 = data[len - 3].val;
    if (prev > cur && prev > prev2 && prev > threshold) {
      if (now - this.vitals.lastPeakTime > 350) {
        const timeDiff = now - this.vitals.lastPeakTime;
        if (this.vitals.lastPeakTime > 0 && timeDiff < 1500) {
          const instantBPM = 60000 / timeDiff;
          if (instantBPM >= 45 && instantBPM <= 180) {
            this.vitals.bpmHistory.push(instantBPM);
            if (this.vitals.bpmHistory.length > 8) this.vitals.bpmHistory.shift();
            if (this.vitals.bpmHistory.length >= 4) {
              const avgBPM   = this.vitals.bpmHistory.reduce((a, b) => a + b, 0) / this.vitals.bpmHistory.length;
              const bpmRange = Math.max(...this.vitals.bpmHistory) - Math.min(...this.vitals.bpmHistory);
              document.getElementById('bpm-display').innerText = Math.round(avgBPM);
              document.getElementById('vitals-status').innerText = 'Detecting pulse…';
              this.vitals.lastStableBpm = Math.round(avgBPM);
              this.updateVitalsSummary(false);
              if (bpmRange <= 14 && this.vitals.bpmHistory.length >= 5) {
                this.vitals.stableBpmCount++;
                if (this.vitals.stableBpmCount >= 2) {
                  document.getElementById('vitals-status').innerText = `✓ Stable: ${Math.round(avgBPM)} BPM`;
                  clearTimeout(this.vitals.timeoutId);
                  this.vitals.lastStableBpm = Math.round(avgBPM);
                  this.updateVitalsSummary(true);
                  setTimeout(() => this.stopVitals(true), 800);
                }
              } else { this.vitals.stableBpmCount = 0; }
            }
          }
        }
        this.vitals.lastPeakTime = now;
      }
    }
  }

  drawPPGGraph() {
    const ctx = this.vitals.ctx, w = this.vitals.canvas.width, h = this.vitals.canvas.height;
    const data = this.vitals.smoothed.length >= 5 ? this.vitals.smoothed : this.vitals.redValues;
    ctx.clearRect(0, 0, w, h);
    if (data.length < 2) return;
    let min = Infinity, max = -Infinity;
    for (const d of data) { if (d.val < min) min = d.val; if (d.val > max) max = d.val; }
    const range = max - min || 1;
    ctx.beginPath(); ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((data[i].val - min) / range) * (h * 0.8) - (h * 0.1);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  /* -------------------- CARE FINDER -------------------- */
  async ensureGoogleMaps() {
    if (this.googleMapsReady)   return;
    if (this.googleMapsLoading) return this.googleMapsLoading;
    this.googleMapsLoading = new Promise(async (resolve, reject) => {
      try {
        if (window.google?.maps?.places) { this.googleMapsReady = true; this.initGoogleServices(); resolve(); return; }
        const cfg    = await fetch('/config').then(r => r.json());
        const apiKey = cfg.google_maps_api_key;
        if (!apiKey) throw new Error('GOOGLE_MAPS_API_KEY not set on server.');
        window.__pulseNovaMapsInit = () => {
          this.googleMapsReady = true;
          try { this.initGoogleServices(); resolve(); } catch (e) { reject(e); } finally { delete window.__pulseNovaMapsInit; }
        };
        if (document.getElementById('pulse-nova-gmaps')) return;
        const script   = document.createElement('script');
        script.id      = 'pulse-nova-gmaps';
        script.src     = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&libraries=places,geometry&callback=__pulseNovaMapsInit&loading=async`;
        script.async   = true; script.defer = true;
        script.onerror = () => reject(new Error('Failed to load Google Maps SDK'));
        document.head.appendChild(script);
      } catch (err) { reject(err); }
    });
    return this.googleMapsLoading;
  }

  initGoogleServices() {
    if (this.gmap) return;
    const host     = document.getElementById('care-map-host');
    this.gmap      = new google.maps.Map(host, { center: { lat: 41.7658, lng: -72.6734 }, zoom: 12, disableDefaultUI: true });
    this.placesService = new google.maps.places.PlacesService(this.gmap);
    this.geocoder      = new google.maps.Geocoder();
  }

  setCareStatus(text, isError = false) {
    const el = document.getElementById('care-status'); if (!el) return;
    el.textContent = text || '';
    el.className   = `text-xs flex items-center ${isError ? 'text-red-500' : 'text-slate-400'}`;
  }

  initCareSearch() {
    const input = document.getElementById('care-search');
    const r     = document.getElementById('care-radius');
    const rv    = document.getElementById('care-radius-val');
    if (input) input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); this.searchCareByText(); } });
    if (r && rv) { rv.textContent = r.value; r.addEventListener('input', () => rv.textContent = r.value); }
  }

  _careRadiusMeters() { return Math.round(Number(document.getElementById('care-radius')?.value || 8) * 1609.34); }

  async searchCareByText() {
    const query = document.getElementById('care-search')?.value.trim() || '';
    if (!query) { this.setCareStatus('Enter a city or ZIP code.', true); return; }
    try {
      this.setCareStatus('Finding area…');
      await this.ensureGoogleMaps();
      const geo = await new Promise((resolve, reject) =>
        this.geocoder.geocode({ address: query }, (results, status) =>
          status === 'OK' && results?.length ? resolve(results[0]) : reject(new Error(`Geocoding failed: ${status}`))
        )
      );
      const loc = geo.geometry.location;
      this.care.userLat     = loc.lat(); this.care.userLon = loc.lng();
      this.care.centerLabel = geo.formatted_address || query;
      if (this.gmap) { this.gmap.setCenter({ lat: this.care.userLat, lng: this.care.userLon }); this.gmap.setZoom(12); }
      await this.fetchNearbyProviders(this.care.userLat, this.care.userLon);
    } catch (err) { console.error(err); this.setCareStatus('Could not search this area.', true); }
  }

  async locateUser() {
    const btn  = document.getElementById('btn-locate'), orig = btn?.innerHTML;
    if (btn) { btn.disabled = true; btn.innerHTML = 'Locating…'; }
    this.setCareStatus('Requesting location…');
    try {
      await this.ensureGoogleMaps();
      const pos = await new Promise((resolve, reject) => {
        if (!navigator.geolocation) { reject(new Error('Geolocation not supported')); return; }
        navigator.geolocation.getCurrentPosition(resolve, reject, { enableHighAccuracy: true, timeout: 10000 });
      });
      this.care.userLat     = pos.coords.latitude; this.care.userLon = pos.coords.longitude;
      this.care.centerLabel = 'your location';
      await this.fetchNearbyProviders(this.care.userLat, this.care.userLon);
    } catch (e) { console.error(e); this.setCareStatus("Couldn't get location. Try a ZIP/city.", true); this.toast('Enable location permission or search by city/ZIP', 'alert-triangle'); }
    finally { if (btn) { btn.disabled = false; btn.innerHTML = orig; } if (window.lucide) lucide.createIcons(); }
  }

  nearbySearchPromise(request) {
    return new Promise((resolve, reject) => this.placesService.nearbySearch(request, (results, status) => {
      const ok = google.maps.places.PlacesServiceStatus;
      if (status === ok.OK || status === ok.ZERO_RESULTS) resolve(results || []);
      else reject(new Error(`Places search failed: ${status}`));
    }));
  }

  textSearchPromise(request) {
    return new Promise((resolve, reject) => this.placesService.textSearch(request, (results, status) => {
      const ok = google.maps.places.PlacesServiceStatus;
      if (status === ok.OK || status === ok.ZERO_RESULTS) resolve(results || []);
      else reject(new Error(`Places text search failed: ${status}`));
    }));
  }

  getDistanceMiles(lat1, lon1, lat2, lon2) {
    if (window.google?.maps?.geometry?.spherical) {
      return google.maps.geometry.spherical.computeDistanceBetween(
        new google.maps.LatLng(lat1, lon1), new google.maps.LatLng(lat2, lon2)
      ) / 1609.344;
    }
    const R = 6371000, toRad = x => x * Math.PI / 180;
    const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)) / 1609.344;
  }

  normalizePlace(place, inferredType) {
    const loc  = place.geometry?.location;
    const lat  = typeof loc?.lat === 'function' ? loc.lat() : loc?.lat;
    const lon  = typeof loc?.lng === 'function' ? loc.lng() : loc?.lng;
    const name = (place.name || '').toLowerCase(), types = place.types || [];
    let type   = inferredType || 'Clinic';
    if (types.includes('hospital'))  type = inferredType === 'Urgent Care' ? 'Urgent Care' : 'Hospital';
    if (types.includes('pharmacy'))  type = 'Pharmacy';
    if (types.includes('dentist'))   type = 'Dentist';
    if (name.includes('urgent care'))                                        type = 'Urgent Care';
    if (name.includes('pediatric'))                                          type = 'Pediatrician';
    if (name.includes('primary care') || name.includes('family medicine'))   type = 'Primary Care';
    if (type === 'Hospital' && (name.includes('emergency') || name.includes(' er '))) type = 'ER';
    const distance = (this.care.userLat != null && lat != null)
      ? this.getDistanceMiles(this.care.userLat, this.care.userLon, lat, lon)
      : null;
    return { id: place.place_id, name: place.name || 'Unknown provider', type, address: place.vicinity || place.formatted_address || 'Address unavailable', rating: place.rating || null, openNow: place.opening_hours?.open_now ?? null, distance, lat, lon, placeId: place.place_id };
  }

  dedupeProviders(list) {
    const seen = new Map();
    for (const p of list) {
      if (!p?.id) continue;
      if (!seen.has(p.id)) seen.set(p.id, p);
      else if (seen.get(p.id).type === 'Hospital' && p.type === 'Urgent Care') seen.set(p.id, p);
    }
    return Array.from(seen.values());
  }

  async fetchNearbyProviders(lat, lon) {
    this.setCareStatus('Searching nearby care…');
    this.providers = []; this.renderProviders();
    try {
      const center = new google.maps.LatLng(lat, lon);
      this.gmap.setCenter(center);
      const radius = this._careRadiusMeters();
      const [hospitals, pharmacies, dentists, urgentCare, primaryCare, pediatrics] = await Promise.all([
        this.nearbySearchPromise({ location: center, radius, type: 'hospital' }),
        this.nearbySearchPromise({ location: center, radius, type: 'pharmacy' }),
        this.nearbySearchPromise({ location: center, radius, type: 'dentist'  }),
        this.textSearchPromise(  { location: center, radius, query: 'urgent care'          }),
        this.textSearchPromise(  { location: center, radius, query: 'primary care clinic'  }),
        this.textSearchPromise(  { location: center, radius, query: 'pediatrician'         }),
      ]);
      let merged = [
        ...hospitals.map(p  => this.normalizePlace(p, 'Hospital'     )),
        ...pharmacies.map(p => this.normalizePlace(p, 'Pharmacy'     )),
        ...dentists.map(p   => this.normalizePlace(p, 'Dentist'      )),
        ...urgentCare.map(p => this.normalizePlace(p, 'Urgent Care'  )),
        ...primaryCare.map(p => this.normalizePlace(p, 'Primary Care')),
        ...pediatrics.map(p => this.normalizePlace(p, 'Pediatrician' )),
      ];
      merged = this.dedupeProviders(merged);
      merged.sort((a, b) => (a.distance ?? 999) - (b.distance ?? 999));
      this.providers = merged.slice(0, 30);
      if (document.getElementById('care-filter'))    document.getElementById('care-filter').value    = 'All';
      if (document.getElementById('provider-filter')) document.getElementById('provider-filter').value = '';
      this.renderProviders();
      const visible = this.getFilteredProviders().length;
      const label   = this.care.centerLabel ? ` near ${this.care.centerLabel}` : '';
      this.setCareStatus(this.providers.length
        ? `Found ${this.providers.length} providers${label}. Showing ${visible}.`
        : `No providers found${label}.`
      );
      this.toast(`${this.providers.length} providers found`, 'map-pin');
    } catch (e) { console.error(e); this.setCareStatus('Care search failed. Check Maps API key.', true); }
  }

  getFilteredProviders() {
    const typeFilter = document.getElementById('care-filter')?.value || 'All';
    const textFilter = (document.getElementById('provider-filter')?.value || '').trim().toLowerCase();
    let filtered = this.providers.slice();
    if (typeFilter !== 'All') filtered = filtered.filter(p => p.type === typeFilter);
    if (textFilter)           filtered = filtered.filter(p => (p.name || '').toLowerCase().includes(textFilter) || (p.address || '').toLowerCase().includes(textFilter));
    return filtered;
  }

  renderProviders() {
    const list = document.getElementById('providers-list'); if (!list) return;
    list.innerHTML = '';
    const filtered = this.getFilteredProviders();
    if (!filtered.length) {
      list.innerHTML = `<div class="col-span-full bg-white border border-slate-200 rounded-2xl p-6 text-center text-slate-400 text-sm">No providers yet. Use <strong>Near Me</strong> or search a city/ZIP.</div>`;
      return;
    }
    filtered.forEach(p => {
      const colorClass    = p.type === 'ER' ? 'bg-red-100 text-red-700' : p.type === 'Urgent Care' ? 'bg-blue-100 text-blue-700' : p.type === 'Hospital' ? 'bg-orange-100 text-orange-700' : p.type === 'Pharmacy' ? 'bg-green-100 text-green-700' : p.type === 'Dentist' ? 'bg-purple-100 text-purple-700' : p.type === 'Primary Care' ? 'bg-cyan-100 text-cyan-700' : p.type === 'Pediatrician' ? 'bg-pink-100 text-pink-700' : 'bg-slate-100 text-slate-700';
      const distText      = typeof p.distance === 'number' ? p.distance.toFixed(1) : '--';
      const directionsUrl = (p.lat != null && p.lon != null)
        ? `https://www.google.com/maps/dir/?api=1&destination=${p.lat},${p.lon}`
        : `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(p.name + ' ' + p.address)}`;
      const openChip = p.openNow === true ? `<span class="chip chip-green">OPEN</span>` : p.openNow === false ? `<span class="chip chip-red">CLOSED</span>` : '';
      list.innerHTML += `
        <div class="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow flex justify-between items-start gap-3">
          <div class="min-w-0">
            <div class="flex items-center gap-2 mb-1 flex-wrap">
              <span class="text-xs font-bold px-2 py-0.5 rounded-md ${colorClass}">${this.escapeHtml(p.type)}</span>
              ${openChip}
              ${p.rating ? `<span class="text-xs text-slate-400">★ ${this.escapeHtml(String(p.rating))}</span>` : ''}
            </div>
            <h3 class="font-bold text-slate-800 leading-tight text-sm">${this.escapeHtml(p.name)}</h3>
            <p class="text-slate-400 text-xs mb-2">${this.escapeHtml(p.address)}</p>
            <div class="flex gap-2">
              <a href="${directionsUrl}" target="_blank" rel="noopener noreferrer" class="text-xs bg-blue-50 text-blue-600 px-2.5 py-1 rounded-lg border border-blue-100 font-medium hover:bg-blue-100 flex items-center gap-1">
                <i data-lucide="navigation" class="w-3 h-3"></i> Directions
              </a>
            </div>
          </div>
          <div class="text-right shrink-0">
            <span class="block text-xl font-extrabold text-slate-900">${this.escapeHtml(distText)}</span>
            <span class="text-xs text-slate-400">mi</span>
          </div>
        </div>`;
    });
    if (window.lucide) lucide.createIcons();
  }

  /* -------------------- HELPERS -------------------- */
  escapeHtml(str) {
    return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }
  formatText(text) { return this.escapeHtml(text || '').replace(/\n/g, '<br>'); }
  formatMarkdown(text) {
    const safe = this.escapeHtml(text || '');
    let html   = safe
      .replace(/^#### (.*)$/gm, '<h4 class="text-sm font-bold text-slate-700 mt-3 mb-1">$1</h4>')
      .replace(/^### (.*)$/gm,  '<h3 class="text-base font-bold text-blue-800 mt-5 mb-2 pb-1 border-b border-blue-100">$1</h3>')
      .replace(/^## (.*)$/gm,   '<h2 class="text-lg font-bold text-slate-900 mt-5 mb-2">$1</h2>')
      .replace(/\*\*(.*?)\*\*/g,'<span class="font-semibold text-slate-900">$1</span>')
      .replace(/^\-{3,}$/gm,    '<hr class="my-4 border-slate-200">')
      .replace(/^\*⚕️(.*)\*$/gm, '<p class="text-xs text-slate-400 italic mt-4 pt-3 border-t border-slate-100">⚕️$1</p>')
      .replace(/^\*DISCLAIMER:(.*)\*$/gm, '<p class="text-xs text-slate-400 italic mt-4 border-t pt-2">$1</p>')
      .replace(/^\- (.*)$/gm,   '<li class="ml-4 list-disc mb-1.5 text-sm leading-relaxed">$1</li>')
      .replace(/\n/g, '<br>');
    if (html.includes('<li')) {
      html = html.replace(/(<li[\s\S]*?<\/li>)/g, '<ul class="my-2 space-y-0.5">$1</ul>').replace(/<\/ul><br><ul[^>]*>/g, '');
    }
    return html;
  }

  /* ------------------------------------------------------------------ */
  /* RX INIT                                                              */
  /* ------------------------------------------------------------------ */
  initRx() {
    this.rxMode         = 'upload';   // 'upload' | 'manual'
    this.rxImage        = null;       // base64 string of uploaded file
    this.rxIsPdf        = false;
    this.rxExtracted    = [];         // meds found by AI from image/PDF
    this.rxManualDraft  = [];         // meds added by manual form
    this.rxSaved        = JSON.parse(localStorage.getItem('pulsenova_rx') || '[]');
    this.rxSelectedDays = new Set();
    this._alexaPhrase   = '';

    // Wire up repeat select → show/hide custom days picker
    const repeatSel = document.getElementById('rx-manual-repeat');
    if (repeatSel) {
      repeatSel.addEventListener('change', () => {
        const daysWrap = document.getElementById('rx-manual-days');
        if (daysWrap) daysWrap.classList.toggle('hidden', repeatSel.value !== 'CUSTOM_DAYS');
      });
    }

    this.rxSetMode('upload');
    this._renderRxSaved();
  }

  /* ------------------------------------------------------------------ */
  /* TAB SWITCHING                                                        */
  /* ------------------------------------------------------------------ */
  rxSetMode(mode) {
    this.rxMode = mode;

    const uploadPane = document.getElementById('rx-pane-upload');
    const manualPane = document.getElementById('rx-pane-manual');
    const uploadTab  = document.getElementById('rx-tab-upload');
    const manualTab  = document.getElementById('rx-tab-manual');

    if (uploadPane) uploadPane.classList.toggle('hidden', mode !== 'upload');
    if (manualPane) manualPane.classList.toggle('hidden', mode !== 'manual');

    const active   = 'px-3 py-2 rounded-xl text-xs font-bold border flex items-center gap-2 transition-colors bg-sky-50 border-sky-300 text-sky-700';
    const inactive = 'px-3 py-2 rounded-xl text-xs font-bold border flex items-center gap-2 transition-colors bg-white border-slate-200 text-slate-700 hover:bg-slate-50';

    if (uploadTab) uploadTab.className = mode === 'upload' ? active : inactive;
    if (manualTab) manualTab.className = mode === 'manual' ? active : inactive;
  }

  /* ------------------------------------------------------------------ */
  /* FILE UPLOAD                                                          */
  /* ------------------------------------------------------------------ */
  handleRxUpload(input) {
    const file = input.files?.[0];
    if (!file) return;

    this.rxIsPdf = file.type === 'application/pdf';

    const reader = new FileReader();
    reader.onloadend = () => {
      this.rxImage = reader.result.split(',')[1];

      const previewImg  = document.getElementById('rx-preview-img');
      const previewPdf  = document.getElementById('rx-preview-pdf');
      const placeholder = document.getElementById('rx-placeholder');

      if (placeholder) placeholder.classList.add('hidden');

      if (this.rxIsPdf) {
        if (previewImg) previewImg.classList.add('hidden');
        if (previewPdf) previewPdf.classList.remove('hidden');
      } else {
        if (previewPdf) previewPdf.classList.add('hidden');
        if (previewImg) { previewImg.src = reader.result; previewImg.classList.remove('hidden'); }
      }

      const btn = document.getElementById('btn-extract-rx');
      if (btn) btn.disabled = false;
      this.toast('Prescription uploaded', 'file-scan');
    };
    reader.readAsDataURL(file);
  }

  /* ------------------------------------------------------------------ */
  /* EXTRACT FROM IMAGE / PDF VIA AI                                     */
  /* ------------------------------------------------------------------ */
  async extractPrescription() {
    if (!this.rxImage) return;

    const btn   = document.getElementById('btn-extract-rx');
    const badge = document.getElementById('rx-extract-badge');

    if (btn)   { btn.disabled = true; btn.innerHTML = `<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Extracting…`; }
    if (badge) { badge.textContent = 'Processing'; badge.className = 'chip chip-blue'; }

    const prompt = `You are a clinical pharmacist assistant. Extract ALL medications visible in this prescription.
        CRITICAL: Output ONLY a raw valid JSON array. No markdown fences, no preamble, no explanation.
        Each item must have exactly these keys:
          "name"   — medication name (string)
          "dose"   — dosage and unit e.g. "500 mg" (string, empty string if unknown)
          "times"  — array of 24-hour time strings e.g. ["08:00","20:00"] (empty array if unknown)
          "repeat" — one of exactly: "DAILY", "WEEKDAYS", "CUSTOM_DAYS"
          "days"   — array of day codes when repeat is CUSTOM_DAYS e.g. ["MO","WE","FR"], else []
          "notes"  — special instructions e.g. "Take with food" (string, empty string if none)
        If no medications are found, return [].`;

    try {
      const payload = this.rxIsPdf
        ? { pdf_base64: this.rxImage, prompt, language: this.selectedLanguage, max_tokens: 1500 }
        : { image_base64: this.rxImage, prompt, language: this.selectedLanguage, max_tokens: 1500 };

      const res = await fetch('/api/vision', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const raw  = (data.text || '').replace(/```json|```/g, '').trim();
      this.rxExtracted = JSON.parse(raw);
    } catch (e) {
      console.error('Rx extraction failed:', e);
      this.rxExtracted = [];
      this.toast('Could not extract medications. Try a clearer image.', 'alert-triangle');
    }

    this._renderRxExtracted();

    if (btn)   { btn.disabled = false; btn.innerHTML = `<i data-lucide="sparkles" class="w-4 h-4"></i> Extract Prescription`; }
    if (badge) {
      badge.textContent = this.rxExtracted.length ? `${this.rxExtracted.length} found` : 'None found';
      badge.className   = this.rxExtracted.length ? 'chip chip-green' : 'chip chip-amber';
    }
    if (window.lucide) lucide.createIcons();
  }

  _renderRxExtracted() {
    const list = document.getElementById('rx-extracted-list');
    if (!list) return;

    if (!this.rxExtracted.length) {
      list.innerHTML = `<div class="text-sm text-slate-400 text-center py-8">No medications found. Try a clearer image.</div>`;
      return;
    }

    list.innerHTML = '';
    this.rxExtracted.forEach((med, idx) => {
      const card = document.createElement('div');
      card.className = 'bg-slate-50 border border-slate-200 rounded-xl p-3 space-y-1';
      card.innerHTML = `
        <div class="flex items-start justify-between gap-2">
          <div>
            <div class="font-bold text-slate-800 text-sm">${this.escapeHtml(med.name)}</div>
            <div class="text-xs text-slate-500">${this.escapeHtml(med.dose)} · ${this.escapeHtml((med.times || []).join(', ') || '--')}</div>
          </div>
          <button onclick="app.rxRemoveExtracted(${idx})" class="text-slate-300 hover:text-red-400 transition-colors shrink-0">
            <i data-lucide="x" class="w-4 h-4"></i>
          </button>
        </div>
        ${med.notes ? `<div class="text-xs text-slate-400 italic">${this.escapeHtml(med.notes)}</div>` : ''}
        <div class="text-[10px] text-slate-400">Repeat: ${this.escapeHtml(med.repeat)}${med.days?.length ? ' · ' + med.days.join(', ') : ''}</div>`;
      list.appendChild(card);
    });
    if (window.lucide) lucide.createIcons();
  }

  rxRemoveExtracted(idx) {
    this.rxExtracted.splice(idx, 1);
    this._renderRxExtracted();
    const badge = document.getElementById('rx-extract-badge');
    if (badge) {
      badge.textContent = this.rxExtracted.length ? `${this.rxExtracted.length} found` : 'None found';
      badge.className   = this.rxExtracted.length ? 'chip chip-green' : 'chip chip-amber';
    }
  }

  /* ------------------------------------------------------------------ */
  /* MANUAL ENTRY                                                         */
  /* ------------------------------------------------------------------ */
  toggleRxDay(btn) {
    const day = btn.dataset.day;
    if (this.rxSelectedDays.has(day)) {
      this.rxSelectedDays.delete(day);
      btn.className = 'rx-day-btn px-3 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-50';
    } else {
      this.rxSelectedDays.add(day);
      btn.className = 'rx-day-btn px-3 py-2 rounded-xl border border-sky-400 bg-sky-50 text-xs font-semibold text-sky-700';
    }
  }

  addManualPrescription() {
    const name   = document.getElementById('rx-manual-name')?.value.trim();
    const dose   = document.getElementById('rx-manual-dose')?.value.trim()  || '';
    const times  = document.getElementById('rx-manual-times')?.value.trim() || '';
    const repeat = document.getElementById('rx-manual-repeat')?.value       || 'DAILY';
    const notes  = document.getElementById('rx-manual-notes')?.value.trim() || '';

    if (!name) { this.toast('Medication name is required.', 'alert-triangle'); return; }

    this.rxManualDraft.push({
      name,
      dose,
      times:  times ? times.split(',').map(t => t.trim()).filter(Boolean) : [],
      repeat,
      days:   repeat === 'CUSTOM_DAYS' ? Array.from(this.rxSelectedDays) : [],
      notes,
    });

    this._renderRxManualList();
    this.clearManualPrescription();
    this.toast(`${name} added to list`, 'check');
  }

  clearManualPrescription() {
    ['rx-manual-name', 'rx-manual-dose', 'rx-manual-times', 'rx-manual-notes'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = '';
    });
    const repeatSel = document.getElementById('rx-manual-repeat');
    if (repeatSel) repeatSel.value = 'DAILY';
    document.getElementById('rx-manual-days')?.classList.add('hidden');
    this.rxSelectedDays.clear();
    document.querySelectorAll('.rx-day-btn').forEach(btn => {
      btn.className = 'rx-day-btn px-3 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-50';
    });
  }

  _renderRxManualList() {
    const list = document.getElementById('rx-manual-list');
    if (!list) return;

    if (!this.rxManualDraft.length) {
      list.innerHTML = `<div class="text-sm text-slate-400 text-center py-8">Add a medication above to build your list.</div>`;
      return;
    }

    list.innerHTML = '';
    this.rxManualDraft.forEach((med, idx) => {
      const card = document.createElement('div');
      card.className = 'bg-slate-50 border border-slate-200 rounded-xl p-3 flex items-start justify-between gap-2';
      card.innerHTML = `
        <div class="min-w-0">
          <div class="font-bold text-slate-800 text-sm">${this.escapeHtml(med.name)}</div>
          <div class="text-xs text-slate-500">${this.escapeHtml(med.dose)} · ${this.escapeHtml((med.times || []).join(', ') || '--')}</div>
          ${med.notes ? `<div class="text-xs text-slate-400 italic mt-0.5">${this.escapeHtml(med.notes)}</div>` : ''}
          <div class="text-[10px] text-slate-400 mt-0.5">Repeat: ${this.escapeHtml(med.repeat)}${med.days?.length ? ' · ' + med.days.join(', ') : ''}</div>
        </div>
        <button onclick="app.rxRemoveManual(${idx})" class="text-slate-300 hover:text-red-400 transition-colors shrink-0">
          <i data-lucide="x" class="w-4 h-4"></i>
        </button>`;
      list.appendChild(card);
    });
    if (window.lucide) lucide.createIcons();
  }

  rxRemoveManual(idx) {
    this.rxManualDraft.splice(idx, 1);
    this._renderRxManualList();
  }

  /* ------------------------------------------------------------------ */
  /* SAVE TO LOCAL STORAGE                                                */
  /* ------------------------------------------------------------------ */
  savePrescriptions() {
    const toSave = [...this.rxExtracted, ...this.rxManualDraft];
    if (!toSave.length) { this.toast('No medications to save.', 'alert-triangle'); return; }

    const now = new Date().toLocaleDateString();
    let added = 0;
    toSave.forEach(med => {
      const exists = this.rxSaved.some(s => s.name === med.name && s.dose === med.dose);
      if (!exists) { this.rxSaved.push({ ...med, savedAt: now }); added++; }
    });

    localStorage.setItem('pulsenova_rx', JSON.stringify(this.rxSaved));
    this._renderRxSaved();
    this.toast(added ? `${added} medication(s) saved` : 'Already saved — no duplicates added', added ? 'check' : 'info');
  }

  loadPrescriptions() {
    this.rxSaved = JSON.parse(localStorage.getItem('pulsenova_rx') || '[]');
    this._renderRxSaved();
    this.toast('Prescriptions refreshed', 'refresh-cw');
  }

  _renderRxSaved() {
    const list = document.getElementById('rx-saved-list');
    if (!list) return;

    if (!this.rxSaved.length) {
      list.innerHTML = `<p class="text-sm text-slate-400 text-center py-6">No saved prescriptions yet.</p>`;
      return;
    }

    list.innerHTML = '';
    this.rxSaved.forEach((med, idx) => {
      const card = document.createElement('div');
      card.className = 'bg-slate-50 border border-slate-200 rounded-xl p-3 flex items-start justify-between gap-2';
      card.innerHTML = `
        <div class="flex-1 min-w-0">
          <div class="font-bold text-slate-800 text-sm">${this.escapeHtml(med.name)}</div>
          <div class="text-xs text-slate-500">${this.escapeHtml(med.dose)} · ${this.escapeHtml((med.times || []).join(', ') || '--')}</div>
          ${med.notes ? `<div class="text-xs text-slate-400 italic mt-0.5">${this.escapeHtml(med.notes)}</div>` : ''}
          <div class="text-[10px] text-slate-400 mt-0.5">
            Repeat: ${this.escapeHtml(med.repeat)}${med.days?.length ? ' · ' + med.days.join(', ') : ''}
            ${med.savedAt ? ` · Saved ${med.savedAt}` : ''}
          </div>
        </div>
        <button onclick="app.rxDeleteSaved(${idx})" class="text-slate-300 hover:text-red-400 transition-colors shrink-0 p-1">
          <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
        </button>`;
      list.appendChild(card);
    });
    if (window.lucide) lucide.createIcons();
  }

  rxDeleteSaved(idx) {
    this.rxSaved.splice(idx, 1);
    localStorage.setItem('pulsenova_rx', JSON.stringify(this.rxSaved));
    this._renderRxSaved();
    this.toast('Removed', 'trash');
  }

  /* ------------------------------------------------------------------ */
  /* ALEXA MODAL                                                          */
  /* ------------------------------------------------------------------ */
  openAlexaSendModal() {
    // Target the specific config container inside the modal
    const configEl = document.getElementById('alexa-reminder-config');
    const modal = document.getElementById('alexa-send-modal');
    
    if (!modal || !configEl) return;

    configEl.innerHTML = '';
    
    // Check if there are any saved prescriptions to display
    if (!this.rxSaved || this.rxSaved.length === 0) {
      configEl.innerHTML = '<div class="text-sm text-slate-400 text-center py-8">Add and save prescriptions first, then open this.</div>';
      modal.classList.remove('hidden');
      return;
    }

    // Generate an editable row for each saved prescription
    this.rxSaved.forEach((m, idx) => {
      // Convert array ["08:00", "20:00"] to a comma-separated string for easy editing
      const timeStr = (m.times && m.times.length) ? m.times.join(', ') : '08:00';
      
      const row = document.createElement('div');
      // Adding 'alexa-rx-item' and 'data-idx' so sendToAlexa can loop through them
      row.className = 'alexa-rx-item bg-white border border-slate-200 rounded-xl p-3 flex items-center justify-between gap-3 mb-2';
      row.dataset.idx = idx;
      
      row.innerHTML = `
        <div class="flex items-center gap-3 flex-1 min-w-0">
          <input type="checkbox" id="alexa-med-${idx}" class="alexa-rx-check w-4 h-4 accent-blue-600 shrink-0" checked>
          <label for="alexa-med-${idx}" class="min-w-0 cursor-pointer flex-1">
            <div class="font-semibold text-slate-800 text-sm truncate">${this.escapeHtml(m.name)}</div>
            <div class="text-xs text-slate-400">${this.escapeHtml(m.dose || '--')}</div>
          </label>
        </div>
        <div class="flex items-center gap-3">
          <div class="flex flex-col items-end">
            <label class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">Times (HH:MM)</label>
            <input type="text" class="alexa-rx-times border border-slate-200 rounded-lg px-2 py-1.5 text-sm w-32 outline-none focus:border-blue-500 text-right" value="${this.escapeHtml(timeStr)}" placeholder="08:00, 20:00">
          </div>
          <span class="chip chip-slate text-[10px] shrink-0 mt-4">${this.escapeHtml(m.repeat || 'DAILY')}</span>
        </div>
      `;
      configEl.appendChild(row);
    });

    modal.classList.remove('hidden');
    if (window.lucide) lucide.createIcons();
  }

  closeAlexaSendModal() {
    document.getElementById('alexa-send-modal')?.classList.add('hidden');
  }

  async sendToAlexa() {
    const btn = document.getElementById('btn-send-to-alexa');
    if (btn) { 
        btn.disabled = true; 
        btn.innerHTML = '<i data-lucide="loader" class="w-4 h-4 animate-spin"></i> Syncing to Database...'; 
    }

    const items = document.querySelectorAll('.alexa-rx-item');
    let updatedSaved = [...this.rxSaved];
    let selectedCount = 0;

    items.forEach(item => {
      const idx = parseInt(item.dataset.idx);
      const isChecked = item.querySelector('.alexa-rx-check').checked;
      const timesInput = item.querySelector('.alexa-rx-times').value;

      if (isChecked && updatedSaved[idx]) {
        selectedCount++;
        // Parse the comma-separated string and validate military time format (HH:MM)
        const rawTimes = timesInput.split(',').map(t => t.trim());
        const validTimes = rawTimes.filter(t => /^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/.test(t));
        
        // Update the array; fallback to 08:00 if the user typed nonsense
        updatedSaved[idx].times = validTimes.length > 0 ? validTimes : ["08:00"];
      }
    });

    if (selectedCount === 0) {
      this.toast('Select at least one medication.', 'alert-triangle');
      if (btn) { btn.disabled = false; btn.innerHTML = `<i data-lucide="send" class="w-4 h-4"></i> Send Request`; }
      return;
    }

    // 1. Update local browser state
    this.rxSaved = updatedSaved;
    localStorage.setItem('pulsenova_rx', JSON.stringify(this.rxSaved));

    // 2. Push the updated times to PostgreSQL
    await this._savePrescriptionsToDB(this.rxSaved);

    // 3. Update the visible saved list in the UI to reflect new times
    this._renderRxSaved();

    // 4. UI Feedback
    if (btn) { 
        btn.disabled = false; 
        btn.innerHTML = `<i data-lucide="send" class="w-4 h-4"></i> Send Request`; 
    }
    if (window.lucide) lucide.createIcons();

    this.closeAlexaSendModal();
    this.toast('Database updated!', 'check');
    setTimeout(() => this.toast('Say: "Alexa, ask Pulse Nova to sync my meds"', 'mic'), 1500);
  }

  copyAlexaPhrase() {
    const phrase = 'Alexa, ask Pulse Nova to sync my meds';
    navigator.clipboard?.writeText(phrase)
      .then(()  => this.toast('Copied to clipboard', 'copy'))
      .catch(() => this.toast('Copy failed — try manually', 'alert-triangle'));
  }

// =============================================================================
// BOOTSTRAP
// FIX: Full flow is now:
//  1. Instantiate app (isAuthenticated = false, safe default)
//  2. Navigate to home
//  3. Call pulseNovaInitAuthNav() which hits /me
//  4. Inside .then(): isAuthenticated is now correctly set on the instance
//  5. Only then call loadUserHistory() — guaranteed to run with the right value
// =============================================================================
const app = new PulseNovaApp();
window.app = app;
app.navigate('home');

pulseNovaInitAuthNav().then(() => {
  if (app.isAuthenticated && typeof app.loadUserHistory === 'function') {
    app.loadUserHistory();
  }
});

// Ensure icons render even on fresh browsers or slow connections
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) lucide.createIcons();
});

window.addEventListener('load', () => {
  if (window.lucide) lucide.createIcons();
});

// Fallback observer for dynamic content
const observer = new MutationObserver(() => {
  if (window.lucide) lucide.createIcons();
});
observer.observe(document.body, { childList: true, subtree: true });
