// professor.js — D.Sugisawa LT viewer.
// Runs in the browser (S3-hosted). All API calls are relative; the tunnel
// in front of S3 routes /chat /deck /config to your_professor_server.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  ensureGoogleSignIn,
  signInWithLab,
  getAuthHeaders,
  getUserIdMap,
  clearStoredSession,
} from "./auth.js";

// AbortError is our expected control-flow signal whenever the user pauses
// (Speak), raises a hand, or navigates mid-speech. It's caught at every
// site we care about, but Promise.race orphans, reader.cancel() rejection,
// and similar edges can still surface it as an unhandled rejection in
// devtools. Suppress AbortError specifically so it doesn't pollute the
// console — anything else falls through to the default handler.
window.addEventListener("unhandledrejection", (event) => {
  const r = event.reason;
  if (r && (r.name === "AbortError" || /aborted/i.test(r.message || ""))) {
    event.preventDefault();
  }
});

// ---------- session ids (filled in after Google Sign-In) ----------
// Each Lab keeps its own user registry, so we hold one user_id per Lab.
// USER_IDS = { lab_id: user_id }. The `getAuthHeaders` helper turns the
// map into a bundle of `X-User-Id-{lab_id}` headers; whichever Lab the
// proxy routes the request to picks its own header off the bundle.
let USER_IDS = {};
let DISPLAY_NAME = null;

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const elTitle = $("talk-title");
const elPresenter = $("presenter");
const elVenue = $("venue");
const elPageNum = $("page-num");
const elSlideTitle = $("slide-title");
const elBullets = $("slide-bullets");
const elStatus = $("status");
const elAvatar = $("avatar");

const btnPrev = $("btn-prev");
const btnSpeak = $("btn-speak");
const btnNext = $("btn-next");
const btnAsk = $("btn-ask");
const btnEnd = $("btn-end");
const btnLogout = $("btn-logout");
const inputQA = $("qa-input");
const btnUpload = $("btn-upload");
const btnMemo = $("btn-memo");
const fileInput = $("file-paper");
const memoOverlay = $("memo-overlay");
const memoTitle = $("memo-title");
const memoBody = $("memo-body");
const memoTarget = $("memo-target");
const memoStage = $("memo-stage");
const memoCancel = $("memo-cancel");
const memoSubmit = $("memo-submit");
const elSlide = $("slide");
const elSlideFrame = $("slide-frame");
const uploadOverlay = $("upload-overlay");
const uploadStage = $("upload-stage");
const uploadSpinner = $("upload-spinner");
const uploadFilename = $("upload-filename");
const uploadLog = $("upload-log");
const uploadClose = $("upload-close");
const uploadApply = $("upload-apply");
const uploadIndicator = $("upload-indicator");
const uploadIndicatorText = $("upload-indicator-text");
const elScriptIdx = $("script-idx");
const elScriptTitle = $("script-title");
const elScriptProgress = $("script-progress");
const elScriptLines = $("script-lines");
const paperLibrary = $("paper-library");
const btnPaperApply = $("btn-paper-apply");
const btnPaperDelete = $("btn-paper-delete");

// ---------- state ----------
const state = {
  deck: null,
  config: null,
  pageIndex: 0,           // 0 = before page 1 (waiting); 1..N = slide pages
  stage: "waiting",       // waiting | presenting | qa | closing
  busy: false,
  audioCtx: null,
  ttsGain: null,
  ttsSampleRate: 0,
  speaking: false,
  mixer: null,
  actions: { idleList: [], speakList: [], current: null },
  clock: new THREE.Clock(),
  // Per-slide speaker script (from generate_scripts.py), and which slide
  // index in the iframe is currently shown. When scriptSlides is present
  // the Speak button reads its lines line-by-line via TTS instead of
  // calling /api/chat.
  scriptSlides: null,     // [{index, title, lines:[str]}] or null
  scriptIndex: 1,         // 1-indexed; mirrors iframe.contentWindow.currentIndex+1
  scriptSpeakAbort: null, // AbortController to interrupt mid-line iteration
  scriptResumeAt: null,   // {slide, lineIdx} captured when a hand was raised
  handRaised: false,      // soft interrupt flag — checked between sentences
  speakActive: false,     // any TTS sequence is in flight; drives the
                          // Speak button's play/pause icon
  ackInFlight: false,     // hand-raise ack speech is playing — Speak
                          // clicks are ignored while true so a spam-press
                          // doesn't queue duplicate acks.
  qaInFlight: false,      // a QA exchange (LLM reply + its TTS) is being
                          // played — Speak is locked so the audience can't
                          // interrupt the answer with another hand-raise.
  currentLabId: null,     // Lab whose paper is currently applied. Drives
                          // single-target chat for non-qa stages so the LT
                          // narration stays on the originating backend even
                          // when other Labs are also toggled ON for Q&A.
};

function setStatus(s) { elStatus.textContent = s; }
function setBusy(b) {
  state.busy = b;
  // Speak is intentionally NOT disabled here — it's the play/pause toggle
  // and the user must always be able to press it to interrupt. 挙手 is
  // wired separately and likewise stays clickable mid-speech.
  for (const el of [btnPrev, btnNext, btnAsk, btnEnd, inputQA]) el.disabled = b;
  if (!b) updateNav();
}
// Speak button doubles as a play/pause toggle. The icon mirrors whether
// any TTS sequence is in flight (single utterance, per-line script, or
// auto-progress prompt). setSpeakActive() is the single mutation point.
const SPEAK_ICON_PLAY = "▶ Speak";
const SPEAK_ICON_PAUSE = "✋ 挙手";

function updateSpeakIcon() {
  btnSpeak.textContent = state.speakActive ? SPEAK_ICON_PAUSE : SPEAK_ICON_PLAY;
}

function setSpeakActive(b) {
  state.speakActive = !!b;
  updateSpeakIcon();
}

// Read the iframe slide state from the DOM rather than from globals on
// `contentWindow`. The generated deck declares `slides` and `currentIndex`
// with `const`/`let` at script scope so they don't attach to window.
// Only `showSlide` (a function declaration) is callable via `cw.showSlide`.
function _readIframeSlideState() {
  if (!elSlide.classList.contains("has-frame")) return null;
  const cw = elSlideFrame.contentWindow;
  if (!cw || !cw.document) return null;
  const slides = cw.document.querySelectorAll(".slide");
  if (!slides.length) return null;
  let cur = 0;
  for (let i = 0; i < slides.length; i++) {
    if (slides[i].style.display !== "none") { cur = i; break; }
  }
  return { total: slides.length, cur };
}

function updateNav() {
  if (state.busy) return;
  // In iframe mode, navigation bounds come from the iframe's own slide
  // count, not deck.json — and Speak is always allowed (it just reads
  // the current iframe slide's script if any).
  const iframeState = _readIframeSlideState();
  // While a hand-raise ack or a QA reply is in flight, the active flow
  // owns btnSpeak.disabled — leave it alone here so updateNav doesn't
  // race-flicker the button to "enabled" mid-cycle (the user sees a
  // "✋ 挙手 輝度" flash between the prior speak's setBusy(false) and the
  // ack's finally{}).
  const speakOwnedExternally = state.ackInFlight || state.qaInFlight;
  if (iframeState) {
    const { total, cur } = iframeState;
    btnPrev.disabled = total === 0 || cur <= 0;
    btnNext.disabled = total === 0 || cur >= total - 1;
    if (!speakOwnedExternally) btnSpeak.disabled = total === 0;
  } else {
    const pages = state.deck?.pages || [];
    btnPrev.disabled = state.pageIndex <= 1;
    btnNext.disabled = state.pageIndex >= pages.length;
    if (!speakOwnedExternally) {
      btnSpeak.disabled = state.pageIndex < 1 || state.pageIndex > pages.length;
    }
  }
  // Ask は Lab 選択 (≥1) を要件にする。論文未適用でも質問可。
  const hasLab = labState.selectedLabIds.size > 0;
  btnAsk.disabled = !hasLab;
  inputQA.disabled = !hasLab;
  btnEnd.disabled = false;
}

// ---------- deck rendering ----------
function renderPage(idx) {
  const pages = state.deck?.pages || [];
  if (idx === 0) {
    elPageNum.textContent = "= = = =";
    elSlideTitle.textContent = "= = = =";
    elBullets.innerHTML = "";
    return null;
  }
  if (idx < 1 || idx > pages.length) return null;
  const p = pages[idx - 1];
  elPageNum.textContent = `Slide ${p.page} / ${pages.length}`;
  elSlideTitle.textContent = p.title || "";
  elBullets.innerHTML = "";
  for (const b of (p.bullets || [])) {
    const li = document.createElement("li");
    li.textContent = b;
    elBullets.appendChild(li);
  }
  return p;
}

// ---------- API ----------
async function apiGet(path) {
  const r = await fetch(path, { headers: getAuthHeaders(USER_IDS) });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

// activeLabForStage(stage): which Lab to send a non-qa-fanout chat to.
// Priority: the Lab that owns the currently-applied paper (state.currentLabId,
// set by applyPaperById) so presenter narration stays on the right backend.
// Falls back to the single selected Lab when nothing applied yet, else null
// (no Lab selected → proxy default routing, which is fine for boot calls).
function activeLabForStage(_stage) {
  if (state.currentLabId && labState.selectedLabIds.has(state.currentLabId)) {
    return state.currentLabId;
  }
  return singleSelectedLab();
}

async function chat({ stage, slide = null, message = "", operatorId = null, groupId = "" }) {
  const opId = operatorId ?? activeLabForStage(stage);
  // /api/chat reads user_id from the body (not the header), so we must
  // send the uid that belongs to whichever Lab opId routes to.
  const labId = labState.labIdByOpId[opId] || null;
  const uid = labId ? USER_IDS[labId] : null;
  if (!uid) {
    throw new Error(`/api/chat aborted: no auth for lab op=${opId} lab=${labId || "?"}`);
  }
  const body = { user_id: uid, stage, message };
  if (slide) body.slide = slide;
  // Multi-Lab fan-out: same group_id sent to every Lab so the timeline
  // can re-mark ★ best across the group on reload (the in-memory badge
  // is otherwise lost with the page).
  if (groupId) body.group_id = groupId;
  const r = await fetch(withLab("/api/chat", opId), {
    method: "POST",
    headers: getAuthHeaders(USER_IDS),
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.text();
    throw new Error(`/api/chat ${r.status}: ${err}`);
  }
  return r.json();
}

// ---------- TTS ----------
// Shared AudioContext, allocated lazily with the first response's sample rate
// (matches the existing playGreetingTTS pattern). We don't close it between
// utterances so playback latency stays low.
async function getTTSAudioCtx(sampleRate) {
  if (!state.audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    state.audioCtx = new Ctx({ sampleRate });
    state.ttsGain = state.audioCtx.createGain();
    state.ttsGain.gain.value = 1.0;
    state.ttsGain.connect(state.audioCtx.destination);
    state.ttsSampleRate = sampleRate;
  }
  // resume() is async — must await it, otherwise on Safari/iOS the context
  // can still be in "suspended" state when source.start() runs and the
  // sample plays inaudibly (avatar mouth still animates because that's
  // a separate UI signal, but no audio reaches the device).
  if (state.audioCtx.state === "suspended") {
    try { await state.audioCtx.resume(); } catch (_) {}
  }
  return state.audioCtx;
}

async function unlockAudio() {
  // Must be called inside a user-gesture handler so AudioContext starts.
  // The resume() must be awaited (see getTTSAudioCtx) so the context is
  // fully running before any source.start() in the same call chain.
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!state.audioCtx) {
    state.audioCtx = new Ctx();
    state.ttsGain = state.audioCtx.createGain();
    state.ttsGain.gain.value = 1.0;
    state.ttsGain.connect(state.audioCtx.destination);
    state.ttsSampleRate = state.audioCtx.sampleRate;
  }
  if (state.audioCtx.state === "suspended") {
    try { await state.audioCtx.resume(); } catch (_) {}
  }
}

// Strip / normalize characters that the TTS engine reads literally and
// which break prosody (markdown asterisks become "asterisk asterisk",
// raw quote marks become "double quote", half-width parens stutter the
// flow, etc.). Half-width parens are mapped to full-width so the TTS
// treats them as the natural Japanese reading-pause they're meant to be.
function sanitizeForTTS(text) {
  if (!text) return "";
  return String(text)
    // markdown emphasis / inline code
    .replace(/[*＊]+/g, " ")
    .replace(/_{2,}/g, " ")
    .replace(/`+/g, " ")
    // bullet-like glyphs used as markdown list markers
    .replace(/[•◦▪▫·]/g, " ")
    // straight + curly + guillemet quotes — TTS reads them aloud
    .replace(/["“”„‟"']/g, " ")
    .replace(/[‘’‚‛«»『』「」]/g, " ")
    // half-width parens / brackets -> full-width so TTS treats them as
    // a natural pause instead of a literal "left parenthesis"
    .replace(/\(/g, "（")
    .replace(/\)/g, "）")
    .replace(/\[/g, "［")
    .replace(/\]/g, "］")
    .replace(/\{/g, "｛")
    .replace(/\}/g, "｝")
    // angle brackets / chevrons read literally too
    .replace(/[<>]/g, " ")
    // hash / pipe / tilde — markdown table & heading residue
    .replace(/[#|~]+/g, " ")
    // typographic marks the TTS reads literally as "section sign" / etc.
    .replace(/[§¶†‡°№]/g, " ")
    // currency glyphs — likewise read out word-for-word
    .replace(/[¥￥$＄€£₩￠￡￢￤]/g, " ")
    // Wide Unicode sweeps for symbol blocks that the TTS reads literally
    // as their character names (e.g. "Right arrow", "Black square",
    // "Heavy check mark"). These are blanket ranges — anything in here
    // is non-textual decoration that breaks the reading flow.
    .replace(/[←-⇿]/g, " ")  // arrows
    .replace(/[∀-⋿]/g, " ")  // mathematical operators (∑ ∏ ∫ ∈ ⊂ etc.)
    .replace(/[⌀-⏿]/g, " ")  // miscellaneous technical
    .replace(/[␀-␿]/g, " ")  // control pictures
    .replace(/[─-╿]/g, " ")  // box drawing
    .replace(/[▀-▟]/g, " ")  // block elements
    .replace(/[■-◿]/g, " ")  // geometric shapes (■ □ ▲ ● ◆ ★ etc.)
    .replace(/[☀-⛿]/g, " ")  // miscellaneous symbols (☆ ♠ ♥ ☎ ☞ etc.)
    .replace(/[✀-➿]/g, " ")  // dingbats (✓ ✗ ✦ ✪ etc.)
    .replace(/[⤀-⥿]/g, " ")  // supplemental arrows
    .replace(/[⬀-⯿]/g, " ")  // miscellaneous symbols and arrows
    // Astral / supplementary plane = emoji + math alphanumerics + …
    .replace(/[\uD83C-\uDBFF][\uDC00-\uDFFF]/g, " ")
    // Variation selectors + zero-width / bidi formatting
    .replace(/[︀-️]/g, "")
    .replace(/[​-‏‪-‮⁠-⁯]/g, "")
    // collapse runs of whitespace
    .replace(/\s+/g, " ")
    .trim();
}

// Break a long utterance into smaller TTS units that fit comfortably in
// one synth request. Sentence-end (。) is the primary boundary; for
// sentences longer than MAX_LEN we further split at clause-end (、) and
// re-merge tiny fragments so each chunk lands roughly TARGET_LEN chars.
function splitForTTS(text) {
  const MAX_LEN = 80;
  const TARGET_LEN = 40;
  if (!text) return [];
  const sentences = text.split(/(?<=[。．！!？?])/)
    .map((s) => s.trim()).filter(Boolean);
  const out = [];
  for (const sent of sentences) {
    if (sent.length <= MAX_LEN) {
      out.push(sent);
      continue;
    }
    const fragments = sent.split(/(?<=[、,])/)
      .map((s) => s.trim()).filter(Boolean);
    let buf = "";
    for (const f of fragments) {
      if (!buf) {
        buf = f;
      } else if (buf.length + f.length <= MAX_LEN || buf.length < TARGET_LEN) {
        buf += f;
      } else {
        out.push(buf);
        buf = f;
      }
    }
    if (buf) out.push(buf);
  }
  return out;
}

// Tracks the in-flight TTS so the audience-question button (挙手) can
// kill it mid-stream: the fetch reader is aborted and every scheduled
// AudioBufferSource gets stop()ed so the rest of the buffered audio
// doesn't keep playing in the background.
const speakRuntime = {
  abort: null,            // AbortController for the active fetch
  reader: null,           // ReadableStreamDefaultReader currently being drained
  sources: new Set(),     // BufferSources that have been start()ed and not yet ended
};

function stopAllSpeech() {
  // Cut the per-line script loop first so it doesn't queue another speak().
  if (state.scriptSpeakAbort) {
    try { state.scriptSpeakAbort.abort(); } catch (_) {}
  }
  cancelAuto();
  if (speakRuntime.abort) {
    try { speakRuntime.abort.abort(); } catch (_) {}
  }
  if (speakRuntime.reader) {
    // reader.cancel() returns a Promise that may reject with AbortError
    // when the underlying stream is already aborted; attach a catch to
    // swallow it so it doesn't surface as an unhandled rejection.
    try { speakRuntime.reader.cancel().catch(() => {}); } catch (_) {}
  }
  for (const src of speakRuntime.sources) {
    try { src.stop(0); } catch (_) {}
  }
  speakRuntime.sources.clear();
  setSpeakingAnim(false);
  setSpeakActive(false);
}

async function speak(text, voice) {
  text = sanitizeForTTS(text);
  if (!text) return;
  // Long utterances are sliced at sentence/clause boundaries and played
  // back-to-back. We await each chunk's stream so the next request only
  // fires after the prior buffer has fully drained — that matches the
  // user's "completion -> next request" requirement and keeps the TTS
  // engine from receiving anything too long in one shot.
  const chunks = splitForTTS(text);
  for (const chunk of chunks) {
    // 挙手 was pressed mid-utterance: stop queuing further chunks. The
    // caller's outer loop (speakScriptForIndex / runAutoProgress) will
    // observe handRaised and play the ack phrase.
    if (state.handRaised) break;
    await speakChunk(chunk, voice);
  }
}

// AbortError surfaces here whenever the user pauses (Speak / 挙手) mid-TTS.
// Safari's fetch reports it as "Load failed" without setting e.name, so we
// also pattern-match the message text.
function _isAbortError(e) {
  if (!e) return false;
  if (e.name === "AbortError") return true;
  const msg = (e.message || "").toLowerCase();
  return /abort/.test(msg) || msg === "load failed";
}

async function speakChunk(text, voice) {
  if (!text) return;
  const ttsUrl = state.config?.tts_url;
  if (!ttsUrl) {
    console.warn("no tts_url; skipping TTS");
    return;
  }

  setSpeakingAnim(true);

  // Single AbortController for the whole utterance: the 10 s connect
  // timer aborts on its own, and stopAllSpeech() aborts on user request.
  const abortCtrl = new AbortController();
  speakRuntime.abort = abortCtrl;
  const connectTimer = setTimeout(() => abortCtrl.abort(), 10000);

  let resp;
  try {
    resp = await fetch(ttsUrl, {
      method: "POST",
      headers: getAuthHeaders(USER_IDS),
      body: JSON.stringify({
        gender: voice || "male",
        style: "neutral",
        out_lang: "ja",
        text,
      }),
      signal: abortCtrl.signal,
    });
  } catch (e) {
    // AbortError here is expected when the user pauses (Speak / 挙手);
    // suppress so the console isn't spammed during normal control flow.
    if (!_isAbortError(e)) console.warn("tts fetch failed:", e.message);
    setSpeakingAnim(false);
    speakRuntime.abort = null;
    return;
  } finally {
    clearTimeout(connectTimer);
  }

  if (!resp.ok) {
    console.warn("tts non-200:", resp.status);
    setSpeakingAnim(false);
    speakRuntime.abort = null;
    return;
  }

  const sampleRate = parseInt(resp.headers.get("X-Sample-Rate") || "24000", 10);
  const audioCtx = await getTTSAudioCtx(sampleRate);
  const gainNode = state.ttsGain;

  const reader = resp.body.getReader();
  speakRuntime.reader = reader;
  let nextStartTime = 0;
  let leftover = null;

  try {
    while (true) {
      if (abortCtrl.signal.aborted) break;
      // Per-chunk read timeout: 10s. Clear the timer when read() wins so
      // the rejection Promise doesn't outlive the race as an orphan
      // (which would surface as an unhandled rejection later).
      let readTimeoutId;
      const timeoutPromise = new Promise((_, reject) => {
        readTimeoutId = setTimeout(
          () => reject(new Error("tts stream read timeout")),
          10000,
        );
      });
      let readResult;
      try {
        readResult = await Promise.race([reader.read(), timeoutPromise]);
      } finally {
        clearTimeout(readTimeoutId);
      }
      const { done, value } = readResult;
      if (done) break;
      if (abortCtrl.signal.aborted) break;

      // Carry odd-byte tail from previous chunk so Int16 alignment holds.
      let bytes;
      if (leftover) {
        bytes = new Uint8Array(leftover.length + value.length);
        bytes.set(leftover, 0);
        bytes.set(value, leftover.length);
        leftover = null;
      } else {
        bytes = value;
      }
      const usableLen = bytes.length & ~1;
      if (bytes.length > usableLen) leftover = bytes.slice(usableLen);
      if (usableLen === 0) continue;

      const int16 = new Int16Array(bytes.buffer, bytes.byteOffset, usableLen / 2);
      const float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

      const audioBuffer = audioCtx.createBuffer(1, float32.length, sampleRate);
      audioBuffer.getChannelData(0).set(float32);
      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(gainNode);
      source.onended = () => speakRuntime.sources.delete(source);

      const startTime = Math.max(audioCtx.currentTime, nextStartTime);
      source.start(startTime);
      speakRuntime.sources.add(source);
      nextStartTime = startTime + audioBuffer.duration;
    }
  } catch (e) {
    if (!_isAbortError(e)) console.warn("tts stream error:", e.message);
  } finally {
    // Wait for the last scheduled buffer to finish so the caller's
    // `await speak()` lines up with audible end — unless we got aborted,
    // in which case skip the wait so control returns immediately.
    if (!abortCtrl.signal.aborted) {
      const remaining = Math.max(0, nextStartTime - audioCtx.currentTime);
      if (remaining > 0) await new Promise((r) => setTimeout(r, remaining * 1000));
    }
    setSpeakingAnim(false);
    speakRuntime.abort = null;
    speakRuntime.reader = null;
  }
}

// ---------- GLB / Three.js ----------
function initAvatar() {
  const w = elAvatar.clientWidth || 320;
  const h = elAvatar.clientHeight || 480;

  const scene = new THREE.Scene();
  // Transparent canvas so the CSS-animated geometric backdrop shows through
  // wherever there is no model geometry.
  scene.background = null;

  const camera = new THREE.PerspectiveCamera(35, w / h, 0.1, 100);
  camera.position.set(0, 1.5, 2.6);
  camera.lookAt(0, 1.3, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  elAvatar.appendChild(renderer.domElement);

  // Soft studio: ambient floor + hemi tint + 3-point directionals at modest
  // intensities. Glossy materials (the jacket) blow out very fast on glTF
  // PBR, so keep the directionals down and lean on hemi/ambient.
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  scene.add(new THREE.HemisphereLight(0xfff6e0, 0x9aa6c8, 1.2));

  const keyLight = new THREE.DirectionalLight(0xfff2d4, 0.9);
  keyLight.position.set(2.5, 4, 3);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xcfe1ff, 0.5);
  fillLight.position.set(-3, 2, 2);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
  rimLight.position.set(0, 4, -3);
  scene.add(rimLight);

  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  const loader = new GLTFLoader();
  loader.load(
    "/model/professor.glb",
    (gltf) => {
      scene.add(gltf.scene);

      // Force fabric-like matte on every PBR mesh. The model ships with
      // low roughness + non-zero metalness on the shirt/jacket which reads
      // as polished metal under any direct light. Pin roughness to fully
      // rough and metalness to zero to kill the specular sheen entirely.
      gltf.scene.traverse((obj) => {
        if (!obj.isMesh) return;
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const m of mats) {
          if (!m) continue;
          if (m.roughness !== undefined) m.roughness = 1.0;
          if (m.metalness !== undefined) m.metalness = 0.0;
          if (m.envMapIntensity !== undefined) m.envMapIntensity = 0.0;
          // Some glTF authors bake a roughness/metalness texture; without
          // disabling these maps, the per-pixel values override the scalar
          // we just set and the sheen comes back.
          if (m.metalnessMap) { m.metalnessMap = null; }
          if (m.roughnessMap) { m.roughnessMap = null; }
          m.needsUpdate = true;
        }
      });

      const clips = gltf.animations || [];
      console.log("GLB animations:", clips.map((c) => c.name));
      if (!clips.length) return;

      const mixer = new THREE.AnimationMixer(gltf.scene);
      state.mixer = mixer;

      const byName = (name) => clips.find((c) => c.name === name);

      // Both idle and speaking are random-chained one-shots, but draw from
      // different pools. Speaking pool = idles + acks so the "talking" gesture
      // varies across clips. Idle pool = idles only.
      const idleClips = clips.filter((c) => /^idle\d+$/.test(c.name));
      const ackClips = clips.filter((c) => /^ack\d+$/.test(c.name));
      const seedIdles = idleClips.length ? idleClips : [clips[0]];

      const makeAction = (clip) => {
        // mixer.clipAction caches by clip, so the same clip in both pools
        // returns the same AnimationAction (intentional — single source of
        // truth for that clip's playback state).
        const a = mixer.clipAction(clip);
        a.loop = THREE.LoopOnce;
        a.clampWhenFinished = true;
        return a;
      };

      state.actions.idleList = seedIdles.map(makeAction);
      state.actions.speakList = [...seedIdles, ...ackClips].map(makeAction);

      mixer.addEventListener("finished", (e) => {
        if (e.action !== state.actions.current) return;
        const pool = state.speaking ? state.actions.speakList : state.actions.idleList;
        pickFromPool(pool);
      });

      pickFromPool(state.actions.idleList);
    },
    undefined,
    (err) => console.warn("GLB load failed", err),
  );

  function onResize() {
    const ww = elAvatar.clientWidth;
    const hh = elAvatar.clientHeight;
    if (!ww || !hh) return;
    camera.aspect = ww / hh;
    camera.updateProjectionMatrix();
    renderer.setSize(ww, hh);
  }
  window.addEventListener("resize", onResize);
  onResize();

  function tick() {
    requestAnimationFrame(tick);
    const dt = state.clock.getDelta();
    if (state.mixer) state.mixer.update(dt);
    renderer.render(scene, camera);
  }
  tick();
}

function pickFromPool(pool) {
  if (!pool || !pool.length) return;
  let next;
  if (pool.length === 1) {
    next = pool[0];
  } else {
    do {
      next = pool[Math.floor(Math.random() * pool.length)];
    } while (next === state.actions.current);
  }
  const prev = state.actions.current;
  if (prev && prev !== next) prev.fadeOut(0.25);
  next.reset().fadeIn(0.25).play();
  state.actions.current = next;
}

function setSpeakingAnim(on) {
  state.speaking = on;
  elAvatar.classList.toggle("speaking", on);
  // Switch pools by picking a fresh clip from the new pool. The mixer's
  // 'finished' event will then keep chaining within that pool.
  pickFromPool(on ? state.actions.speakList : state.actions.idleList);
}

// ---------- flow ----------
async function doStage(stage, opts = {}) {
  setBusy(true);
  setStatus(`stage=${stage}…`);
  state.stage = stage;
  setSpeakActive(true);
  try {
    const reply = await chat({ stage, slide: opts.slide || null, message: opts.message || "" });
    setStatus(`stage=${stage} ✓`);
    if (reply?.reply) {
      await speak(reply.reply, reply.voice);
    }
  } catch (e) {
    console.error(e);
    setStatus(`error: ${e.message}`);
  } finally {
    setBusy(false);
    setSpeakActive(false);
  }
}

function currentSlidePayload() {
  const pages = state.deck?.pages || [];
  if (state.pageIndex < 1 || state.pageIndex > pages.length) return null;
  const p = pages[state.pageIndex - 1];
  return { page: p.page, title: p.title, bullets: p.bullets || [], notes: p.notes || "" };
}

async function gotoPage(idx) {
  cancelAuto();   // any pending auto chain belongs to the previous slide
  const pages = state.deck?.pages || [];
  if (idx < 0 || idx > pages.length) return;
  state.pageIndex = idx;
  renderPage(idx);
  // Page navigation alone does NOT trigger TTS — the user has to press
  // Speak (▶) explicitly. Speak is the single play / pause control.
}

// ---------- buttons ----------
// ---------- auto-progression ----------
// After each slide finishes speaking we proactively prompt for questions,
// pause, and (if no one interrupts) auto-advance to the next slide.
// Speak (= 挙手) during this cycle: cancelAuto() sets autoActive=false and
// resolves any pending autoSleep, the next `if (!autoActive) return;`
// guard exits the chain, and the click handler itself speaks HAND_ACK.
const AUTO_PROMPT_TEXT = "ここまでで、コメント、質問等、ございましたら、お願いいたします。";
const AUTO_PROCEED_TEXT = "なさそうでしたら、次に進めたいと思います。";
const AUTO_WAIT_QA_MS = 8000;     // pause after the QA prompt
const AUTO_WAIT_FINAL_MS = 3000;  // pause after announcing we're moving on

let autoActive = false;
let autoTimer = null;
let autoSleepResolve = null;

function cancelAuto() {
  autoActive = false;
  if (autoTimer) {
    clearTimeout(autoTimer);
    autoTimer = null;
  }
  // Wake any pending autoSleep so runAutoProgress can fall through to its
  // next autoActive guard and exit cleanly. Without this the sleep promise
  // hangs forever and Speak (= 挙手) silently breaks the loop.
  if (autoSleepResolve) {
    const r = autoSleepResolve;
    autoSleepResolve = null;
    r();
  }
}

function autoSleep(ms) {
  return new Promise((resolve) => {
    autoSleepResolve = resolve;
    autoTimer = setTimeout(() => {
      autoTimer = null;
      autoSleepResolve = null;
      resolve();
    }, ms);
  });
}

async function runAutoProgress() {
  // Don't run for waiting / qa / closing, only for presenting transitions.
  // Don't run past the last slide either — there's nowhere to advance to.
  // In iframe mode the iframe owns the slide count, otherwise deck.json.
  const st = _readIframeSlideState();
  if (st) {
    if (st.total === 0 || st.cur >= st.total - 1) return;
  } else {
    const pages = state.deck?.pages || [];
    if (state.pageIndex < 1 || state.pageIndex >= pages.length) return;
  }
  if (autoActive) return;
  autoActive = true;
  // Show the "✋ 挙手" icon throughout the auto cycle (including the silent
  // autoSleep windows) so a Speak click during this whole period is
  // routed to the hand-raise path, not the play/resume path.
  setSpeakActive(true);
  try {
    if (!autoActive) return;
    await speak(AUTO_PROMPT_TEXT, "male");
    if (!autoActive) return;
    await autoSleep(AUTO_WAIT_QA_MS);
    if (!autoActive) return;
    await speak(AUTO_PROCEED_TEXT, "male");
    if (!autoActive) return;
    await autoSleep(AUTO_WAIT_FINAL_MS);
    if (!autoActive) return;
    autoActive = false;
    if (btnNext.disabled) return;
    btnNext.click();
    // btnNext only flips the page (manual Next intentionally doesn't speak).
    // Auto-progression must keep talking — speak the new slide's script,
    // then chain into the next auto cycle.
    if (state.scriptSlides && _scriptForIndex(state.scriptIndex)) {
      setBusy(true);
      state.stage = "presenting";
      try {
        await speakScriptForIndex(state.scriptIndex);
      } finally {
        setBusy(false);
      }
      // Same guard as btnSpeak: don't recurse into the next auto cycle if
      // the user paused via Speak (= 挙手) mid-slide — that would
      // force-advance past where they wanted to stop.
      if (!state.scriptResumeAt) {
        runAutoProgress();
      }
    }
  } catch (e) {
    console.warn("[auto] error:", e.message);
    autoActive = false;
  } finally {
    // Only clear the icon if neither speakScriptForIndex (which manages
    // its own setSpeakActive) nor a chained runAutoProgress is still
    // running. autoActive=false + nothing speaking → idle, show ▶.
    if (!autoActive && !state.scriptSpeakAbort) {
      setSpeakActive(false);
    }
  }
}

// ---------- presence + qa timeline ----------
const POLL_INTERVAL_MS = 10_000;
let pollTimer = null;
// Per-Lab cursor: each Lab assigns qa_ids independently, so a single
// global cursor would skip items from late-joining Labs. Cleared by
// toggleLab so a freshly-selected Lab pulls from 0.
const lastQaIdByLab = {};

const elParticipantsList = document.getElementById("participants-list");
const elParticipantsCount = document.getElementById("participants-count");
const elQaBody = document.getElementById("qa-body");
const elQaCount = document.getElementById("qa-count");
const elLabsList = document.getElementById("labs-list");
const elLabsCount = document.getElementById("labs-count");

// Multi-lab fan-out: the user can toggle multiple Labs ON. Paper library
// fans out to each selected Lab in parallel and groups results by Lab in
// the dropdown. Upload requires exactly one selected Lab so the target is
// unambiguous. labels[] caches lab display names so groups stay labeled
// even if the dropdown is rendered before the next refreshLabs tick.
const labState = {
  selectedLabIds: new Set(),
  labels: {},        // operator_id -> human label
  labIdByOpId: {},   // operator_id -> lab_id (used to pick the right uid)
  // Per-Lab auth status, keyed by operator_id:
  //   "pending" — sign-in attempt in flight
  //   "ok"      — Google sign-in registered/verified on that Lab
  //   "failed"  — Lab rejected (e.g. domain not allowed) or unreachable
  // Only "ok" Labs are selectable in the GUI; the rest stay visible but
  // grayed-out so the user can see they exist without being able to
  // dispatch QA fan-outs that would just 401.
  authStatus: {},
};

// Promises so concurrent refreshLabs ticks don't double-fire signInWithLab
// for the same op while an attempt is in flight.
const _labAuthInFlight = {};

function _kickLabAuth(opId, labId) {
  if (!opId || !labId) return;
  if (_labAuthInFlight[opId]) return;
  if (labState.authStatus[opId] === "ok") return;
  labState.authStatus[opId] = "pending";
  _labAuthInFlight[opId] = (async () => {
    try {
      const res = await signInWithLab({ opId, labId });
      if (res && res.userId) {
        USER_IDS[labId] = res.userId;
        if (res.displayName && !DISPLAY_NAME) DISPLAY_NAME = res.displayName;
        labState.authStatus[opId] = "ok";
        return true;
      }
      labState.authStatus[opId] = "failed";
      return false;
    } catch (e) {
      console.warn(`[lab-auth] sign-in failed op=${opId} lab=${labId}:`, e.message);
      labState.authStatus[opId] = "failed";
      return false;
    } finally {
      delete _labAuthInFlight[opId];
    }
  })();
}

// Append operator_id=<id> to a /api/* path so the proxy routes the request
// to that Lab's tunnel. Pass the operator_id explicitly — callers either
// know which Lab (per-paper-row actions) or use singleSelectedLab() for
// upload-style "one-target" calls.
function withLab(path, operatorId) {
  if (!operatorId) return path;
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}operator_id=${encodeURIComponent(operatorId)}`;
}

// Returns the single selected Lab id, or null if 0 or >1 are selected.
// Used by the upload flow which can't disambiguate which Lab to target.
function singleSelectedLab() {
  return labState.selectedLabIds.size === 1
    ? Array.from(labState.selectedLabIds)[0]
    : null;
}

// Encode/decode the dropdown's option value. Each option carries both the
// owning Lab and the job_id so the apply/delete handlers can route
// per-paper rather than per-(globally selected)-Lab.
const PAPER_KEY_SEP = "|";
function paperKey(operatorId, jobId) { return `${operatorId}${PAPER_KEY_SEP}${jobId}`; }
function parsePaperKey(value) {
  if (!value) return null;
  const idx = value.indexOf(PAPER_KEY_SEP);
  if (idx < 0) return null;
  return { operator_id: value.slice(0, idx), job_id: value.slice(idx + 1) };
}

// Hamburger accordion: toggles `.collapsed` on the parent .panel.
// Default state lives in the HTML class list (panels start collapsed).
for (const panel of document.querySelectorAll("#side .panel[data-collapsible]")) {
  const ham = panel.querySelector("h3 .ham");
  if (!ham) continue;
  ham.addEventListener("click", () => {
    const nowCollapsed = panel.classList.toggle("collapsed");
    ham.setAttribute("aria-expanded", String(!nowCollapsed));
  });
}

async function heartbeat() {
  // Multi-Lab presence: send one heartbeat per authenticated Lab, each
  // carrying that Lab's own user_id in the body and routed via
  // ?operator_id= so it lands on the right backend. Without explicit
  // routing the proxy would default-route them all to one Lab and the
  // others would mark us as gone.
  const targets = [];
  for (const [opId, labId] of Object.entries(labState.labIdByOpId)) {
    const uid = USER_IDS[labId];
    if (uid) targets.push({ opId, uid });
  }
  if (!targets.length) return;
  await Promise.all(targets.map(async ({ opId, uid }) => {
    try {
      await fetch(withLab("/api/presence/heartbeat", opId), {
        method: "POST",
        headers: getAuthHeaders(USER_IDS),
        body: JSON.stringify({ user_id: uid }),
      });
    } catch (e) {
      console.warn(`[presence] heartbeat failed (op=${opId}):`, e.message);
    }
  }));
}

function _isMe(uid) {
  // With one user_id per Lab, "me" is anyone whose uid matches *any*
  // entry in our per-Lab map.
  if (!uid) return false;
  for (const v of Object.values(USER_IDS)) {
    if (v === uid) return true;
  }
  return false;
}

async function refreshParticipants() {
  // Fan-out: each Lab keeps its own participants list with its own per-Lab
  // user_ids, so the same Google account shows up as a different uid on
  // every Lab. Pull from every authed Lab in parallel and merge by
  // email_local (Google's stable identity prefix) — falling back to
  // display_name + uid when a Lab somehow stripped the email.
  const targets = Object.entries(labState.labIdByOpId)
    .filter(([opId]) => labState.authStatus[opId] === "ok")
    .map(([opId]) => opId);
  if (!targets.length) {
    elParticipantsCount.textContent = "0";
    elParticipantsList.innerHTML = "";
    return;
  }
  const responses = await Promise.all(targets.map(async (opId) => {
    try {
      const r = await fetch(
        withLab("/api/presence/users", opId),
        { headers: getAuthHeaders(USER_IDS) },
      );
      if (!r.ok) return null;
      return await r.json();
    } catch (e) {
      console.warn(`[presence] users fetch failed (op=${opId}):`, e.message);
      return null;
    }
  }));

  // Identity merge: same email_local => same person across Labs.
  // Merge rules:
  //   online   = OR across Labs (online on any one ⇒ shown online)
  //   joined_at = min  (earliest join wins)
  //   last_seen = max  (most recent ping wins)
  //   qa_count = sum   (questions across all Labs the user used)
  //   user_ids = the per-Lab uids we saw, used by _isMe to mark "me".
  const merged = new Map();   // key -> aggregated row
  for (const data of responses) {
    if (!data || !Array.isArray(data.users)) continue;
    for (const u of data.users) {
      const key = u.email_local
        ? `email:${u.email_local}`
        : `uid:${u.user_id}`;
      const cur = merged.get(key);
      if (!cur) {
        merged.set(key, {
          email_local: u.email_local || "",
          display_name: u.display_name || u.user_id.slice(0, 8),
          online: !!u.online,
          joined_at: u.joined_at || 0,
          last_seen: u.last_seen || 0,
          qa_count: u.qa_count || 0,
          user_ids: [u.user_id],
        });
      } else {
        cur.online = cur.online || !!u.online;
        if (u.joined_at && (!cur.joined_at || u.joined_at < cur.joined_at)) {
          cur.joined_at = u.joined_at;
        }
        if (u.last_seen && u.last_seen > cur.last_seen) {
          cur.last_seen = u.last_seen;
        }
        cur.qa_count += (u.qa_count || 0);
        if (!cur.user_ids.includes(u.user_id)) cur.user_ids.push(u.user_id);
        if (!cur.display_name && u.display_name) cur.display_name = u.display_name;
      }
    }
  }

  const rows = Array.from(merged.values()).sort((a, b) => a.joined_at - b.joined_at);
  const onlineCount = rows.filter((r) => r.online).length;
  elParticipantsCount.textContent = `${onlineCount}/${rows.length}`;
  elParticipantsList.innerHTML = "";
  for (const u of rows) {
    const row = document.createElement("div");
    const cls = ["user-row"];
    if (u.user_ids.some((id) => _isMe(id))) cls.push("me");
    if (!u.online) cls.push("offline");
    row.className = cls.join(" ");

    const dot = document.createElement("div"); dot.className = "dot";

    const wrap = document.createElement("div"); wrap.className = "name-wrap";
    const name = document.createElement("div"); name.className = "name";
    name.textContent = u.display_name;
    wrap.appendChild(name);
    if (u.email_local) {
      const sub = document.createElement("div");
      sub.className = "name-sub";
      sub.textContent = `(${u.email_local})`;
      wrap.appendChild(sub);
    }
    // 3rd line: how many Labs this person is authenticated against.
    // user_ids.length counts per-Lab uids we saw across the merged
    // responses — i.e. one entry per Lab that has them registered.
    const labsLine = document.createElement("div");
    labsLine.className = "name-sub";
    labsLine.textContent = `認証済み Lab: ${u.user_ids.length}`;
    wrap.appendChild(labsLine);
    row.append(dot, wrap);
    elParticipantsList.appendChild(row);
  }
}

function renderQaItem(item) {
  // item is one Lab's QA record, decorated with _opId by refreshQaTimeline.
  const div = document.createElement("div");
  div.className = "qa-item";
  // Composite key: qa_id is per-Lab so we prefix with the Lab id to keep
  // the dataset key globally unique within the merged DOM.
  div.dataset.qaId = `${item._opId}:${item.qa_id}`;
  div.dataset.opId = item._opId || "";
  // group_id correlates fan-out responses across Labs; recomputeBestBadges
  // groups by it and marks the highest-scoring item ★ best. Same scoring
  // formula as the server-less prior implementation:
  //   score = top_score * 1.0 + (answer_len / 1000) * 0.1
  // top_score (RAG match strength) dominates; answer length only breaks
  // ties. Stash the inputs on the element so the recompute is data-only.
  div.dataset.groupId = item.group_id || "";
  div.dataset.topScore = String(Number(item.accuracy?.top_score) || 0);
  div.dataset.answerLen = String((item.answer || "").length);

  const meta = document.createElement("div");
  meta.className = "meta";
  const labLabel = labState.labels[item._opId] || (item._opId || "").slice(0, 6) || "Lab";
  const badge = document.createElement("span");
  badge.className = "lab-badge";
  badge.textContent = labLabel;
  badge.title = `Lab: ${labLabel}`;
  const who = document.createElement("span"); who.className = "who";
  who.textContent = item.display_name;
  const when = document.createElement("span");
  when.textContent = new Date(item.ts * 1000).toLocaleTimeString();
  meta.append(badge, who, when);
  if (item.slide_page) {
    const sp = document.createElement("span"); sp.textContent = `slide ${item.slide_page}`;
    meta.append(sp);
  }
  // Accuracy badge: server packs {rag_hits, top_score, mean_score}. We
  // surface mean_score (overall match) + hits count; top_score lives in
  // the tooltip for users who care about the strongest grounding chunk.
  const acc = item.accuracy || {};
  if (acc.rag_hits || acc.mean_score) {
    const ab = document.createElement("span");
    ab.className = "acc-badge";
    const mean = (acc.mean_score || 0).toFixed(2);
    ab.textContent = `🎯 ${mean} / ${acc.rag_hits || 0}`;
    ab.title = `mean_score=${(acc.mean_score || 0).toFixed(3)}, top_score=${(acc.top_score || 0).toFixed(3)}, hits=${acc.rag_hits || 0}`;
    meta.append(ab);
  }
  // Per-item Speak: only the user-clicked answer plays via TTS, never an
  // auto-play after fan-out. (Single-Lab path keeps its existing auto-TTS
  // in btnAsk; this button is the multi-Lab "pick which to speak".)
  const speakBtn = document.createElement("button");
  speakBtn.className = "qa-speak";
  speakBtn.type = "button";
  speakBtn.textContent = "🔊";
  speakBtn.title = "この回答を読み上げる";
  speakBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!item.answer) return;
    speakBtn.disabled = true;
    try {
      await unlockAudio();
      await speak(item.answer, "male");
    } catch (err) {
      console.warn("[qa] speak failed:", err.message);
    } finally {
      speakBtn.disabled = false;
    }
  });
  meta.append(speakBtn);

  const q = document.createElement("div"); q.className = "q";
  q.textContent = item.question;
  const a = document.createElement("div"); a.className = "a";
  a.textContent = item.answer;
  div.append(meta, q, a);
  return div;
}

// Mutex: fanOutQa awaits refreshQaTimeline right after a question is sent,
// but pollOnce also fires it on a 10s tick. Without a lock, two concurrent
// calls both read the same pre-update lastQaIdByLab[opId] as `start`, both
// fetch the same items, and both append → visible duplicates in the DOM.
let _qaRefreshInFlight = null;
async function refreshQaTimeline() {
  if (_qaRefreshInFlight) return _qaRefreshInFlight;
  _qaRefreshInFlight = (async () => {
    // Per-Lab fan-out + merge by ts. With multi-select, each Lab keeps its
    // own qa_id sequence and own /api/qa/timeline; the GUI fetches each in
    // parallel, decorates items with their owning op_id, and inserts them
    // into the merged DOM in chronological order.
    const labs = Array.from(labState.selectedLabIds);
    if (!labs.length) return;
    const fetched = await Promise.all(labs.map(async (opId) => {
      try {
        const start = lastQaIdByLab[opId] || 0;
        const url = withLab(
          `/api/qa/timeline?start=${start}&end=-1&limit=50`, opId);
        const r = await fetch(url, { headers: getAuthHeaders(USER_IDS) });
        if (!r.ok) return [];
        const data = await r.json();
        const items = (data.items || []).map((it) => ({ ...it, _opId: opId }));
        for (const it of items) {
          lastQaIdByLab[opId] = Math.max(lastQaIdByLab[opId] || 0, it.qa_id);
        }
        return items;
      } catch (e) {
        console.warn(`[qa] timeline fetch failed for op=${opId}:`, e.message);
        return [];
      }
    }));
    const allNew = fetched.flat().sort((a, b) => a.ts - b.ts);
    if (!allNew.length) return;
    const empty = elQaBody.querySelector(".empty");
    if (empty) empty.remove();
    for (const item of allNew) {
      // Belt-and-suspenders: even with the mutex, a stale fanOutQa→star
      // tagging path could still try to render an item already in the DOM
      // if the user re-asks before the previous call's mutex releases on
      // network failure. Skip if the composite key already exists.
      const key = `${item._opId}:${item.qa_id}`;
      if (elQaBody.querySelector(
            `.qa-item[data-qa-id="${CSS.escape(key)}"]`)) continue;
      elQaBody.appendChild(renderQaItem(item));
    }
    // Re-derive ★ best across every group_id currently in the DOM. This
    // is what makes the badge survive reload: server returns group_id,
    // we recompute, no in-memory state lost between page loads.
    recomputeBestBadges();
    elQaBody.scrollTop = elQaBody.scrollHeight;
    elQaCount.textContent =
      elQaBody.querySelectorAll(".qa-item:not(.empty)").length;
  })();
  try {
    return await _qaRefreshInFlight;
  } finally {
    _qaRefreshInFlight = null;
  }
}

// Walk every qa-item in the DOM, group by data-group-id, mark the
// highest-scoring item per group as ★ best. Idempotent: re-running clears
// stale badges from items that lost the lead (e.g. a late competing
// response just landed). Single-Lab questions carry no group_id (or a
// group of size 1) and are skipped — best only makes sense across rivals.
function recomputeBestBadges() {
  const groups = new Map();
  for (const el of elQaBody.querySelectorAll(".qa-item:not(.empty)")) {
    const gid = el.dataset.groupId;
    if (!gid) continue;
    const top = Number(el.dataset.topScore) || 0;
    const len = Number(el.dataset.answerLen) || 0;
    const score = top * 1.0 + (len / 1000) * 0.1;
    if (!groups.has(gid)) groups.set(gid, []);
    groups.get(gid).push({ el, score });
  }
  for (const arr of groups.values()) {
    if (arr.length < 2) {
      for (const x of arr) {
        x.el.classList.remove("is-best");
        x.el.querySelector(".best-badge")?.remove();
      }
      continue;
    }
    let best = arr[0];
    for (const x of arr) if (x.score > best.score) best = x;
    for (const x of arr) {
      const isBest = (x === best);
      x.el.classList.toggle("is-best", isBest);
      const existing = x.el.querySelector(".best-badge");
      if (isBest && !existing) {
        const badge = document.createElement("span");
        badge.className = "best-badge";
        badge.textContent = "★ best";
        badge.title = `score=${best.score.toFixed(3)}`;
        x.el.querySelector(".meta")?.appendChild(badge);
      } else if (!isBest && existing) {
        existing.remove();
      }
    }
  }
}

function _formatTimestamp(unix) {
  if (!unix) return "—";
  const d = new Date(unix * 1000);
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleDateString();
}

async function refreshLabs() {
  // /api/tunnel/info is served directly by the proxy (not forwarded through
  // the tunnel) — no auth header needed. operator.lab is populated by the
  // tunnel_client's post-register update_operator push, so a freshly-
  // connected operator may show "(準備中…)" until its first push lands.
  try {
    const r = await fetch("/api/tunnel/info");
    if (!r.ok) return;
    const data = await r.json();
    const ops = Array.isArray(data.operators) ? data.operators : [];
    elLabsCount.textContent = ops.length;
    elLabsList.innerHTML = "";

    // Refresh the label + lab_id caches so paperLibrary's optgroup labels
    // stay in sync even if it's rendered between refreshLabs ticks, and so
    // chat()/heartbeat() can map opId → lab_id → uid.
    const newLabels = {};
    const newLabIdByOpId = {};
    for (const op of ops) {
      const lab = op.lab || {};
      newLabels[op.id] = lab.name || lab.lab_id || op.name || op.id?.slice(0, 6) || "Lab";
      if (lab.lab_id) newLabIdByOpId[op.id] = lab.lab_id;
    }
    labState.labels = newLabels;
    labState.labIdByOpId = newLabIdByOpId;

    // Per-Lab Google sign-in: try every visible Lab once. The Lab list
    // is shown regardless of auth state, but a Lab the user couldn't
    // authenticate to (e.g. it's offline now, or its users.json is
    // gone) stays non-selectable below.
    for (const op of ops) {
      const labId = newLabIdByOpId[op.id];
      if (!labId) continue;
      _kickLabAuth(op.id, labId);
    }

    // Drop selections for Labs that disappeared OR are no longer
    // authenticated (e.g. the user revoked / users.json wiped).
    let dropped = false;
    for (const id of Array.from(labState.selectedLabIds)) {
      const stillThere = ops.some((o) => o.id === id);
      const stillOk = labState.authStatus[id] === "ok";
      if (!stillThere || !stillOk) {
        labState.selectedLabIds.delete(id);
        dropped = true;
      }
    }
    // Garbage-collect authStatus entries for Labs that disappeared so the
    // dict doesn't grow unbounded across reconnects.
    for (const opId of Object.keys(labState.authStatus)) {
      if (!ops.some((o) => o.id === opId)) delete labState.authStatus[opId];
    }

    if (!ops.length) {
      const empty = document.createElement("div");
      empty.className = "lab-empty";
      empty.textContent = "接続中の Lab はありません。";
      elLabsList.appendChild(empty);
      if (dropped) refreshPaperLibrary();
      return;
    }

    for (const op of ops) {
      const lab = op.lab || {};
      const trust = lab.trust || {};
      const row = document.createElement("div");
      row.className = "lab-row";
      const auth = labState.authStatus[op.id] || "pending";
      // Only authed Labs get the click/select treatment. Pending +
      // failed Labs are visible (so the user knows they exist) but the
      // row is non-interactive and a tag explains why.
      const interactive = (auth === "ok");
      if (!interactive) row.classList.add("lab-no-auth");
      if (auth === "pending") row.classList.add("lab-auth-pending");
      if (auth === "failed") row.classList.add("lab-auth-failed");
      const isSel = interactive && labState.selectedLabIds.has(op.id);
      if (isSel) row.classList.add("selected");
      row.dataset.operatorId = op.id || "";
      if (interactive) {
        row.tabIndex = 0;
        row.setAttribute("role", "checkbox");
        row.setAttribute("aria-checked", String(isSel));
      } else {
        row.setAttribute("aria-disabled", "true");
        row.title = (auth === "pending")
          ? "認証中…"
          : "この Lab に Google サインインできませんでした（選択不可）";
      }

      const head = document.createElement("div");
      head.className = "lab-head";
      // ☑/☐ visual to make it obvious this is a multi-select toggle.
      const check = document.createElement("span");
      check.className = "lab-check";
      check.setAttribute("aria-hidden", "true");
      const dot = document.createElement("div"); dot.className = "lab-dot";
      const name = document.createElement("span"); name.className = "lab-name";
      name.textContent = newLabels[op.id];
      const stats = document.createElement("span"); stats.className = "lab-stats";
      const papers = trust.papers ?? 0;
      const chunks = trust.corpus_chunks ?? 0;
      stats.textContent = `📄 ${papers} / chunks ${chunks.toLocaleString()}`;
      head.append(check, dot, name, stats);
      if (!interactive) {
        const tag = document.createElement("span");
        tag.className = "lab-auth-tag";
        tag.textContent = (auth === "pending") ? "🔒 認証中…" : "🔒 未認証";
        head.append(tag);
      }

      const summary = document.createElement("div");
      summary.className = "lab-summary";
      summary.textContent = lab.summary || "(準備中…)";

      const meta = document.createElement("div");
      meta.className = "lab-meta";
      const updated = _formatTimestamp(trust.last_updated);
      const model = trust.embed_model || "—";
      meta.textContent = `index ${updated} · ${model}`;

      row.append(head, summary, meta);
      if (interactive) {
        const onToggle = () => toggleLab(op.id);
        row.addEventListener("click", onToggle);
        row.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onToggle();
          }
        });
      }
      elLabsList.appendChild(row);
    }
    if (dropped) refreshPaperLibrary();
    updateAskAvailability();
  } catch (e) {
    console.warn("[labs] tunnel info fetch failed:", e.message);
  }
}

// Toggle one Lab's selection ON/OFF. Re-renders the row checkbox state and
// forces the paper library to refresh from the new selection set (or back
// to the locked "Lab を選択してください" placeholder when the set is empty).
function toggleLab(id) {
  if (!id) return;
  // Defense in depth: refreshLabs already drops the click handler on
  // un-authenticated rows, but a stale click could still fire during
  // re-render — refuse the toggle if we don't have a uid for this Lab.
  if (labState.authStatus[id] !== "ok") return;
  if (labState.selectedLabIds.has(id)) {
    labState.selectedLabIds.delete(id);
  } else {
    labState.selectedLabIds.add(id);
  }
  for (const row of elLabsList.querySelectorAll(".lab-row")) {
    const sel = labState.selectedLabIds.has(row.dataset.operatorId);
    row.classList.toggle("selected", sel);
    row.setAttribute("aria-checked", String(sel));
  }
  // QA timeline items are Lab-owned. A selection change can add or remove
  // visible items, so we wipe the merged DOM + per-Lab cursors and
  // re-fetch from the new selection set on the next tick.
  for (const k of Object.keys(lastQaIdByLab)) delete lastQaIdByLab[k];
  elQaBody.innerHTML = '<div class="qa-item empty">まだ質問はありません。</div>';
  elQaCount.textContent = "0";
  refreshPaperLibrary();
  refreshQaTimeline();
  updateAskAvailability();
}

// Ask は論文未適用でも、Lab が 1 つ以上選択されていれば許可する。
// 論文を適用すると updateNav() が走って同じ enable をするが、それより
// 前段（Lab 選択直後）でも質問できるようにこちらでもゲートを開ける。
// 0 Lab に戻ったら逆に Ask を閉じる（fan-out 先がないため）。
function updateAskAvailability() {
  const hasLab = labState.selectedLabIds.size > 0;
  btnAsk.disabled = !hasLab;
  inputQA.disabled = !hasLab;
}

// Best-effort poll of /api/upload/eligibility so a ticket created on
// another browser (e.g. ticket.html on the admin's phone) flips the
// upload / delete / memo buttons live, without a full reload. Boot does
// the initial fetch with explicit reset-on-error semantics; this poll
// skips reset on transient errors to avoid UI flicker.
async function refreshEligibilityForPoll() {
  try {
    const elig = await apiGet("/api/upload/eligibility");
    applyEligibility({
      upload: !!elig.upload,
      remove: !!elig.remove,
      technote: !!elig.technote,
    });
  } catch (_) { /* swallow — keep the last-known state */ }
}

async function pollOnce() {
  await Promise.all([
    heartbeat(),
    refreshParticipants(),
    refreshQaTimeline(),
    refreshLabs(),
    refreshEligibilityForPoll(),
  ]);
}

function startPolling() {
  if (pollTimer) return;
  pollOnce();
  pollTimer = setInterval(pollOnce, POLL_INTERVAL_MS);
}


// In iframe-mode (a generated deck is loaded), the iframe owns slide
// navigation. We toggle the slide DOM directly (mirroring the deck's own
// showSlide) so we don't depend on the deck exposing it as a global —
// `function showSlide()` at the top of a srcdoc script is supposed to
// land on `window`, but in practice it isn't reliable across browsers,
// and a missing `cw.showSlide` was silently making the Next button
// (and runAutoProgress's auto-advance) no-op, causing the script loop
// to read slide 1 forever. Page navigation only — never starts TTS.
function _navInIframe(delta) {
  const st = _readIframeSlideState();
  console.log("[nav] _navInIframe", { delta, st, hasFrame: elSlide.classList.contains("has-frame") });
  if (!st) return false;
  const cw = elSlideFrame.contentWindow;
  if (!cw || !cw.document) {
    console.warn("[nav] iframe contentWindow/document missing");
    return false;
  }
  const next = Math.max(0, Math.min(st.cur + delta, st.total - 1));
  if (next === st.cur) {
    console.log("[nav] at edge, no-op", { cur: st.cur, total: st.total });
    return true;
  }
  // Prefer the deck's own showSlide when it's actually exposed (it also
  // updates the deck's progress bar / page counter / lazy chart init).
  // Fall back to a direct DOM toggle when the deck didn't put it on
  // window — the user-visible result (right slide on screen) is the
  // same, just without the deck's chrome side-effects.
  console.log("[nav] advancing", { from: st.cur, to: next, hasShowSlide: typeof cw.showSlide === "function" });
  if (typeof cw.showSlide === "function") {
    cw.showSlide(next);
  } else {
    const slides = cw.document.querySelectorAll(".slide");
    slides.forEach((s, i) => {
      s.style.display = (i === next) ? "flex" : "none";
    });
  }
  state.scriptIndex = next + 1;
  renderScriptPane(state.scriptIndex);
  updateNav();
  return true;
}

btnPrev.addEventListener("click", () => {
  cancelAuto();
  cancelResume();
  cancelHandRaiseAutoResume();
  if (_navInIframe(-1)) return;
  // Plain deck.json mode: just re-render the prior page, no chat call.
  const idx = state.pageIndex - 1;
  if (idx < 0) return;
  state.pageIndex = idx;
  renderPage(idx);
  updateNav();
});
btnNext.addEventListener("click", () => {
  console.log("[nav] btnNext click", {
    disabled: btnNext.disabled,
    busy: state.busy,
    pageIndex: state.pageIndex,
    scriptIndex: state.scriptIndex,
  });
  cancelAuto();
  cancelResume();
  cancelHandRaiseAutoResume();
  if (_navInIframe(+1)) return;
  console.log("[nav] iframe nav declined, falling back to deck.json mode");
  const pages = state.deck?.pages || [];
  const idx = state.pageIndex + 1;
  if (idx > pages.length) {
    console.log("[nav] deck.json mode: at last page", { idx, total: pages.length });
    return;
  }
  state.pageIndex = idx;
  renderPage(idx);
  updateNav();
});

// Speak is the single audience-control button:
//   - speech in flight  ->  caption shows "✋ 挙手"; clicking stops the
//      current TTS, saves a resume point (handled by speakScriptForIndex's
//      finally), and speaks the hand-raise ack. The user then types their
//      question into the QA input. After they Ask, scheduleResume() picks
//      up scriptResumeAt and continues from the saved line.
//   - nothing speaking ->  caption shows "▶ Speak"; clicking resumes from
//      scriptResumeAt for this slide (if any) or starts from the top.
btnSpeak.addEventListener("click", async () => {
  // Re-entry guard: while the hand-raise ack is speaking, OR while a QA
  // exchange's TTS reply is playing, ignore further clicks. updateNav()
  // may re-enable the disabled prop on slide events, so we rely on state
  // flags rather than the DOM disabled attribute alone.
  if (state.ackInFlight || state.qaInFlight) return;
  cancelAuto();
  cancelResume();
  cancelHandRaiseAutoResume();
  if (state.speakActive) {
    // Hand-raise path. Set handRaised first so multi-chunk speak() loops
    // bail at the next chunk boundary (stopAllSpeech only aborts the
    // current chunk's fetch — the loop would otherwise queue the next
    // chunk and keep talking). Then stopAllSpeech kills active fetches
    // + buffered audio. Yield to let those microtasks settle (loops bail,
    // finally{}s save scriptResumeAt, setSpeakActive(false) propagates).
    // Then clear handRaised so the ack speech itself isn't skipped, and
    // speak the ack.
    state.ackInFlight = true;
    btnSpeak.disabled = true;
    state.handRaised = true;
    stopAllSpeech();
    // stopAllSpeech() flips speakActive=false (→ "▶ Speak"). Restore it
    // synchronously so the icon stays "✋ 挙手" through the whole ack
    // cycle — no ▶ Speak flash between abort and ack speech.
    setSpeakActive(true);
    await new Promise((r) => setTimeout(r, 50));
    state.handRaised = false;
    setStatus("挙手 — お受けします");
    try {
      await _speakHandAck();
    } finally {
      setSpeakActive(false);
      state.ackInFlight = false;
      btnSpeak.disabled = false;
    }
    // Start the silence timer: if the user neither types into QA nor
    // presses Ask within HAND_RAISE_SILENCE_MS, auto-resume by simulating
    // a Speak click. Each keystroke into the QA input restarts the timer.
    scheduleHandRaiseAutoResume();
    return;
  }
  await unlockAudio();
  if (state.scriptSlides && _scriptForIndex(state.scriptIndex)) {
    // Resume from the pause point if it belongs to this slide; otherwise
    // start at line 0. speakScriptForIndex() clears scriptResumeAt at the
    // start of its run, so we capture the line index before calling.
    let startLineIdx = 0;
    if (state.scriptResumeAt && state.scriptResumeAt.slide === state.scriptIndex) {
      startLineIdx = state.scriptResumeAt.lineIdx;
    }
    setBusy(true);
    state.stage = "presenting";
    try {
      await speakScriptForIndex(state.scriptIndex, startLineIdx);
    } finally {
      setBusy(false);
    }
    // Chain into auto-advance only if the slide finished naturally.
    // scriptResumeAt is set on hard-abort (Speak pause) or 挙手 — both
    // mean "the user wanted to stop here", so don't pre-empt with
    // AUTO_PROMPT and skip to the next slide.
    if (!state.scriptResumeAt) {
      runAutoProgress();
    }
    return;
  }
  await doStage("presenting", { slide: currentSlidePayload() });
});

const QA_RESUME_TRANSITION_PHRASE = "それでは、発表を続けさせていただきます。";

// Multi-Lab QA fan-out: send the question to every selected Lab in parallel,
// pulling their fresh QA items into the merged timeline. No auto-TTS — the
// user picks which answer to play via the per-item 🔊 button.
async function fanOutQa(question, labs) {
  const slide = currentSlidePayload();
  setStatus(`asking ${labs.length} labs…`);
  // Mint one UUID per Ask click so every Lab's response is tagged with
  // the same group_id. The server persists it on the QA entry, the
  // timeline echoes it back, and recomputeBestBadges re-derives ★ best
  // from the group on every render — including after a page reload.
  const groupId = (crypto.randomUUID && crypto.randomUUID())
    || `g-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const results = await Promise.allSettled(labs.map((opId) =>
    chat({ stage: "qa", slide, message: question, operatorId: opId, groupId })
  ));
  const ok = results.filter((r) => r.status === "fulfilled").length;
  const failed = results.length - ok;
  setStatus(failed
    ? `asked ${ok}/${labs.length} labs (${failed} failed)`
    : `asked ${ok}/${labs.length} labs ✓`);

  // Pull the just-recorded items into the merged timeline immediately so
  // the user doesn't wait the full 10s poll tick to see answers. The
  // timeline render path now invokes recomputeBestBadges() itself, which
  // walks every group_id present in the DOM and marks the highest-scoring
  // item per group — so ★ best is derivable from server state alone and
  // survives reload.
  await refreshQaTimeline();
}

btnAsk.addEventListener("click", async () => {
  cancelAuto();
  cancelResume();    // any pending 30s resume belongs to a previous question
  cancelHandRaiseAutoResume();
  await unlockAudio();  // user gesture: ensures the AudioContext is live for TTS
  const q = inputQA.value.trim();
  if (!q) return;
  const labs = Array.from(labState.selectedLabIds);
  if (!labs.length) {
    alert("Lab を 1 つ以上選択してください");
    return;
  }
  inputQA.value = "";
  state.qaInFlight = true;
  btnSpeak.disabled = true;
  try {
    if (labs.length === 1) {
      // Single-Lab path: keep auto-TTS + bridge-phrase + slide-resume so
      // the original LT flow is unchanged when only one Lab is active.
      await doStage("qa", { message: q });
      if (state.scriptResumeAt) {
        setSpeakActive(true);
        try {
          await speak(QA_RESUME_TRANSITION_PHRASE, "male");
        } finally {
          setSpeakActive(false);
        }
      }
    } else {
      // Multi-Lab compare mode: parallel chat, NO auto-TTS, NO bridge.
      // The user reads/compares answers in the timeline and clicks a 🔊
      // on whichever Lab they want to actually hear spoken.
      await fanOutQa(q, labs);
    }
  } finally {
    state.qaInFlight = false;
    btnSpeak.disabled = false;
  }
  // Auto-resume the slide narration only on the single-Lab path (the
  // multi-Lab compare flow has no concept of "the answer" to bridge from).
  if (labs.length === 1 && state.scriptResumeAt) {
    btnSpeak.click();
  }
});


// Typing in the QA input cancels the auto chain so the audience has time
// to compose without being talked over. If the 30-second hand-raise
// silence timer is running, restart it on every keystroke so the user
// gets a fresh 30s window after each edit (their forgotten Enter still
// triggers auto-resume eventually, but only after they stop typing).
inputQA.addEventListener("input", () => {
  if (inputQA.value.trim()) cancelAuto();
  if (handRaiseSilenceTimer) {
    scheduleHandRaiseAutoResume();
  }
});
// Skip Enter while the IME is composing — the kanji-confirmation Enter
// must not fire the submit. e.isComposing covers most modern browsers;
// keyCode === 229 catches the Safari/older-Chrome edge case where
// isComposing has already flipped back to false on the confirm key.
inputQA.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  if (e.isComposing || e.keyCode === 229) return;
  e.preventDefault();
  btnAsk.click();
});

btnEnd.addEventListener("click", async () => {
  cancelAuto();
  cancelResume();
  cancelHandRaiseAutoResume();
  state.scriptResumeAt = null;
  await doStage("closing");
  for (const b of [btnPrev, btnSpeak, btnNext, btnAsk, btnEnd]) b.disabled = true;
  inputQA.disabled = true;
  setStatus("closed");
});

btnLogout.addEventListener("click", async () => {
  if (!confirm("ログアウトしますか？")) return;
  cancelAuto();
  btnLogout.disabled = true;
  // Fan-out logout to every authed Lab so each one drops us from its
  // own participants/qa state (a single un-routed POST would only land
  // on whichever Lab the proxy default-picks, leaving the others stuck
  // showing us as still present until their PRESENCE_TIMEOUT elapses).
  try {
    await Promise.allSettled(
      Object.entries(labState.labIdByOpId)
        .filter(([opId]) => labState.authStatus[opId] === "ok")
        .map(([opId]) =>
          fetch(withLab("/api/auth/logout", opId), {
            method: "POST",
            headers: getAuthHeaders(USER_IDS),
          }),
        ),
    );
  } catch (e) {
    console.warn("logout request failed:", e.message);
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  clearStoredSession();
  // Stop Google auto-sign-in so the next visit shows the picker fresh.
  if (typeof google !== "undefined" && google.accounts && google.accounts.id) {
    try { google.accounts.id.disableAutoSelect(); } catch (_) {}
  }
  // Reload to re-run the auth boot flow.
  location.reload();
});

// ---------- paper upload ----------
//
// User selects a PDF -> POST /api/upload/paper -> server runs
// generate_slides.py asynchronously. We poll /api/upload/jobs/{id} every
// 2s, mirroring stdout into the modal's <pre>. When status == "done",
// the user can apply the generated single-page deck (rendered as an
// iframe in #slide). Arrow-key navigation lives inside the iframe.

const uploadState = {
  jobId: null,
  // Lab the upload was started against. All subsequent job-status / slide
  // fetches must reuse this operator_id, even if the user toggles other
  // Labs ON/OFF mid-flight, so they keep talking to the right backend.
  operatorId: null,
  polling: null,
  status: null,    // awaiting_upload | queued | downloading | running | done | error
  filename: "",
  // Number of log lines already mirrored into the modal's <pre>; sent as
  // ?since=<n> on each poll so we only fetch lines we haven't seen.
  logOffset: 0,
};

function showUploadOverlay() { uploadOverlay.classList.add("show"); }
function hideUploadOverlay() { uploadOverlay.classList.remove("show"); }

function setUploadStage(text, kind) {
  uploadStage.textContent = text;
  uploadStage.classList.remove("done", "error");
  if (kind) uploadStage.classList.add(kind);
  // Spinner is only meaningful while we're still working. Hide it once
  // the job lands on a terminal state.
  if (kind === "done" || kind === "error") {
    uploadSpinner.style.display = "none";
  } else {
    uploadSpinner.style.display = "";
  }
}

function appendUploadLog(lines) {
  if (!lines || !lines.length) return;
  // Append a delta — never overwrite — so the user sees the stdout grow
  // line by line through the multi-minute generation.
  const prefix = uploadLog.textContent && !uploadLog.textContent.endsWith("\n") ? "\n" : "";
  uploadLog.textContent += prefix + lines.join("\n");
  uploadLog.scrollTop = uploadLog.scrollHeight;
}

function setSlideLoading(on, label) {
  if (on) {
    elSlide.classList.add("is-generating");
    elSlide.classList.remove("has-frame");
    if (label) {
      const msg = document.getElementById("slide-loader-msg");
      if (msg) msg.firstChild.nodeValue = label;
    }
  } else {
    elSlide.classList.remove("is-generating");
  }
}

function setUploadIndicator(active, label) {
  if (active) {
    uploadIndicatorText.textContent = label || "処理中…";
    uploadIndicator.classList.add("show");
  } else {
    uploadIndicator.classList.remove("show");
  }
}

function applyTalkMeta(meta) {
  // meta = {theme, presenter, venue} — apply whatever is non-empty,
  // leave the other fields blank so an empty meta stays empty rather
  // than retaining a stale value from a previous upload.
  if (!meta) return;
  if (meta.theme) {
    elTitle.textContent = meta.theme;
    document.title = `D.Sugisawa LT — ${meta.theme}`;
  }
  if (typeof meta.presenter === "string") elPresenter.textContent = meta.presenter;
  if (typeof meta.venue === "string") elVenue.textContent = meta.venue;
  // Show the middle-dot only when both fields are present.
  const sep = document.getElementById("venue-sep");
  if (sep) sep.hidden = !(meta.presenter && meta.venue);
}

// Generated decks author every slide at a fixed 1280x720 layout (see
// presentation.html template — `.slide-container { width:1280px; height:720px }`).
// The iframe in the host is flex-sized, so without intervention the slide
// overflows or floats in whitespace as the viewport changes. fitIframeToContainer
// injects a CSS transform on `.slide-container` that scales it to fit the
// iframe while preserving aspect ratio, then re-runs on every iframe resize.
const SLIDE_DESIGN_W = 1280;
const SLIDE_DESIGN_H = 720;
let _slideResizeObserver = null;

function fitIframeToContainer() {
  const cw = elSlideFrame.contentWindow;
  const cd = elSlideFrame.contentDocument;
  if (!cw || !cd || !cd.body) return;
  const w = elSlideFrame.clientWidth;
  const h = elSlideFrame.clientHeight;
  if (w === 0 || h === 0) return;
  const scale = Math.min(w / SLIDE_DESIGN_W, h / SLIDE_DESIGN_H);
  let styleEl = cd.getElementById("__host-fit-style");
  if (!styleEl) {
    styleEl = cd.createElement("style");
    styleEl.id = "__host-fit-style";
    cd.head.appendChild(styleEl);
  }
  // Scale .slide-container around its center; body's flex centering keeps
  // the unscaled box at the iframe center, so the scaled visual box fits
  // exactly in the iframe (letterbox / pillarbox as needed).
  styleEl.textContent = `
    html, body { overflow: hidden !important; margin: 0 !important; }
    .slide-container {
      transform: scale(${scale});
      transform-origin: center center;
    }
  `;
  if (!_slideResizeObserver && typeof ResizeObserver !== "undefined") {
    _slideResizeObserver = new ResizeObserver(() => fitIframeToContainer());
    _slideResizeObserver.observe(elSlideFrame);
  }
}

async function applyGeneratedSlides(operatorId, jobId) {
  // iframes don't carry custom HTTP headers, so the slides endpoint can't
  // see X-User-Id when set via src=URL — it'd 401. Fetch the HTML manually
  // with auth headers and inject it via srcdoc.
  const url = withLab(`/api/upload/jobs/${encodeURIComponent(jobId)}/slides`, operatorId);
  let html;
  try {
    const r = await fetch(url, { headers: getAuthHeaders(USER_IDS) });
    if (!r.ok) throw new Error(`slides ${r.status}: ${await r.text()}`);
    html = await r.text();
  } catch (e) {
    console.error("[upload] failed to fetch slides:", e);
    setSlideLoading(false);
    alert(`スライドの取得に失敗しました: ${e.message}`);
    return;
  }
  elSlideFrame.removeAttribute("src");
  elSlideFrame.srcdoc = html;
  elSlide.classList.remove("is-generating");
  elSlide.classList.add("has-frame");
  elSlideFrame.addEventListener("load", () => {
    const cw = elSlideFrame.contentWindow;
    try { cw.focus(); } catch (_) {}
    // The generated deck owns its own arrow-key handler, but its HTML is
    // same-origin (we injected via srcdoc) so we can peek at the global
    // `currentIndex` it maintains and mirror it into our state.
    const sync = () => {
      try {
        const st = _readIframeSlideState();
        const idx = (st ? st.cur : 0) + 1;
        if (idx !== state.scriptIndex) {
          state.scriptIndex = idx;
          renderScriptPane(idx);
        }
        updateNav();
      } catch (_) {}
    };
    sync();
    try {
      cw.document.addEventListener("keydown", () => setTimeout(sync, 0));
    } catch (_) {}
    // Slides are authored at a fixed 1280x720 design size. Without scaling,
    // a smaller iframe leaves the layout overflowing or surrounded by white
    // space. Scale .slide-container to fit the iframe and re-fit on resize.
    fitIframeToContainer();
  }, { once: true });
  // Pull the per-slide speaker script (best-effort — the iframe still
  // works without it, just falls back to the LLM chat call).
  try {
    const r = await fetch(
      withLab(`/api/upload/jobs/${encodeURIComponent(jobId)}/script`, operatorId),
      { headers: getAuthHeaders(USER_IDS) });
    if (r.ok) {
      const data = await r.json();
      state.scriptSlides = Array.isArray(data.slides) ? data.slides : null;
      renderScriptPane(state.scriptIndex);
    } else {
      state.scriptSlides = null;
    }
  } catch (e) {
    console.warn("[script] fetch failed:", e.message);
    state.scriptSlides = null;
  }
}

function _scriptForIndex(idx) {
  if (!state.scriptSlides) return null;
  return state.scriptSlides.find((s) => s.index === idx) || null;
}

function renderScriptPane(idx, activeLineIdx = -1) {
  const slide = _scriptForIndex(idx);
  if (!slide) {
    elScriptIdx.textContent = `${idx}`;
    elScriptTitle.textContent = "";
    elScriptProgress.textContent = "";
    elScriptLines.innerHTML = '<div class="empty">このスライドのスクリプトはありません。</div>';
    return;
  }
  elScriptIdx.textContent = `${slide.index}`;
  elScriptTitle.textContent = slide.title || "";
  const total = slide.lines.length;
  elScriptProgress.textContent = activeLineIdx >= 0
    ? `${activeLineIdx + 1} / ${total}`
    : `${total} 行`;
  elScriptLines.innerHTML = "";
  // Show all lines but the pane caps to ~10 lines via CSS overflow-y.
  // The currently-speaking line gets scrolled into view.
  slide.lines.forEach((text, i) => {
    const div = document.createElement("div");
    div.className = "line" + (i === activeLineIdx ? " active" : "");
    div.textContent = text;
    elScriptLines.appendChild(div);
  });
  if (activeLineIdx >= 0) {
    const node = elScriptLines.children[activeLineIdx];
    if (node) node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

// "あ、はい挙手された方…" — TTS spoken when the soft-interrupt flag fires.
const HAND_ACK_PHRASE = "あ、はい挙手された方、質問をお願いいたします。";

async function _speakHandAck() {
  await speak(HAND_ACK_PHRASE, "male");
  try { inputQA.focus(); } catch (_) {}
  setStatus("質問をどうぞ");
}

async function speakScriptForIndex(idx, startLineIdx = 0) {
  const slide = _scriptForIndex(idx);
  if (!slide || !slide.lines.length) return false;
  // Cancel any in-flight script playback before starting a new one.
  if (state.scriptSpeakAbort) state.scriptSpeakAbort.abort();
  const ctrl = new AbortController();
  state.scriptSpeakAbort = ctrl;
  // Starting fresh playback clears any stale resume point.
  state.scriptResumeAt = null;
  let i = startLineIdx;
  let handled = false;          // hand-raise was acknowledged inside this run
  setSpeakActive(true);
  try {
    for (; i < slide.lines.length; i++) {
      if (ctrl.signal.aborted) break;
      renderScriptPane(idx, i);
      await speak(slide.lines[i], "male");
      // If the speak above was hard-aborted (Speak pause / navigation),
      // break BEFORE the loop's implicit i++ so finally{} saves the
      // interrupted line as the resume point — not the next one. Without
      // this the second half of the cut-off line would be skipped on
      // resume.
      if (ctrl.signal.aborted) break;
      // Soft interrupt: the user clicked 挙手 while this sentence played.
      // We finished the sentence cleanly, now ack and pause until they
      // ask their question.
      if (state.handRaised) {
        state.handRaised = false;
        state.scriptResumeAt = { slide: idx, lineIdx: i + 1 };
        handled = true;
        await _speakHandAck();
        break;
      }
    }
  } finally {
    if (state.scriptSpeakAbort === ctrl) state.scriptSpeakAbort = null;
    // Hard abort path (stopAllSpeech, e.g. when the user clicks Speak
    // again or navigates) — save where we cut off so a future caller
    // can decide what to do with the resume point.
    if (!handled && ctrl.signal.aborted && i < slide.lines.length) {
      state.scriptResumeAt = { slide: idx, lineIdx: i };
    } else if (!handled) {
      renderScriptPane(idx, -1);
    }
    setSpeakActive(false);
  }
  return true;
}

// 挙手 -> stop -> ask -> wait 30s -> "過不足ございませんでしょうか…" ->
// resume the per-slide script from where we paused. Pre-resume timer is
// cancelled the moment the user asks another question or moves on, so
// we never speak the resume phrase on top of a fresh utterance.
const RESUME_PHRASE = "過不足ございませんでしょうか？それでは、発表を続けさせていただきます。";
const RESUME_WAIT_MS = 30000;
// 挙手 -> ack -> ... user neither asks (Ask) nor types for HAND_RAISE_SILENCE_MS
// -> auto-trigger Speak (which resumes from scriptResumeAt). Any QA-input
// keystroke restarts the silence timer so the user has time to compose.
const HAND_RAISE_SILENCE_MS = 30000;

let resumeTimer = null;
let handRaiseSilenceTimer = null;

function cancelResume() {
  if (resumeTimer) {
    clearTimeout(resumeTimer);
    resumeTimer = null;
  }
}

function cancelHandRaiseAutoResume() {
  if (handRaiseSilenceTimer) {
    clearTimeout(handRaiseSilenceTimer);
    handRaiseSilenceTimer = null;
  }
}

function scheduleHandRaiseAutoResume() {
  cancelHandRaiseAutoResume();
  if (!state.scriptResumeAt) return;
  handRaiseSilenceTimer = setTimeout(() => {
    handRaiseSilenceTimer = null;
    if (!state.scriptResumeAt) return;
    // Same effect as the user clicking Speak themselves: handler reads
    // scriptResumeAt and resumes speakScriptForIndex from the saved line.
    btnSpeak.click();
  }, HAND_RAISE_SILENCE_MS);
}

function scheduleResume() {
  cancelResume();
  if (!state.scriptResumeAt) return;
  resumeTimer = setTimeout(async () => {
    resumeTimer = null;
    const pos = state.scriptResumeAt;
    if (!pos) return;
    state.scriptResumeAt = null;
    setBusy(true);
    state.stage = "presenting";
    try {
      await speak(RESUME_PHRASE, "male");
      await speakScriptForIndex(pos.slide, pos.lineIdx);
    } finally {
      setBusy(false);
    }
    runAutoProgress();
  }, RESUME_WAIT_MS);
}

async function pollUploadJob(jobId) {
  try {
    const url = withLab(
      `/api/upload/jobs/${encodeURIComponent(jobId)}?since=${uploadState.logOffset || 0}`,
      uploadState.operatorId);
    const r = await fetch(url, {
      headers: getAuthHeaders(USER_IDS),
    });
    if (!r.ok) throw new Error(`status ${r.status}`);
    const data = await r.json();
    appendUploadLog(data.log || []);
    if (typeof data.log_total === "number") uploadState.logOffset = data.log_total;
    uploadState.status = data.status;
    if (data.status === "awaiting_upload") {
      setUploadStage("アップロード待ち");
      setUploadIndicator(true, `待機 — ${uploadState.filename}`);
    } else if (data.status === "queued") {
      setUploadStage("待機中");
      setUploadIndicator(true, `待機中 — ${uploadState.filename}`);
    } else if (data.status === "downloading") {
      setUploadStage("S3 取得中…");
      setUploadIndicator(true, `S3 取得中 — ${uploadState.filename}`);
    } else if (data.status === "running") {
      setUploadStage("生成中…");
      setUploadIndicator(true, `生成中 — ${uploadState.filename}`);
    } else if (data.status === "done") {
      setUploadStage("完了", "done");
      setUploadIndicator(false);
      uploadClose.disabled = false;
      uploadApply.disabled = false;
      stopUploadPolling();
      // Meta commits the moment generation completes (even if the user
      // has not yet clicked "Use these slides"): update the page header
      // (title + presenter + venue) and remember it for re-renders.
      applyTalkMeta({
        theme: data.theme || "",
        presenter: data.presenter || "",
        venue: data.venue || "",
      });
      // Auto-swap the slide panel from the loader to the generated deck
      // so the user doesn't have to click anything to see it.
      applyGeneratedSlides(uploadState.operatorId, jobId);
      // Pull the new paper into the dropdown so the user can re-select
      // it later without re-uploading.
      refreshPaperLibrary(paperKey(uploadState.operatorId, jobId));
    } else if (data.status === "error") {
      setUploadStage("失敗", "error");
      setUploadIndicator(false);
      uploadLog.textContent += `\n\n[error] ${data.error || "unknown"}`;
      uploadClose.disabled = false;
      setSlideLoading(false);
      stopUploadPolling();
    }
  } catch (e) {
    console.warn("[upload] poll failed:", e.message);
  }
}

// Generation takes minutes, so a 2-second tick is overkill (~150 reqs/job).
// 10 s gives the user near-realtime status without hammering the proxy/tunnel.
const UPLOAD_POLL_INTERVAL_MS = 10000;

function startUploadPolling(jobId) {
  stopUploadPolling();
  uploadState.jobId = jobId;
  pollUploadJob(jobId);
  uploadState.polling = setInterval(() => pollUploadJob(jobId), UPLOAD_POLL_INTERVAL_MS);
}

function stopUploadPolling() {
  if (uploadState.polling) {
    clearInterval(uploadState.polling);
    uploadState.polling = null;
  }
}

// Upload + delete are paid actions, each gated by its own ticket type
// (.ticket.available for upload, .ticket.remove for delete). The boot
// sequence calls this with the per-action result of
// /api/upload/eligibility. We render ineligible state as **disabled +
// half-opacity** (the existing footer button:disabled CSS) rather than
// hidden — the user needs to see the feature exists but is gated,
// otherwise "the upload button is missing!" reads as a bug. Tooltip
// swaps to a "ticket required" hint so hover explains the lock.
//
// Delete carries an extra constraint: even with a remove ticket, the
// button only makes sense when a paper is selected in the dropdown.
// We stash the ticket flag on btnPaperDelete.dataset so the
// per-selection updater (updatePaperButtons) re-applies both
// constraints without needing to re-fetch eligibility.
function applyEligibility({ upload = false, remove = false, technote = false } = {}) {
  // Stash all three ticket flags on dataset so updatePaperButtons can
  // reapply the (lab-selected ∧ ticket-ok) gate after Lab selection
  // changes too.
  btnUpload.dataset.ticketOk = upload ? "1" : "";
  btnUpload.title = upload
    ? "論文PDFをアップロードしてスライド生成"
    : "論文アップロードはチケット (.ticket.available) が必要です";
  btnPaperDelete.dataset.ticketOk = remove ? "1" : "";
  btnPaperDelete.title = remove
    ? "選択した論文を削除"
    : "論文削除はチケット (.ticket.remove) が必要です";
  if (btnMemo) {
    btnMemo.dataset.ticketOk = technote ? "1" : "";
    btnMemo.title = technote
      ? "技術メモ（テキスト）をRAGに追加"
      : "技術メモ追加はチケット (.ticket.technote) が必要です";
  }
  // Re-run the selection-aware enable/disable so dropdown + lab-selection
  // state is respected after the eligibility flip.
  if (typeof updatePaperButtons === "function") updatePaperButtons();
}

// Pull a human message out of a fetch Response. FastAPI errors come as
// {"detail": "..."}; if the body isn't JSON, fall back to the raw text.
// Used by upload + delete handlers so the user sees "no ticket: ..."
// (or whatever the server said) instead of a generic HTTP-status throw.
async function _extractErrorMessage(res) {
  let body = "";
  try { body = await res.text(); } catch (_) { body = ""; }
  if (!body) return `${res.status} ${res.statusText}`;
  try {
    const j = JSON.parse(body);
    if (j && typeof j.detail === "string") return j.detail;
    if (j && j.detail) return JSON.stringify(j.detail);
  } catch (_) {}
  return body;
}

btnUpload.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  fileInput.value = "";
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    alert("PDF ファイルを選択してください");
    return;
  }
  const targetLab = singleSelectedLab();
  if (!targetLab) {
    alert("アップロード先の Lab を 1 つだけ選択してください");
    return;
  }

  uploadState.filename = file.name;
  uploadState.operatorId = targetLab;
  uploadState.logOffset = 0;
  setUploadStage("アップロード準備中…");
  uploadFilename.textContent = file.name;
  uploadLog.textContent = "";
  uploadClose.disabled = false;     // user can hide modal anytime; job keeps running
  uploadApply.disabled = true;
  showUploadOverlay();
  setSlideLoading(true);
  setUploadIndicator(true, `準備中 — ${file.name}`);

  // Upload path: presign -> PUT direct to S3 -> start. The PDF never goes
  // through the WS tunnel (which has a 1 MB frame cap); only the small
  // presign/start control-plane calls do. Ticket failures (403) at
  // either presign or start get surfaced via alert AND the modal log
  // because the server's "no ticket" message is the actionable bit and
  // the modal alone is easy to miss when it's still loading.
  try {
    const presignRes = await fetch(withLab("/api/upload/presign", targetLab), {
      method: "POST",
      headers: getAuthHeaders(USER_IDS),
      body: JSON.stringify({ filename: file.name }),
    });
    if (!presignRes.ok) {
      const detail = await _extractErrorMessage(presignRes);
      const err = new Error(`presign ${presignRes.status}: ${detail}`);
      err.detail = detail;
      err.status = presignRes.status;
      throw err;
    }
    const presign = await presignRes.json();
    uploadLog.textContent = `[presign] job_id=${presign.job_id}\n[presign] key=${presign.key}\n`;

    setUploadStage("アップロード中…");
    setUploadIndicator(true, `アップロード中 — ${file.name}`);
    const putRes = await fetch(presign.url, {
      method: "PUT",
      headers: { "Content-Type": "application/pdf" },
      body: file,
    });
    if (!putRes.ok) {
      throw new Error(`s3 PUT ${putRes.status}: ${await putRes.text()}`);
    }
    uploadLog.textContent += `[s3] uploaded ${file.size} bytes\n`;

    const startRes = await fetch(withLab("/api/upload/start", targetLab), {
      method: "POST",
      headers: getAuthHeaders(USER_IDS),
      body: JSON.stringify({ job_id: presign.job_id }),
    });
    if (!startRes.ok) {
      const detail = await _extractErrorMessage(startRes);
      const err = new Error(`start ${startRes.status}: ${detail}`);
      err.detail = detail;
      err.status = startRes.status;
      throw err;
    }
    setUploadStage("待機中");
    startUploadPolling(presign.job_id);
  } catch (e) {
    console.error("[upload] failed:", e);
    setUploadStage("失敗", "error");
    uploadLog.textContent += `\n[error] ${e.message || e}`;
    uploadClose.disabled = false;
    setSlideLoading(false);
    setUploadIndicator(false);
    // Pop an alert with the actionable message — server's `detail`
    // when present, otherwise the raw error. 403 with "no ticket: ..."
    // is the common case here and the user needs to see it.
    const msg = e.detail || e.message || String(e);
    alert(`アップロードに失敗しました:\n\n${msg}`);
    // If the failure was a ticket-eligibility 403, re-fetch eligibility
    // so the button half-opacity / tooltip catch up immediately
    // (otherwise the user could re-trigger and hit the same 403).
    if (e.status === 403) {
      try {
        const elig = await apiGet("/api/upload/eligibility");
        applyEligibility({ upload: !!elig.upload, remove: !!elig.remove, technote: !!elig.technote });
      } catch (_) { /* best-effort */ }
    }
  }
});

// Closing the modal does NOT cancel the job — generation keeps running in
// the background and the footer indicator stays lit. Click the indicator
// to re-open the modal and watch the log.
uploadClose.addEventListener("click", () => {
  hideUploadOverlay();
});

uploadIndicator.addEventListener("click", () => {
  if (!uploadState.jobId) return;
  showUploadOverlay();
});

uploadApply.addEventListener("click", () => {
  if (!uploadState.jobId || !uploadState.operatorId) return;
  applyGeneratedSlides(uploadState.operatorId, uploadState.jobId);
  hideUploadOverlay();
});


// ---------- tech-memo (pasted text) upload ----------
//
// Same control plane as the PDF upload (presign -> S3 PUT -> start) but
// the body is a textarea instead of a file, the destination on the server
// is EXTERNAL_TEXT_DIR/<safe>.md, and there's no slide pipeline / no
// ticket gate. Success just enqueues a RAG rebuild — the next swap makes
// the memo searchable in QA.

const MEMO_MAX_BYTES = 1024 * 1024;   // matches server UPLOAD_TEXT_MAX_KB default

function showMemoOverlay() { memoOverlay.classList.add("show"); }
function hideMemoOverlay() { memoOverlay.classList.remove("show"); }

function setMemoStage(text, kind) {
  memoStage.textContent = text || "";
  memoStage.classList.remove("done", "error");
  if (kind) memoStage.classList.add(kind);
}

function setMemoBusy(busy) {
  memoSubmit.disabled = !!busy;
  memoCancel.disabled = !!busy;
  memoTitle.disabled = !!busy;
  memoBody.disabled = !!busy;
}

// Render the memo modal's send-target as "<DisplayName> (<UUID>)" — UUID
// alone is unreadable when several Labs are connected. Falls back to just
// the UUID if the labels cache hasn't populated yet (rare race with the
// initial refreshLabs tick).
function formatLabTarget(opId) {
  const label = labState.labels[opId];
  return label ? `${label} (${opId})` : opId;
}

btnMemo.addEventListener("click", () => {
  const targetLab = singleSelectedLab();
  if (!targetLab) {
    alert("送信先の Lab を 1 つだけ選択してください");
    return;
  }
  memoTarget.textContent = formatLabTarget(targetLab);
  memoTitle.value = "";
  memoBody.value = "";
  setMemoStage("");
  setMemoBusy(false);
  showMemoOverlay();
  setTimeout(() => memoTitle.focus(), 0);
});

memoCancel.addEventListener("click", () => {
  hideMemoOverlay();
});

memoSubmit.addEventListener("click", async () => {
  const targetLab = singleSelectedLab();
  if (!targetLab) {
    alert("送信先の Lab を 1 つだけ選択してください");
    return;
  }
  const title = memoTitle.value.trim();
  const body = memoBody.value;
  if (!title) {
    alert("タイトルを入力してください");
    memoTitle.focus();
    return;
  }
  if (!body.trim()) {
    alert("本文を入力してください");
    memoBody.focus();
    return;
  }
  // UTF-8 byte length — string.length undercounts non-ASCII. Pre-flight
  // here so the user sees the limit before we hit S3.
  const bodyBytes = new TextEncoder().encode(body);
  if (bodyBytes.byteLength > MEMO_MAX_BYTES) {
    alert(`本文が大きすぎます（${bodyBytes.byteLength} bytes、上限 ${MEMO_MAX_BYTES} bytes）`);
    return;
  }

  setMemoBusy(true);
  setMemoStage("送信中…");
  try {
    const presignRes = await fetch(withLab("/api/upload/text/presign", targetLab), {
      method: "POST",
      headers: getAuthHeaders(USER_IDS),
      body: JSON.stringify({ filename: title, title }),
    });
    if (!presignRes.ok) {
      const detail = await _extractErrorMessage(presignRes);
      const err = new Error(`presign ${presignRes.status}: ${detail}`);
      err.status = presignRes.status;
      err.detail = detail;
      throw err;
    }
    const presign = await presignRes.json();

    setMemoStage("アップロード中…");
    const putRes = await fetch(presign.url, {
      method: "PUT",
      headers: { "Content-Type": "text/markdown; charset=utf-8" },
      body: bodyBytes,
    });
    if (!putRes.ok) {
      throw new Error(`s3 PUT ${putRes.status}: ${await putRes.text()}`);
    }

    setMemoStage("取り込み中…");
    const startRes = await fetch(withLab("/api/upload/text/start", targetLab), {
      method: "POST",
      headers: getAuthHeaders(USER_IDS),
      body: JSON.stringify({ job_id: presign.job_id, title }),
    });
    if (!startRes.ok) {
      const detail = await _extractErrorMessage(startRes);
      const err = new Error(`start ${startRes.status}: ${detail}`);
      err.status = startRes.status;
      err.detail = detail;
      throw err;
    }
    setMemoStage("送信完了 — 管理者の RAG 再構築で反映されます", "done");
    setMemoBusy(false);
    // Auto-close after a short pause so the user sees the success state.
    setTimeout(() => {
      hideMemoOverlay();
      setMemoStage("");
    }, 1500);
  } catch (e) {
    console.error("[memo] upload failed:", e);
    setMemoStage("失敗", "error");
    setMemoBusy(false);
    const msg = e.detail || e.message || String(e);
    alert(`技術メモの送信に失敗しました:\n\n${msg}`);
    // Ticket-eligibility 403: refresh per-action flags so the memo
    // button half-opacity / tooltip catch up immediately.
    if (e.status === 403) {
      try {
        const elig = await apiGet("/api/upload/eligibility");
        applyEligibility({
          upload: !!elig.upload,
          remove: !!elig.remove,
          technote: !!elig.technote,
        });
      } catch (_) { /* best-effort */ }
    }
  }
});


// ---------- paper library (already-processed papers) ----------
//
// Lists papers persisted under professor_data/uploads/ that have a full
// artifact set (HTML + script.md). Selecting one reuses those artifacts —
// no regeneration — and just refreshes current_meta server-side.

async function refreshPaperLibrary(selectKey) {
  // Until the user explicitly toggles at least one Lab ON, the dropdown
  // stays locked. We bail before touching the network so a stale list
  // from previously-selected Labs can't briefly remain selectable.
  const selectedIds = Array.from(labState.selectedLabIds);
  if (!selectedIds.length) {
    paperLibrary.innerHTML = "";
    const ph = document.createElement("option");
    ph.value = "";
    ph.textContent = "Lab を選択してください";
    paperLibrary.appendChild(ph);
    paperLibrary.disabled = true;
    updatePaperButtons();
    return;
  }

  // Fan out to every selected Lab in parallel. A partial failure (one Lab
  // offline / errored) doesn't block the rest from rendering — we just
  // console.warn and skip that group.
  const results = await Promise.all(selectedIds.map(async (opId) => {
    try {
      const r = await fetch(withLab("/api/upload/papers", opId),
                            { headers: getAuthHeaders(USER_IDS) });
      if (!r.ok) throw new Error(`status ${r.status}`);
      const data = await r.json();
      return { opId, papers: Array.isArray(data.papers) ? data.papers : [] };
    } catch (e) {
      console.warn(`[library] fetch failed for op=${opId}:`, e.message);
      return { opId, papers: [], error: e.message };
    }
  }));

  paperLibrary.innerHTML = "";
  const totalCount = results.reduce((n, r) => n + r.papers.length, 0);
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = totalCount
    ? "アップロード済み論文…"
    : "（選択 Lab に論文なし）";
  paperLibrary.appendChild(placeholder);

  // One <optgroup> per Lab so the user can see which Lab a paper belongs to
  // — that mapping is what option.value (encoded as "<op>|<job>") preserves
  // for apply/delete handlers.
  for (const { opId, papers, error } of results) {
    if (!papers.length && !error) continue;
    const grp = document.createElement("optgroup");
    const labelBase = labState.labels[opId] || opId.slice(0, 6);
    grp.label = error ? `${labelBase} (取得失敗)` : labelBase;
    for (const p of papers) {
      const opt = document.createElement("option");
      opt.value = paperKey(opId, p.job_id);
      const label = p.theme || p.filename || p.job_id;
      opt.textContent = `${label} (${p.slides}枚)`;
      grp.appendChild(opt);
    }
    paperLibrary.appendChild(grp);
  }

  if (selectKey) {
    const opt = paperLibrary.querySelector(`option[value="${CSS.escape(selectKey)}"]`);
    if (opt) paperLibrary.value = selectKey;
  }
  paperLibrary.disabled = totalCount === 0;
  updatePaperButtons();
}

function updatePaperButtons() {
  const labCount = labState.selectedLabIds.size;
  const has = !!paperLibrary.value;
  btnPaperApply.disabled = labCount === 0 || !has;
  // Delete is gated by paper selection + remove-ticket. The owning Lab is
  // baked into the option value so we don't need to enforce single-Lab.
  const ticketOk = btnPaperDelete.dataset.ticketOk === "1";
  btnPaperDelete.disabled = labCount === 0 || !has || !ticketOk;
  // Upload requires EXACTLY one selected Lab — uploading with 0 or >1 Labs
  // selected has no unambiguous target. Tooltip explains the gate.
  const uploadTicket = btnUpload?.dataset.ticketOk === "1";
  if (btnUpload) {
    btnUpload.disabled = labCount !== 1 || !uploadTicket;
    if (!uploadTicket) {
      btnUpload.title = "論文アップロードはチケット (.ticket.available) が必要です";
    } else if (labCount === 0) {
      btnUpload.title = "アップロードする Lab を 1 つ選択してください";
    } else if (labCount > 1) {
      btnUpload.title = "アップロード対象の Lab を 1 つに絞ってください";
    } else {
      btnUpload.title = "論文PDFをアップロードしてスライド生成";
    }
  }
  // Tech-memo upload: same single-Lab gate as PDF upload, plus a
  // technote ticket (.ticket.technote) requirement.
  const technoteTicket = btnMemo?.dataset.ticketOk === "1";
  if (btnMemo) {
    btnMemo.disabled = labCount !== 1 || !technoteTicket;
    if (!technoteTicket) {
      btnMemo.title = "技術メモ追加はチケット (.ticket.technote) が必要です";
    } else if (labCount === 0) {
      btnMemo.title = "送信先の Lab を 1 つ選択してください";
    } else if (labCount > 1) {
      btnMemo.title = "送信先 Lab を 1 つに絞ってください";
    } else {
      btnMemo.title = "技術メモ（テキスト）をRAGに追加";
    }
  }
}

paperLibrary.addEventListener("change", updatePaperButtons);

async function applyPaperById(operatorId, jobId) {
  if (!operatorId || !jobId) throw new Error("operator/job missing");
  cancelAuto();
  cancelResume();
  const r = await fetch(
    withLab(`/api/upload/papers/${encodeURIComponent(jobId)}/select`, operatorId),
    { method: "POST", headers: getAuthHeaders(USER_IDS) },
  );
  if (!r.ok) throw new Error(`select ${r.status}: ${await r.text()}`);
  const data = await r.json();
  // Pin presenter narration / waiting / closing chat calls to this Lab so
  // the LT persona stays consistent with the slides we just applied. Q&A
  // still fans out to all selected Labs in parallel; that path bypasses
  // currentLabId.
  state.currentLabId = operatorId;
  applyTalkMeta({
    theme: data.theme || "",
    presenter: data.presenter || "",
    venue: data.venue || "",
  });
  await applyGeneratedSlides(operatorId, jobId);
  setStatus(`paper applied: ${data.filename || jobId}`);
  return data;
}

btnPaperApply.addEventListener("click", async () => {
  const parsed = parsePaperKey(paperLibrary.value);
  if (!parsed) return;
  btnPaperApply.disabled = true;
  try {
    await applyPaperById(parsed.operator_id, parsed.job_id);
  } catch (e) {
    console.error("[library] apply failed:", e);
    alert(`論文の適用に失敗しました: ${e.message}`);
  } finally {
    updatePaperButtons();
  }
});

btnPaperDelete.addEventListener("click", async () => {
  const parsed = parsePaperKey(paperLibrary.value);
  if (!parsed) return;
  const opt = paperLibrary.options[paperLibrary.selectedIndex];
  const label = opt ? opt.textContent : parsed.job_id;
  if (!confirm(`「${label}」を削除します。よろしいですか?`)) return;
  btnPaperDelete.disabled = true;
  let lastStatus = 0;
  try {
    const r = await fetch(
      withLab(`/api/upload/papers/${encodeURIComponent(parsed.job_id)}`,
              parsed.operator_id),
      { method: "DELETE", headers: getAuthHeaders(USER_IDS) },
    );
    if (!r.ok) {
      lastStatus = r.status;
      const detail = await _extractErrorMessage(r);
      const err = new Error(`delete ${r.status}: ${detail}`);
      err.detail = detail;
      throw err;
    }
    // If the deleted paper was the active one, drop the iframe and reset
    // the header so the UI stops referencing artifacts that just vanished.
    if (state.scriptSlides) {
      state.scriptSlides = null;
      state.scriptIndex = 1;
      elSlide.classList.remove("has-frame");
      elSlideFrame.removeAttribute("srcdoc");
      elSlideFrame.removeAttribute("src");
      renderScriptPane(1);
    }
    applyTalkMeta({ theme: "", presenter: "", venue: "" });
    elTitle.textContent = "= = = =";
    document.title = "D.Sugisawa LT";
    await refreshPaperLibrary();
  } catch (e) {
    console.error("[library] delete failed:", e);
    const msg = e.detail || e.message || String(e);
    alert(`論文の削除に失敗しました:\n\n${msg}`);
    // Ticket-eligibility 403: refresh the per-action flags so the
    // delete button half-opacity / tooltip catch up immediately.
    if (lastStatus === 403) {
      try {
        const elig = await apiGet("/api/upload/eligibility");
        applyEligibility({ upload: !!elig.upload, remove: !!elig.remove, technote: !!elig.technote });
      } catch (_) { /* best-effort */ }
    }
  } finally {
    updatePaperButtons();
  }
});


// ---------- boot ----------
(async () => {
  setStatus("認証中…");
  initAvatar();

  // Pop the Google picker once up-front so the user sees a clear "sign
  // in" moment. The credential is then cached in-memory and replayed
  // against each Lab as refreshLabs discovers them — successful Labs
  // become selectable, failed Labs stay visible but locked.
  const session = await ensureGoogleSignIn();
  if (!session) {
    setStatus("Google サインインに失敗しました");
    btnSpeak.disabled = true;
    return;
  }
  DISPLAY_NAME = session.displayName || DISPLAY_NAME;
  // Seed USER_IDS from previously stored {lab_id: uid} pairs so a returning
  // user can auth-check instead of re-registering on every Lab.
  USER_IDS = getUserIdMap();
  console.log("[auth] signed in:", session.email,
    "labs cached:", Object.keys(USER_IDS));

  // Paid actions (upload + delete + technote) are each gated by a
  // single-use ticket. Fetch all three flags once at boot so the UI
  // shows only the buttons this account can actually use. Any error
  // treats the user as ineligible for all of them, since granting paid
  // actions to a failed check would defeat the gate.
  try {
    const elig = await apiGet("/api/upload/eligibility");
    applyEligibility({
      upload: !!elig.upload,
      remove: !!elig.remove,
      technote: !!elig.technote,
    });
  } catch (e) {
    console.warn("[ticket] eligibility check failed; hiding paid UI:", e);
    applyEligibility({ upload: false, remove: false, technote: false });
  }

  // Start the 10s presence/qa polling now that we have an authenticated id.
  startPolling();

  setStatus(`loading…  (${DISPLAY_NAME})`);
  try {
    const [deck, config] = await Promise.all([
      apiGet("/api/deck"),
      apiGet("/api/config"),
    ]);
    state.deck = deck;
    state.config = config;
    // Per-tab UX: every browser load starts blank. The user picks a
    // paper from the dropdown (or uploads one) to populate the header.
    // Server-side `current_meta` still drives /api/chat persona — that
    // is intentional, since the LLM persona shouldn't reset just
    // because somebody refreshed their tab.
    applyTalkMeta({ theme: "", presenter: "", venue: "" });
    renderPage(0);
    // Populate the "アップロード済み論文…" dropdown from disk-rehydrated
    // jobs so the user can pick a previously-processed paper.
    await refreshPaperLibrary();
    // Deep-link: ?lab=<operator_id>&paper=<job_id> auto-selects the named
    // Lab + paper so a tab can boot directly into a specific talk. Both
    // params are required now that papers live under a Lab — a bare
    // ?paper= without ?lab= is logged and ignored.
    const sp = new URLSearchParams(location.search);
    const labParam = sp.get("lab");
    const paperParam = sp.get("paper");
    if (paperParam) {
      try {
        if (!labParam) {
          throw new Error("missing ?lab=<operator_id>");
        }
        labState.selectedLabIds.add(labParam);
        await refreshLabs();
        await applyPaperById(labParam, paperParam);
        await refreshPaperLibrary(paperKey(labParam, paperParam));
      } catch (e) {
        console.warn("[boot] ?paper auto-apply failed:", e.message);
      }
    }
    setStatus(`ready  (${DISPLAY_NAME})`);
  } catch (e) {
    console.error(e);
    setStatus(`boot error: ${e.message}`);
  }
})();
