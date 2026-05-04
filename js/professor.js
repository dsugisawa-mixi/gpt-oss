// professor.js — D.Sugisawa LT viewer.
// Runs in the browser (S3-hosted). All API calls are relative; the tunnel
// in front of S3 routes /chat /deck /config to your_professor_server.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { ensureUser, getAuthHeaders } from "./auth.js";

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

// ---------- session id (filled in after Google Sign-In) ----------
let USER_ID = null;
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
const fileInput = $("file-paper");
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
  btnAsk.disabled = false;
  inputQA.disabled = false;
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
  const r = await fetch(path, { headers: getAuthHeaders(USER_ID) });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

async function chat({ stage, slide = null, message = "" }) {
  const body = { user_id: USER_ID, stage, message };
  if (slide) body.slide = slide;
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: getAuthHeaders(USER_ID),
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
      headers: getAuthHeaders(USER_ID),
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
let lastQaId = 0;

const elParticipantsList = document.getElementById("participants-list");
const elParticipantsCount = document.getElementById("participants-count");
const elQaBody = document.getElementById("qa-body");
const elQaCount = document.getElementById("qa-count");

async function heartbeat() {
  try {
    await fetch("/api/presence/heartbeat", {
      method: "POST",
      headers: getAuthHeaders(USER_ID),
      body: JSON.stringify({ user_id: USER_ID }),
    });
  } catch (e) {
    console.warn("[presence] heartbeat failed:", e.message);
  }
}

async function refreshParticipants() {
  try {
    const r = await fetch("/api/presence/users", { headers: getAuthHeaders(USER_ID) });
    if (!r.ok) return;
    const data = await r.json();
    elParticipantsCount.textContent = data.count;
    elParticipantsList.innerHTML = "";
    for (const u of data.users) {
      const row = document.createElement("div");
      const cls = ["user-row"];
      if (u.user_id === USER_ID) cls.push("me");
      if (!u.online) cls.push("offline");
      row.className = cls.join(" ");

      const dot = document.createElement("div"); dot.className = "dot";

      const wrap = document.createElement("div"); wrap.className = "name-wrap";
      const name = document.createElement("div"); name.className = "name";
      name.textContent = u.display_name || u.user_id.slice(0, 8);
      wrap.appendChild(name);
      if (u.email_local) {
        const sub = document.createElement("div");
        sub.className = "name-sub";
        sub.textContent = `(${u.email_local})`;
        wrap.appendChild(sub);
      }
      row.append(dot, wrap);
      elParticipantsList.appendChild(row);
    }
  } catch (e) {
    console.warn("[presence] users fetch failed:", e.message);
  }
}

async function refreshQaTimeline() {
  try {
    // Server semantics: start = exclusive lower bound on qa_id, end=-1
    // means open-ended. lastQaId starts at 0 so the first poll requests
    // everything (qa_id > 0).
    const url = `/api/qa/timeline?start=${lastQaId}&end=-1&limit=50`;
    const r = await fetch(url, { headers: getAuthHeaders(USER_ID) });
    if (!r.ok) return;
    const data = await r.json();
    if (!data.items.length) return;
    // Replace empty placeholder once we have data
    const empty = elQaBody.querySelector(".empty");
    if (empty) empty.remove();
    for (const item of data.items) {
      const div = document.createElement("div");
      div.className = "qa-item";
      div.dataset.qaId = item.qa_id;
      const meta = document.createElement("div");
      meta.className = "meta";
      const who = document.createElement("span"); who.className = "who";
      who.textContent = item.display_name;
      const when = document.createElement("span");
      when.textContent = new Date(item.ts * 1000).toLocaleTimeString();
      meta.append(who, when);
      if (item.slide_page) {
        const sp = document.createElement("span"); sp.textContent = `slide ${item.slide_page}`;
        meta.append(sp);
      }
      const q = document.createElement("div"); q.className = "q";
      q.textContent = item.question;
      const a = document.createElement("div"); a.className = "a";
      a.textContent = item.answer;
      div.append(meta, q, a);
      elQaBody.appendChild(div);
      lastQaId = Math.max(lastQaId, item.qa_id);
    }
    elQaBody.scrollTop = elQaBody.scrollHeight;
    // Update total count from server-known timeline length: we only know
    // delta here; track via accumulator on the DOM.
    elQaCount.textContent = elQaBody.querySelectorAll(".qa-item:not(.empty)").length;
  } catch (e) {
    console.warn("[qa] timeline fetch failed:", e.message);
  }
}

async function pollOnce() {
  await Promise.all([heartbeat(), refreshParticipants(), refreshQaTimeline()]);
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

btnAsk.addEventListener("click", async () => {
  cancelAuto();
  cancelResume();    // any pending 30s resume belongs to a previous question
  cancelHandRaiseAutoResume();
  await unlockAudio();  // user gesture: ensures the AudioContext is live for TTS
  const q = inputQA.value.trim();
  if (!q) return;
  inputQA.value = "";
  // Lock the Speak (= 挙手) button for the QA TTS reply AND the
  // transitional phrase that follows, so the audience can't interrupt
  // mid-flow with another hand-raise. doStage's chat+speak runs first,
  // then a short bridge phrase ("それでは、発表を続けさせていただきます。")
  // so the resumed slide narration doesn't slam in cold after the answer.
  state.qaInFlight = true;
  btnSpeak.disabled = true;
  try {
    await doStage("qa", { message: q });
    if (state.scriptResumeAt) {
      // doStage finished with setSpeakActive(false) — restore "✋ 挙手"
      // for the bridge phrase so the icon stays consistent through the
      // whole locked window.
      setSpeakActive(true);
      try {
        await speak(QA_RESUME_TRANSITION_PHRASE, "male");
      } finally {
        setSpeakActive(false);
      }
    }
  } finally {
    state.qaInFlight = false;
    btnSpeak.disabled = false;
  }
  // Now resume the slide. btnSpeak.click() routes through the play branch
  // and starts speakScriptForIndex from scriptResumeAt.lineIdx; its
  // synchronous setSpeakActive(true) flips the icon back to "✋ 挙手"
  // before the next paint, so there's no ▶ Speak flicker.
  if (state.scriptResumeAt) {
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
  try {
    await fetch("/api/auth/logout", {
      method: "POST",
      headers: getAuthHeaders(USER_ID),
    });
  } catch (e) {
    console.warn("logout request failed:", e.message);
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  localStorage.removeItem("professor_user_id");
  localStorage.removeItem("professor_display_name");
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

async function applyGeneratedSlides(jobId) {
  // iframes don't carry custom HTTP headers, so the slides endpoint can't
  // see X-User-Id when set via src=URL — it'd 401. Fetch the HTML manually
  // with auth headers and inject it via srcdoc.
  const url = `/api/upload/jobs/${encodeURIComponent(jobId)}/slides`;
  let html;
  try {
    const r = await fetch(url, { headers: getAuthHeaders(USER_ID) });
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
    const r = await fetch(`/api/upload/jobs/${encodeURIComponent(jobId)}/script`, {
      headers: getAuthHeaders(USER_ID),
    });
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
    const url = `/api/upload/jobs/${encodeURIComponent(jobId)}?since=${uploadState.logOffset || 0}`;
    const r = await fetch(url, {
      headers: getAuthHeaders(USER_ID),
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
      applyGeneratedSlides(jobId);
      // Pull the new paper into the dropdown so the user can re-select
      // it later without re-uploading.
      refreshPaperLibrary(jobId);
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
function applyEligibility({ upload = false, remove = false } = {}) {
  btnUpload.disabled = !upload;
  btnUpload.title = upload
    ? "論文PDFをアップロードしてスライド生成"
    : "論文アップロードはチケット (.ticket.available) が必要です";
  btnPaperDelete.dataset.ticketOk = remove ? "1" : "";
  btnPaperDelete.title = remove
    ? "選択した論文を削除"
    : "論文削除はチケット (.ticket.remove) が必要です";
  // Re-run the selection-aware enable/disable so dropdown state is
  // respected after the eligibility flip.
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

  uploadState.filename = file.name;
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
    const presignRes = await fetch("/api/upload/presign", {
      method: "POST",
      headers: getAuthHeaders(USER_ID),
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

    const startRes = await fetch("/api/upload/start", {
      method: "POST",
      headers: getAuthHeaders(USER_ID),
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
        applyEligibility({ upload: !!elig.upload, remove: !!elig.remove });
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
  if (!uploadState.jobId) return;
  applyGeneratedSlides(uploadState.jobId);
  hideUploadOverlay();
});


// ---------- paper library (already-processed papers) ----------
//
// Lists papers persisted under professor_data/uploads/ that have a full
// artifact set (HTML + script.md). Selecting one reuses those artifacts —
// no regeneration — and just refreshes current_meta server-side.

async function refreshPaperLibrary(selectJobId) {
  try {
    const r = await fetch("/api/upload/papers", { headers: getAuthHeaders(USER_ID) });
    if (!r.ok) throw new Error(`status ${r.status}`);
    const data = await r.json();
    const papers = data.papers || [];
    // Rebuild options (keep the placeholder at the top).
    paperLibrary.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = papers.length
      ? "アップロード済み論文…"
      : "（アップロード済み論文なし）";
    paperLibrary.appendChild(placeholder);
    for (const p of papers) {
      const opt = document.createElement("option");
      opt.value = p.job_id;
      const label = p.theme || p.filename || p.job_id;
      opt.textContent = `${label} (${p.slides}枚)`;
      paperLibrary.appendChild(opt);
    }
    if (selectJobId && papers.some((p) => p.job_id === selectJobId)) {
      paperLibrary.value = selectJobId;
    }
    paperLibrary.disabled = papers.length === 0;
    updatePaperButtons();
  } catch (e) {
    console.warn("[library] refresh failed:", e.message);
  }
}

function updatePaperButtons() {
  const has = !!paperLibrary.value;
  btnPaperApply.disabled = !has;
  // Delete is gated by BOTH a selection AND a remove-ticket. The
  // ticket flag was stashed onto the dataset by applyEligibility().
  // Empty dataset (= eligibility never resolved) treats as no ticket
  // so we fail closed.
  const ticketOk = btnPaperDelete.dataset.ticketOk === "1";
  btnPaperDelete.disabled = !has || !ticketOk;
}

paperLibrary.addEventListener("change", updatePaperButtons);

async function applyPaperById(jobId) {
  cancelAuto();
  cancelResume();
  const r = await fetch(
    `/api/upload/papers/${encodeURIComponent(jobId)}/select`,
    { method: "POST", headers: getAuthHeaders(USER_ID) },
  );
  if (!r.ok) throw new Error(`select ${r.status}: ${await r.text()}`);
  const data = await r.json();
  applyTalkMeta({
    theme: data.theme || "",
    presenter: data.presenter || "",
    venue: data.venue || "",
  });
  await applyGeneratedSlides(jobId);
  setStatus(`paper applied: ${data.filename || jobId}`);
  return data;
}

btnPaperApply.addEventListener("click", async () => {
  const jobId = paperLibrary.value;
  if (!jobId) return;
  btnPaperApply.disabled = true;
  try {
    await applyPaperById(jobId);
  } catch (e) {
    console.error("[library] apply failed:", e);
    alert(`論文の適用に失敗しました: ${e.message}`);
  } finally {
    updatePaperButtons();
  }
});

btnPaperDelete.addEventListener("click", async () => {
  const jobId = paperLibrary.value;
  if (!jobId) return;
  const opt = paperLibrary.options[paperLibrary.selectedIndex];
  const label = opt ? opt.textContent : jobId;
  if (!confirm(`「${label}」を削除します。よろしいですか?`)) return;
  btnPaperDelete.disabled = true;
  let lastStatus = 0;
  try {
    const r = await fetch(
      `/api/upload/papers/${encodeURIComponent(jobId)}`,
      { method: "DELETE", headers: getAuthHeaders(USER_ID) },
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
        applyEligibility({ upload: !!elig.upload, remove: !!elig.remove });
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

  const me = await ensureUser({});
  if (!me || !me.userId) {
    setStatus("認証に失敗しました（mixi.co.jp ドメインのみ許可）");
    btnSpeak.disabled = true;
    return;
  }
  USER_ID = me.userId;
  DISPLAY_NAME = me.displayName;
  console.log("[auth] signed in:", USER_ID, DISPLAY_NAME);

  // Paid actions (upload + delete) are each gated by a single-use
  // ticket. Fetch both flags once at boot so the UI shows only the
  // buttons this account can actually use. Any error treats the user
  // as ineligible for both, since granting paid actions to a failed
  // check would defeat the gate.
  try {
    const elig = await apiGet("/api/upload/eligibility");
    applyEligibility({ upload: !!elig.upload, remove: !!elig.remove });
  } catch (e) {
    console.warn("[ticket] eligibility check failed; hiding paid UI:", e);
    applyEligibility({ upload: false, remove: false });
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
    // Deep-link: ?paper=<job_id> auto-selects + applies the named paper
    // so a tab can boot directly into a specific talk.
    const paperParam = new URLSearchParams(location.search).get("paper");
    if (paperParam) {
      try {
        await applyPaperById(paperParam);
        paperLibrary.value = paperParam;
        updatePaperButtons();
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
