// auth.js — Google Sign-In + per-Lab ensureUser flow for the LT viewer.
//
// Each Lab keeps its own users.json (Google sign-in registers a fresh
// user_id per Lab), so a single browser must hold N user_ids when N Labs
// are connected. We store a {lab_id: user_id} map in localStorage and
// emit one ``X-User-Id-{lab_id}`` header per known pair on every request;
// each Lab's require_auth dependency only reads its own header.
//
// Usage:
//   import { ensureGoogleSignIn, signInWithLab,
//            getAuthHeaders, getUserIdMap, GOOGLE_CLIENT_ID } from './auth.js';
//   const session = await ensureGoogleSignIn();
//   for (const op of labs) {
//     await signInWithLab({ opId: op.id, labId: op.lab.lab_id });
//   }

export const GOOGLE_CLIENT_ID =
  "349549531314-2s0rvdf4hsb92l2hae14viintjjpvl5t.apps.googleusercontent.com";

const API_BASE = "";   // relative — proxy/tunnel handles routing
const STORAGE_USER_IDS = "professor_user_ids";       // JSON {lab_id: uid}
const STORAGE_DISPLAY_NAME = "professor_display_name";
const STORAGE_LEGACY_USER_ID = "professor_user_id";  // pre-multi-Lab single uid

let _clientId = GOOGLE_CLIENT_ID;
let _getLang = () => "ja";

// In-memory cache of the most recent successful Google credential JWT.
// Used to register the same Google identity against newly-discovered
// Labs without re-prompting. Lives only for this page session.
let _cachedCredential = null;
let _cachedDisplayName = null;
let _cachedEmail = null;
let _cachedSub = null;

// In-flight Google sign-in promise so two concurrent signInWithLab calls
// don't both pop the picker.
let _signInPromise = null;

export function initAuth(deps = {}) {
  if (deps.clientId) _clientId = deps.clientId;
  if (deps.getLang) _getLang = deps.getLang;
}

// ---- localStorage map helpers ------------------------------------------------

function loadUserIdMap() {
  try {
    const raw = localStorage.getItem(STORAGE_USER_IDS);
    const parsed = raw ? JSON.parse(raw) : null;
    if (parsed && typeof parsed === "object") return parsed;
  } catch (_) { /* fall through */ }
  return {};
}

function saveUserIdMap(map) {
  localStorage.setItem(STORAGE_USER_IDS, JSON.stringify(map));
}

function setLabUserId(labId, uid) {
  if (!labId || !uid) return;
  const m = loadUserIdMap();
  if (m[labId] === uid) return;
  m[labId] = uid;
  saveUserIdMap(m);
}

function clearLabUserId(labId) {
  if (!labId) return;
  const m = loadUserIdMap();
  if (!(labId in m)) return;
  delete m[labId];
  saveUserIdMap(m);
}

export function getUserIdMap() {
  return loadUserIdMap();
}

export function getUserIdForLab(labId) {
  return loadUserIdMap()[labId] || null;
}

export function clearStoredSession() {
  localStorage.removeItem(STORAGE_USER_IDS);
  localStorage.removeItem(STORAGE_LEGACY_USER_ID);
  localStorage.removeItem(STORAGE_DISPLAY_NAME);
  _cachedCredential = null;
  _cachedDisplayName = null;
  _cachedEmail = null;
  _cachedSub = null;
}

// Build per-Lab auth headers. Pass the {lab_id: uid} map from getUserIdMap().
// Each entry becomes one ``X-User-Id-{lab_id}: uid`` header, so a request
// the proxy routes to any Lab is authenticated by that Lab's own uid.
export function getAuthHeaders(userIdMap) {
  const h = { "Content-Type": "application/json" };
  if (userIdMap) {
    for (const [labId, uid] of Object.entries(userIdMap)) {
      if (labId && uid) h[`X-User-Id-${labId}`] = uid;
    }
  }
  return h;
}

// ---- error overlay -----------------------------------------------------------

function showAuthError(msg) {
  let el = document.getElementById("auth-error");
  if (!el) {
    el = document.createElement("div");
    el.id = "auth-error";
    el.style.cssText =
      "position:fixed;top:0;left:0;right:0;z-index:300;" +
      "background:#5a1622;color:#fff;padding:14px 18px;" +
      "font-family:'Hiragino Sans','Yu Gothic','Segoe UI',sans-serif;" +
      "font-size:0.95em;line-height:1.5;text-align:center;" +
      "box-shadow:0 2px 12px rgba(0,0,0,0.4);";
    document.body.appendChild(el);
  }
  el.innerHTML = "";
  const span = document.createElement("span");
  span.textContent = msg;
  el.appendChild(span);
  const btn = document.createElement("button");
  btn.textContent = "再試行";
  btn.style.cssText =
    "margin-left:14px;padding:4px 12px;border-radius:6px;" +
    "border:1px solid #fff;background:transparent;color:#fff;cursor:pointer;";
  btn.onclick = () => location.reload();
  el.appendChild(btn);
}

// ---- Google Sign-In ----------------------------------------------------------

function googleSignInRaw() {
  return new Promise((resolve, reject) => {
    const wait = () => {
      if (typeof google === "undefined" || !google.accounts) {
        setTimeout(wait, 100);
        return;
      }
      google.accounts.id.initialize({
        client_id: _clientId,
        use_fedcm_for_prompt: true,
        callback: (resp) => {
          if (resp.credential) resolve(resp.credential);
          else reject(new Error("Google sign-in failed"));
        },
      });
      google.accounts.id.prompt((notification) => {
        if (notification.isNotDisplayed() || notification.isSkippedMoment()) {
          let btnDiv = document.getElementById("google-signin-btn");
          if (!btnDiv) {
            btnDiv = document.createElement("div");
            btnDiv.id = "google-signin-btn";
            btnDiv.style.cssText =
              "position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:200;";
            document.body.appendChild(btnDiv);
          }
          google.accounts.id.renderButton(btnDiv, {
            theme: "outline",
            size: "large",
            text: "signin_with",
          });
        }
      });
    };
    wait();
  });
}

function decodeJwt(token) {
  const payload = token.split(".")[1];
  const binary = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
  // atob yields a byte-per-char "binary string"; reinterpret as UTF-8
  // so multi-byte names (e.g. Japanese) decode correctly instead of
  // landing as Latin-1 mojibake in jwt.name.
  const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
  return JSON.parse(new TextDecoder("utf-8").decode(bytes));
}

// Get a Google credential JWT, popping the picker only if we don't already
// have a fresh one cached for this page session. Returns
// { credential, email, sub, displayName } or null on failure.
export async function ensureGoogleSignIn() {
  if (_cachedCredential) {
    return {
      credential: _cachedCredential,
      email: _cachedEmail,
      sub: _cachedSub,
      displayName: _cachedDisplayName
        || localStorage.getItem(STORAGE_DISPLAY_NAME),
    };
  }
  if (_signInPromise) return _signInPromise;
  _signInPromise = (async () => {
    let credential;
    try {
      credential = await googleSignInRaw();
    } catch (e) {
      console.warn("googleSignIn failed:", e.message);
      showAuthError("Google サインインに失敗しました。もう一度お試しください。");
      return null;
    }
    const jwt = decodeJwt(credential);
    _cachedCredential = credential;
    _cachedEmail = jwt.email;
    _cachedSub = jwt.sub;
    _cachedDisplayName = jwt.name
      || (jwt.email && jwt.email.split("@")[0])
      || null;
    if (_cachedDisplayName) {
      localStorage.setItem(STORAGE_DISPLAY_NAME, _cachedDisplayName);
    }
    const btnDiv = document.getElementById("google-signin-btn");
    if (btnDiv) btnDiv.remove();
    return {
      credential,
      email: _cachedEmail,
      sub: _cachedSub,
      displayName: _cachedDisplayName,
    };
  })();
  try {
    return await _signInPromise;
  } finally {
    _signInPromise = null;
  }
}

// ---- per-Lab register / verify ----------------------------------------------

function withOp(path, opId) {
  if (!opId) return path;
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}operator_id=${encodeURIComponent(opId)}`;
}

// Verify the cached uid for one Lab still resolves on that Lab. Returns
// the user record on success, null if the Lab rejected (uid missing /
// users.json wiped / etc.).
async function verifyLabSession({ opId, labId, uid }) {
  if (!labId || !uid) return null;
  try {
    const lang = _getLang();
    const url = withOp(
      `${API_BASE}/api/auth/check?lang=${encodeURIComponent(lang)}`, opId);
    const r = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        [`X-User-Id-${labId}`]: uid,
      },
    });
    if (!r.ok) return null;
    const data = await r.json();
    if (!data.userId) return null;
    return data;
  } catch (_) {
    return null;
  }
}

// Register the current Google identity against a specific Lab via
// ``POST /api/auth/google?operator_id=<opId>``. Caches the returned uid
// keyed by labId. Returns { userId, displayName, lang } or null on failure.
async function registerWithLab({ opId, labId }) {
  const session = await ensureGoogleSignIn();
  if (!session) return null;
  let resp;
  try {
    resp = await fetch(withOp(`${API_BASE}/api/auth/google`, opId), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: session.email,
        sub: session.sub,
        credential: session.credential,
      }),
    });
  } catch (e) {
    console.warn(`auth/google fetch failed (lab=${labId}):`, e.message);
    return null;
  }
  if (!resp.ok) {
    let detail = "";
    try { detail = (await resp.json()).detail || ""; } catch (_) {}
    if (resp.status === 403) {
      showAuthError(detail || "ログインが許可されていません（mixi.co.jp ドメインのみ）");
    }
    console.warn(`auth/google failed (lab=${labId}):`, resp.status, detail);
    return null;
  }
  const data = await resp.json();
  if (!data.userId) return null;
  setLabUserId(labId, data.userId);
  if (data.displayName) {
    localStorage.setItem(STORAGE_DISPLAY_NAME, data.displayName);
    _cachedDisplayName = data.displayName;
  }
  return data;
}

// Ensure a usable user_id exists for one Lab. Verifies the cached uid
// first; on miss, registers via Google. Returns
// { userId, displayName, lang } or null.
export async function signInWithLab({ opId, labId }) {
  if (!labId) {
    console.warn("signInWithLab: missing labId");
    return null;
  }
  const cached = getUserIdForLab(labId);
  if (cached) {
    const ok = await verifyLabSession({ opId, labId, uid: cached });
    if (ok && ok.userId) {
      if (ok.displayName) {
        localStorage.setItem(STORAGE_DISPLAY_NAME, ok.displayName);
        _cachedDisplayName = ok.displayName;
      }
      return ok;
    }
    // Cached uid no longer valid on this Lab — drop it and re-register.
    clearLabUserId(labId);
  }
  return registerWithLab({ opId, labId });
}
