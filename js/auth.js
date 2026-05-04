// auth.js — Google Sign-In + ensureUser flow for the LT viewer.
//
// Mirrors myboy/js/auth.js but adapted for the professor LT system:
//   - All paths are relative ("/api/...") since the same proxy origin
//     hosts both the frontend and the API.
//   - Domain restriction is enforced server-side (mixi.co.jp only).
//   - The nickname dialog from the original is included; if the markup
//     is missing, ensureUser falls back to using the Google-provided name.
//
// Usage:
//   import { ensureUser, getAuthHeaders, GOOGLE_CLIENT_ID } from './auth.js';
//   const me = await ensureUser({ userId: localStorage.getItem('professor_user_id') });
//   if (!me) { /* sign-in failed or domain rejected */ }

export const GOOGLE_CLIENT_ID =
  "349549531314-2s0rvdf4hsb92l2hae14viintjjpvl5t.apps.googleusercontent.com";

const API_BASE = "";   // relative — proxy/tunnel handles routing
const STORAGE_USER_ID = "professor_user_id";
const STORAGE_DISPLAY_NAME = "professor_display_name";

let _clientId = GOOGLE_CLIENT_ID;
let _getLang = () => "ja";

export function initAuth(deps = {}) {
  if (deps.clientId) _clientId = deps.clientId;
  if (deps.getLang) _getLang = deps.getLang;
}

export function getAuthHeaders(userId) {
  const h = { "Content-Type": "application/json" };
  if (userId) h["X-User-Id"] = userId;
  return h;
}

// Visible, dismissible error overlay. Shown when sign-in fails for any
// reason — wrong domain, server error, network failure, etc.
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

// Wait for window.google to load, then start the One-Tap / popup flow.
function googleSignIn() {
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
          // One-Tap unavailable — fall back to a centered button.
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
  return JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
}

// Optional nickname prompt — uses #nickname-dialog markup if present;
// otherwise resolves with the Google-provided name.
function promptNickname(fallback) {
  return new Promise((resolve) => {
    const dialog = document.getElementById("nickname-dialog");
    const input = document.getElementById("nickname-input");
    const okBtn = document.getElementById("nickname-ok");
    if (!dialog || !input || !okBtn) {
      resolve(fallback);
      return;
    }
    dialog.classList.remove("hidden");
    input.value = fallback || "";
    input.focus();
    const submit = () => {
      const name = input.value.trim();
      if (!name) { input.focus(); return; }
      dialog.classList.add("hidden");
      okBtn.removeEventListener("click", submit);
      input.removeEventListener("keydown", onKey);
      resolve(name);
    };
    const onKey = (e) => { if (e.key === "Enter") submit(); };
    okBtn.addEventListener("click", submit);
    input.addEventListener("keydown", onKey);
  });
}

/**
 * Guarantee the caller has an authenticated user_id.
 * Returns {userId, displayName, lang} on success, or null on failure
 * (e.g. server rejected the domain).
 */
export async function ensureUser(initial = {}) {
  let userId = initial.userId || localStorage.getItem(STORAGE_USER_ID);
  let displayName = initial.displayName || localStorage.getItem(STORAGE_DISPLAY_NAME);

  if (userId) {
    try {
      const lang = _getLang();
      const url = `${API_BASE}/api/auth/check?lang=${encodeURIComponent(lang)}`;
      const r = await fetch(url, { headers: getAuthHeaders(userId) });
      if (r.ok) {
        const data = await r.json();
        if (data.userId) {
          displayName = data.displayName || displayName;
          if (displayName) localStorage.setItem(STORAGE_DISPLAY_NAME, displayName);
          return { userId, displayName, lang: data.lang || lang };
        }
      }
    } catch (_) { /* fall through to fresh sign-in */ }
    // Stale local id — clear it.
    localStorage.removeItem(STORAGE_USER_ID);
    userId = null;
  }

  // Fresh Google Sign-In.
  let credential;
  try {
    credential = await googleSignIn();
  } catch (e) {
    console.warn("googleSignIn failed:", e.message);
    showAuthError("Google サインインに失敗しました。もう一度お試しください。");
    return null;
  }
  const jwt = decodeJwt(credential);
  const email = jwt.email;
  const googleName = jwt.name || (email && email.split("@")[0]);

  const btnDiv = document.getElementById("google-signin-btn");
  if (btnDiv) btnDiv.remove();

  // First call — see if the user already exists server-side.
  let resp;
  try {
    resp = await fetch(`${API_BASE}/api/auth/google`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, sub: jwt.sub, credential }),
    });
  } catch (e) {
    console.warn("auth/google fetch failed:", e.message);
    showAuthError("認証サーバへ接続できませんでした：" + e.message);
    return null;
  }

  if (!resp.ok) {
    let detail = "";
    try { detail = (await resp.json()).detail || ""; } catch (_) {}
    let msg;
    if (resp.status === 403) {
      msg = detail || "ログインが許可されていません（mixi.co.jp ドメインのみ）";
    } else if (resp.status === 401) {
      msg = detail || "認証に失敗しました（資格情報が無効です）";
    } else {
      msg = `認証サーバとの通信に失敗しました（HTTP ${resp.status}${detail ? ": " + detail : ""}）`;
    }
    showAuthError(msg);
    console.warn("auth/google failed:", resp.status, detail);
    return null;
  }

  let data = await resp.json();
  if (data.userId && data.displayName) {
    localStorage.setItem(STORAGE_USER_ID, data.userId);
    localStorage.setItem(STORAGE_DISPLAY_NAME, data.displayName);
    return { userId: data.userId, displayName: data.displayName, lang: data.lang || _getLang() };
  }

  // (Reserved branch — server currently auto-registers on first POST, so we
  // shouldn't reach here. Kept for future use of an explicit nickname step.)
  const nick = await promptNickname(googleName);
  if (!nick) return null;
  try {
    const r2 = await fetch(`${API_BASE}/api/auth/google`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, sub: jwt.sub, credential, displayName: nick, lang: _getLang() }),
    });
    if (!r2.ok) return null;
    data = await r2.json();
    localStorage.setItem(STORAGE_USER_ID, data.userId);
    localStorage.setItem(STORAGE_DISPLAY_NAME, data.displayName);
    return { userId: data.userId, displayName: data.displayName, lang: data.lang || _getLang() };
  } catch (e) {
    console.warn("auth/google register failed:", e.message);
    return null;
  }
}
