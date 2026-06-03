#!/usr/bin/env python3
"""facr_batch.py — 50 クエリの自動 ASK + FACR バッチ実行スクリプト.

Usage:
    # ブラウザ DevTools Console で localStorage.professor_user_ids を取得:
    #   → '{"IPSJ-Network":"uuid-aaa","Arxiv-A":"uuid-bbb"}'

    python facr_batch.py \
        --base-url https://d2bbeowpg2f545.cloudfront.net \
        --user-ids '{"IPSJ-Network":"uuid-aaa","Arxiv-A":"uuid-bbb"}'

環境変数でも指定可能:
    FACR_BASE_URL, FACR_USER_IDS

処理フロー (1 クエリあたり):
  1. /api/tunnel/info で接続中の Lab (role=gameserver) を検出
  2. localStorage の user_ids で認証済みセッションを再利用
  3. 全 Lab に /api/chat (stage=qa) で ASK
  4. 各 Lab に facr_baseline → prior_evidence をシード
  5. K_MAX ラウンド cross_examine (逐次) → Δ 蓄積
  6. facr-trace-v2 JSON を生成

結果: facr_batch_results.json (JSON Array) として保存
"""

import argparse
import json
import logging
import os
import ssl
import sys
import time
import uuid
import urllib.parse
import urllib.request
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("facr_batch")

# ── FACR パラメータ (professor.js と同一) ──────────────────────
FACR_K_MAX = 2
FACR_EPSILON = 0.05
FACR_DOMINANCE = 1.5

# ── 50 クエリ定義 ─────────────────────────────────────────────
QUERIES = [
    # Group A: 強α (20本) — RFC vs ML
    {"group": "A", "idx": 1,  "text": "QUIC導入で新たな攻撃面は何か"},
    {"group": "A", "idx": 2,  "text": "TCP RenoとBBRの公平性問題"},
    {"group": "A", "idx": 3,  "text": "RTPパケットロス予測にMLは有効か"},
    {"group": "A", "idx": 4,  "text": "WebRTC congestion control改善手法"},
    {"group": "A", "idx": 5,  "text": "SFUにおける再送最適化"},
    {"group": "A", "idx": 6,  "text": "QUIC 0-RTTのセキュリティ課題"},
    {"group": "A", "idx": 7,  "text": "AV1 over RTPの実運用課題"},
    {"group": "A", "idx": 8,  "text": "Wi-Fi環境でのWebRTC品質改善"},
    {"group": "A", "idx": 9,  "text": "TURNサーバ負荷削減手法"},
    {"group": "A", "idx": 10, "text": "NAT越え成功率向上策"},
    {"group": "A", "idx": 11, "text": "DTLSハンドシェイク短縮手法"},
    {"group": "A", "idx": 12, "text": "RTP jitter buffer設計"},
    {"group": "A", "idx": 13, "text": "FECとARQのトレードオフ"},
    {"group": "A", "idx": 14, "text": "SRTP鍵交換方式比較"},
    {"group": "A", "idx": 15, "text": "ICE candidate pruning手法"},
    {"group": "A", "idx": 16, "text": "RTP congestion feedback設計"},
    {"group": "A", "idx": 17, "text": "RTP over QUICの利点"},
    {"group": "A", "idx": 18, "text": "WebTransportとWebRTC比較"},
    {"group": "A", "idx": 19, "text": "SFUクラスタリング方式比較"},
    {"group": "A", "idx": 20, "text": "モバイル回線でのAV1利用課題"},
    # Group B: 中α (15本) — Security vs Network
    {"group": "B", "idx": 1,  "text": "TLS1.3は企業ネットワーク監査を困難にするか"},
    {"group": "B", "idx": 2,  "text": "QUIC暗号化はトラフィック分析を防げるか"},
    {"group": "B", "idx": 3,  "text": "VPN利用による攻撃面の変化"},
    {"group": "B", "idx": 4,  "text": "Zero Trust導入の課題"},
    {"group": "B", "idx": 5,  "text": "DDoS対策におけるAnycast利用"},
    {"group": "B", "idx": 6,  "text": "DNS over HTTPSの利点と欠点"},
    {"group": "B", "idx": 7,  "text": "WebRTCのIPリーク問題"},
    {"group": "B", "idx": 8,  "text": "NATがセキュリティに与える影響"},
    {"group": "B", "idx": 9,  "text": "E2EEと監査要件の両立"},
    {"group": "B", "idx": 10, "text": "QUIC fingerprintingの可能性"},
    {"group": "B", "idx": 11, "text": "Browser sandboxの限界"},
    {"group": "B", "idx": 12, "text": "WebAssemblyセキュリティリスク"},
    {"group": "B", "idx": 13, "text": "OAuth認証の代表的脆弱性"},
    {"group": "B", "idx": 14, "text": "Public Wi-Fi利用リスク"},
    {"group": "B", "idx": 15, "text": "Secure RTPの運用課題"},
    # Group C: 弱α (15本) — 同一ドメイン近傍
    {"group": "C", "idx": 1,  "text": "RTTとスループットの関係"},
    {"group": "C", "idx": 2,  "text": "TCPとQUICの違い"},
    {"group": "C", "idx": 3,  "text": "パケットロスとは何か"},
    {"group": "C", "idx": 4,  "text": "WebRTCの基本構成"},
    {"group": "C", "idx": 5,  "text": "RTPの役割"},
    {"group": "C", "idx": 6,  "text": "NATとは何か"},
    {"group": "C", "idx": 7,  "text": "TCP輻輳制御とは何か"},
    {"group": "C", "idx": 8,  "text": "BBRとは何か"},
    {"group": "C", "idx": 9,  "text": "TURNサーバの役割"},
    {"group": "C", "idx": 10, "text": "ICEの目的"},
    {"group": "C", "idx": 11, "text": "SRTPとは何か"},
    {"group": "C", "idx": 12, "text": "TLS1.3の特徴"},
    {"group": "C", "idx": 13, "text": "HTTP/3の特徴"},
    {"group": "C", "idx": 14, "text": "RTP jitterとは何か"},
    {"group": "C", "idx": 15, "text": "WebTransportとは何か"},
]


# ── HTTP ヘルパー (stdlib のみ) ──────────────────────────────
# macOS の certifi を使う; なければ未検証にフォールバック
_ssl_ctx = ssl.create_default_context()
try:
    import certifi  # type: ignore
    _ssl_ctx.load_verify_locations(certifi.where())
except Exception:
    pass


def _http(method: str, url: str, headers: dict,
          body: dict | None = None, timeout: float = 120) -> dict:
    """stdlib urllib で JSON request/response."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=_ssl_ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {e.code} {method} {url}: {detail[:300]}"
        ) from e


# ── ヘルパー ─────────────────────────────────────────────────
class FACRClient:
    """FACR バッチ実行クライアント."""

    def __init__(self, base_url: str, user_ids: dict[str, str]):
        self.base_url = base_url.rstrip("/")
        # lab_id -> user_id (localStorage.professor_user_ids と同じ形式)
        self.user_ids: dict[str, str] = dict(user_ids)
        # operator_id -> { lab_id, label }
        self.labs: dict[str, dict] = {}

    # ── API helpers ───────────────────────────────────────────

    def _url(self, path: str, operator_id: str = "") -> str:
        url = f"{self.base_url}{path}"
        if operator_id:
            sep = "&" if "?" in url else "?"
            url += f"{sep}operator_id={urllib.parse.quote(operator_id)}"
        return url

    def _auth_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        for lab_id, uid in self.user_ids.items():
            h[f"X-User-Id-{lab_id}"] = uid
        return h

    def _post(self, path: str, body: dict, operator_id: str = "",
              timeout: float = 120) -> dict:
        url = self._url(path, operator_id)
        return _http("POST", url, self._auth_headers(), body, timeout)

    def _get(self, path: str, operator_id: str = "",
             timeout: float = 30) -> dict:
        url = self._url(path, operator_id)
        return _http("GET", url, self._auth_headers(), timeout=timeout)

    # ── 1. Lab 検出 ──────────────────────────────────────────

    def discover_labs(self):
        """GET /api/tunnel/info → gameserver operator 一覧."""
        data = self._get("/api/tunnel/info")
        ops = [o for o in (data.get("operators") or [])
               if o.get("role") == "gameserver"]
        if not ops:
            raise RuntimeError("gameserver operator が見つかりません")
        self.labs = {}
        for op in ops:
            lab = op.get("lab") or {}
            op_id = op["id"]
            lab_id = lab.get("lab_id", "")
            label = lab.get("name") or lab_id or op.get("name") or op_id[:6]
            self.labs[op_id] = {"lab_id": lab_id, "label": label}
        log.info("Labs discovered: %s",
                 {v["label"]: k[:8] for k, v in self.labs.items()})

    # ── 2. セッション確認 ──────────────────────────────────────

    def verify_sessions(self):
        """各 Lab の user_id が有効か /api/auth/check で確認."""
        for op_id, info in self.labs.items():
            lab_id = info["lab_id"]
            uid = self.user_ids.get(lab_id)
            if not uid:
                log.warning("Lab %s (%s): user_id なし — スキップ",
                            info["label"], lab_id)
                continue
            try:
                resp = self._get(f"/api/auth/check?lang=ja",
                                 operator_id=op_id)
                name = resp.get("displayName", "?")
                log.info("  ✓ %s (lab_id=%s) uid=%s name=%s",
                         info["label"], lab_id, uid[:8], name)
            except Exception as e:
                log.warning("  ✗ %s (lab_id=%s): session invalid: %s",
                            info["label"], lab_id, e)
                log.warning("    → この Lab は FACR 対象外になります")
                del self.user_ids[lab_id]

    # ── 3. ASK (fan-out) ─────────────────────────────────────

    def ask(self, question: str, group_id: str) -> dict[str, dict]:
        """全 Lab に ASK. Returns {op_id: ChatResponse}."""
        results = {}
        for op_id, info in self.labs.items():
            lab_id = info["lab_id"]
            uid = self.user_ids.get(lab_id)
            if not uid:
                log.warning("Skipping ASK to %s (no uid)", info["label"])
                continue
            body = {
                "user_id": uid,
                "stage": "qa",
                "message": question,
                "group_id": group_id,
            }
            try:
                resp = self._post("/api/chat", body, operator_id=op_id,
                                  timeout=180)
                results[op_id] = resp
                acc = resp.get("accuracy") or {}
                log.info("  ASK → %s: reply=%d chars, top_score=%.3f",
                         info["label"], len(resp.get("reply", "")),
                         float(acc.get("top_score", 0)))
            except Exception as e:
                log.error("  ASK → %s failed: %s", info["label"], e)
        return results

    # ── 4. FACR baseline ─────────────────────────────────────

    def facr_baseline(self, question: str, group_id: str) -> list[str]:
        """各 Lab の facr_baseline を呼び prior_evidence をシード."""
        prior: list[str] = []
        seen: set[str] = set()
        for op_id, info in self.labs.items():
            lab_id = info["lab_id"]
            uid = self.user_ids.get(lab_id)
            if not uid:
                continue
            body = {
                "user_id": uid,
                "stage": "qa",
                "message": question,
                "group_id": group_id,
                "action": "facr_baseline",
            }
            try:
                resp = self._post("/api/chat", body, operator_id=op_id,
                                  timeout=60)
                evidence = (resp.get("accuracy") or {}).get("evidence") or []
                for e in evidence:
                    t = (e.get("text") or "").strip()
                    if t and t not in seen:
                        seen.add(t)
                        prior.append(t)
                log.info("  baseline → %s: %d chunks", info["label"],
                         len(evidence))
            except Exception as e:
                log.warning("  baseline → %s failed: %s", info["label"], e)
        return prior

    # ── 5. FACR cross_examine ────────────────────────────────

    def run_facr(self, question: str, group_id: str,
                 ask_results: dict[str, dict]) -> dict:
        """FACR を実行し facr-trace-v2 JSON を返す."""
        # Lab 情報構築
        labs = []
        for op_id, resp in ask_results.items():
            info = self.labs[op_id]
            answer = (resp.get("reply") or "").strip()
            if not answer:
                continue
            acc = resp.get("accuracy") or {}
            labs.append({
                "op_id": op_id,
                "lab_id": info["lab_id"],
                "label": info["label"],
                "answer": answer,
                "initial_top": float(acc.get("top_score", 0)),
            })
        if len(labs) < 2:
            log.warning("  FACR skipped: need >= 2 Labs with answers, got %d",
                        len(labs))
            return {}

        # baseline seed
        log.info("  FACR baseline seeding…")
        prior_evidence = self.facr_baseline(question, group_id)
        log.info("  FACR seeded %d chunks across %d Labs",
                 len(prior_evidence), len(labs))

        cumulative = {lab["op_id"]: 0.0 for lab in labs}
        prev_deltas = {lab["op_id"]: 0.0 for lab in labs}
        rounds_log = []
        stop_reason = "k_max"
        stop_gap = 0.0

        for k in range(1, FACR_K_MAX + 1):
            log.info("  FACR round %d/%d", k, FACR_K_MAX)
            per_lab_delta = {lab["op_id"]: 0.0 for lab in labs}
            per_lab_stances = {lab["op_id"]: [] for lab in labs}
            per_lab_ev = {lab["op_id"]: 0 for lab in labs}
            new_evidence_texts = []
            examinations = []

            # 逐次実行 (proxy tunnel の idle timeout 対策)
            for examiner in labs:
                for peer in labs:
                    if peer["op_id"] == examiner["op_id"]:
                        continue
                    lab_id = self.labs[examiner["op_id"]]["lab_id"]
                    uid = self.user_ids.get(lab_id)
                    if not uid:
                        continue
                    body = {
                        "user_id": uid,
                        "stage": "qa",
                        "message": question,
                        "group_id": group_id,
                        "action": "cross_examine",
                        "claim": peer["answer"],
                        "prior_evidence": prior_evidence,
                        "round_idx": k,
                    }
                    try:
                        resp = self._post("/api/chat", body,
                                          operator_id=examiner["op_id"],
                                          timeout=180)
                        acc = resp.get("accuracy") or {}
                        delta = float(acc.get("delta", 0))
                        per_lab_delta[examiner["op_id"]] += delta
                        stances = acc.get("stances") or []
                        per_lab_stances[examiner["op_id"]].extend(stances)
                        evidence = acc.get("evidence") or []
                        per_lab_ev[examiner["op_id"]] += len(evidence)
                        for e in evidence:
                            t = (e.get("text") or "").strip()
                            if t:
                                new_evidence_texts.append(t)
                        examinations.append({
                            "examiner_lab": examiner["lab_id"],
                            "examiner_label": examiner["label"],
                            "target_lab": peer["lab_id"],
                            "target_label": peer["label"],
                            "claim_excerpt": peer["answer"][:240],
                            "query_original": acc.get("query_original", ""),
                            "query_augmented": acc.get("query_augmented", ""),
                            "delta": delta,
                            "note": acc.get("note", ""),
                            "evidence": [
                                {
                                    "relevance": float(e.get("relevance", 0)),
                                    "novelty": float(e.get("novelty", 0)),
                                    "stance": e.get("stance", "NEUTRAL"),
                                    "contrib": float(e.get("delta", 0)),
                                    "score": float(e.get("score", 0)),
                                    "title": e.get("title", ""),
                                    "source": e.get("source", ""),
                                    "text_excerpt": (e.get("text") or "")[:200],
                                }
                                for e in evidence
                            ],
                        })
                        log.info("    %s examines %s claim: Δ=%+.3f",
                                 examiner["label"], peer["label"], delta)
                    except Exception as e:
                        log.error("    cross_examine %s→%s failed: %s",
                                  examiner["label"], peer["label"], e)

            # ラウンドログ
            def _summarize_stances(stances):
                s = sum(1 for x in stances if x == "SUPPORT")
                r = sum(1 for x in stances if x == "REBUT")
                n = len(stances) - s - r
                return {"support": s, "rebut": r, "neutral": n}

            round_entry = {
                "k": k,
                "per_lab": [],
                "examinations": examinations,
            }
            saturated = True
            for lab in labs:
                oid = lab["op_id"]
                d = per_lab_delta[oid]
                cumulative[oid] += d
                prev = prev_deltas[oid]
                if abs(d - prev) >= FACR_EPSILON:
                    saturated = False
                prev_deltas[oid] = d
                round_entry["per_lab"].append({
                    "labId": lab["lab_id"],
                    "label": lab["label"],
                    "delta": d,
                    "stance_summary": _summarize_stances(
                        per_lab_stances[oid]),
                    "ev_count": per_lab_ev[oid],
                })
            rounds_log.append(round_entry)

            # prior_evidence 拡張 (重複排除)
            seen = set(prior_evidence)
            for t in new_evidence_texts:
                if t not in seen:
                    seen.add(t)
                    prior_evidence.append(t)

            # dominance check
            sorted_cum = sorted(cumulative.values(), reverse=True)
            gap = (sorted_cum[0] - sorted_cum[1]) if len(sorted_cum) >= 2 else 0
            if gap >= FACR_DOMINANCE:
                log.info("  FACR: dominant winner at round %d (gap=%.2f)", k, gap)
                stop_reason = "dominance"
                stop_gap = gap
                break
            if saturated and k >= 2:
                log.info("  FACR: saturated at round %d", k)
                stop_reason = "saturation"
                stop_gap = gap
                break
            if k == FACR_K_MAX:
                stop_gap = gap

        # ── facr-trace-v2 JSON 生成 (buildFacrTrace 互換) ────
        # claim_conf: target 側に Δ を再集約 (paper §4.5)
        claim_conf = {lab["lab_id"]: 0.0 for lab in labs}
        for rnd in rounds_log:
            for ex in rnd.get("examinations", []):
                target = ex.get("target_lab", "")
                if target in claim_conf:
                    claim_conf[target] += float(ex.get("delta", 0))
        sorted_claim = sorted(claim_conf.items(), key=lambda x: x[1],
                              reverse=True)
        claim_winner = sorted_claim[0][0] if sorted_claim else None

        # examiner 側 cumulative (実装ネイティブ)
        cum_by_lab = {}
        for lab in labs:
            cum_by_lab[lab["lab_id"]] = cumulative[lab["op_id"]]
        sorted_corpus = sorted(cum_by_lab.items(), key=lambda x: x[1],
                               reverse=True)
        most_supportive = sorted_corpus[0][0] if sorted_corpus else None

        trace = {
            "schema": "facr-trace-v2",
            "ts": datetime.now(timezone.utc).isoformat(),
            "group_id": group_id,
            "question": question,
            "labs": [
                {"opId": l["op_id"], "labId": l["lab_id"], "label": l["label"]}
                for l in labs
            ],
            "params": {
                "k_max": FACR_K_MAX,
                "epsilon": FACR_EPSILON,
                "tau_dom": FACR_DOMINANCE,
            },
            "rounds": [
                {
                    "k": r["k"],
                    "per_lab": r["per_lab"],
                    "examinations": r["examinations"],
                }
                for r in rounds_log
            ],
            "stop": {
                "reason": stop_reason,
                "at_round": len(rounds_log),
                "gap": round(stop_gap, 4),
            },
            "cumulative": cum_by_lab,
            "most_supportive_corpus_lab": most_supportive,
            "claim_conf": claim_conf,
            "claim_winner_lab": claim_winner,
        }
        return trace


# ── メイン ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FACR バッチ実行 (50 クエリ × ASK + FACR)")
    parser.add_argument("--base-url",
                        default=os.environ.get("FACR_BASE_URL",
                            "https://d2bbeowpg2f545.cloudfront.net"),
                        help="プロキシの base URL")
    parser.add_argument("--user-ids",
                        default=os.environ.get("FACR_USER_IDS", ""),
                        help='localStorage.professor_user_ids の値 '
                             '(JSON: {"lab_id":"user_id",...})')
    parser.add_argument("--output", default="facr_batch_results.json",
                        help="出力ファイル名 (default: facr_batch_results.json)")
    parser.add_argument("--start", type=int, default=0,
                        help="開始インデックス (途中再開用, 0-indexed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="クエリ一覧を表示して終了")
    args = parser.parse_args()

    if args.dry_run:
        for i, q in enumerate(QUERIES):
            print(f"  [{i:2d}] Group {q['group']}-{q['idx']:2d}: {q['text']}")
        print(f"\n  Total: {len(QUERIES)} queries")
        return

    if not args.base_url:
        print("ERROR: --base-url required", file=sys.stderr)
        sys.exit(1)
    if not args.user_ids:
        print("ERROR: --user-ids required", file=sys.stderr)
        print("  Chrome DevTools Console で以下を実行:", file=sys.stderr)
        print('    localStorage.getItem("professor_user_ids")', file=sys.stderr)
        print("  出力された JSON 文字列をそのまま --user-ids に渡してください",
              file=sys.stderr)
        sys.exit(1)

    try:
        user_ids = json.loads(args.user_ids)
    except json.JSONDecodeError as e:
        print(f"ERROR: --user-ids の JSON パースに失敗: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(user_ids, dict) or not user_ids:
        print("ERROR: --user-ids は {lab_id: user_id} の JSON オブジェクトが必要",
              file=sys.stderr)
        sys.exit(1)

    log.info("User IDs loaded: %s",
             {k: v[:8] + "…" for k, v in user_ids.items()})

    client = FACRClient(base_url=args.base_url, user_ids=user_ids)

    # Lab 検出 + セッション確認
    log.info("=== Lab discovery ===")
    client.discover_labs()
    log.info("=== Session verification ===")
    client.verify_sessions()

    if not client.user_ids:
        log.error("No valid sessions — aborting")
        log.error("  ブラウザでログインし直してから professor_user_ids を再取得してください")
        sys.exit(1)

    # 途中再開: 既存結果を読み込み
    results: list[dict] = []
    if args.start > 0 and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            results = json.load(f)
        log.info("Loaded %d existing results from %s", len(results), args.output)

    # バッチ実行
    total = len(QUERIES)
    for i, q in enumerate(QUERIES):
        if i < args.start:
            continue

        query_label = f"Group {q['group']}-{q['idx']:02d}"
        log.info("=== [%d/%d] %s: %s ===", i + 1, total, query_label, q["text"])

        group_id = str(uuid.uuid4())

        # ASK
        log.info("  Phase 1: ASK")
        ask_results = client.ask(q["text"], group_id)
        if len(ask_results) < 2:
            log.warning("  Only %d Lab(s) answered — skipping FACR",
                        len(ask_results))
            results.append({
                "query_group": q["group"],
                "query_idx": q["idx"],
                "query_text": q["text"],
                "group_id": group_id,
                "ask_results": {
                    client.labs[op_id]["label"]: {
                        "reply": (r.get("reply") or "")[:500],
                        "top_score": float(
                            (r.get("accuracy") or {}).get("top_score", 0)),
                        "mean_score": float(
                            (r.get("accuracy") or {}).get("mean_score", 0)),
                    }
                    for op_id, r in ask_results.items()
                },
                "facr_trace": None,
                "error": f"only {len(ask_results)} Lab(s) answered",
            })
            _save_intermediate(results, args.output)
            continue

        # FACR
        log.info("  Phase 2: FACR")
        try:
            trace = client.run_facr(q["text"], group_id, ask_results)
        except Exception as e:
            log.error("  FACR failed: %s", e)
            trace = {"error": str(e)}

        # ASK 結果サマリ
        ask_summary = {}
        for op_id, r in ask_results.items():
            lab_label = client.labs[op_id]["label"]
            acc = r.get("accuracy") or {}
            ask_summary[lab_label] = {
                "reply": (r.get("reply") or "")[:500],
                "top_score": float(acc.get("top_score", 0)),
                "mean_score": float(acc.get("mean_score", 0)),
            }

        results.append({
            "query_group": q["group"],
            "query_idx": q["idx"],
            "query_text": q["text"],
            "group_id": group_id,
            "ask_results": ask_summary,
            "facr_trace": trace,
        })

        # 中間保存 (crash 対策)
        _save_intermediate(results, args.output)
        log.info("  → saved (%d/%d complete)", len(results), total)

        # Lab サーバへの負荷軽減用ウェイト
        if i < total - 1:
            log.info("  waiting 3s before next query…")
            time.sleep(3)

    log.info("=== Done: %d results saved to %s ===", len(results), args.output)


def _save_intermediate(results: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
