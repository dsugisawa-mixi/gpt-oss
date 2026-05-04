# Professor LT System -- 設計ドキュメント

ラボ／グループが持つ**高度なドメイン知識なしには読み解けない局所知識** (論文等に書かれてはいるが、行間を読むには専門的文脈が必要な実装判断・実験設計の意図・失敗から得た知見) を、RAG + 遊休 GPU + WebSocket トンネルで**セキュアに社内公開**する知識流通インフラ。

各ラボ／グループは NAT 内の GPU マシン上で独自 RAG を保持し、データを外に出さずに知識 API として機能する。フロントエンドでは、各ラボ／グループの局所知識サマリーからノードを選択し、そのラボが保持するスライド化されたペーパーを検索・閲覧できる。選択したラボ／グループとの Q&A を通じて局所知識に特化した知見を得ることができ、3D アバター付きライブ発表として具現化される。ペーパーのアップロード・削除は当該ラボ／グループによって精査・管理される。

## システム概要

<img src="./professor.svg" style="max-height: 50vh; width: auto;" />


<div style="page-break-before: always;"></div>

## ファイル構成と責務

| ファイル | 役割 |
|---------|------|
| `your_professor_server.py` | FastAPI バックエンド。チャット生成、認証、ファイルサーブ、アップロードパイプライン管理 |
| `paper_rag.py` | ベクトル検索インターフェース。Qwen3-Embedding + LanceDB + CrossEncoder リランカー |
| `build_paper_index.py` | オフラインバッチインデクサー。PDF/MD/TeX を発見→チャンク→埋め込み→永続化 |
| `html/professor.html` | ブラウザ UI コンテナ。3 カラムレイアウト (アバター / スライド / Q&A) |
| `html/js/professor.js` | フロントエンドロジック。TTS 再生、アバターアニメーション、スライド遷移、アップロード UI |
| `html/js/auth.js` | Google OAuth ラッパー。ドメイン制限付きサインイン、セッション管理 |
| `~/git/paper/myboy/aws/proxy_server.py` | AWS 上のリバースプロキシ。WebSocket トンネル管理、HTTP/ストリームリレー |
| `~/git/paper/myboy/aws/tunnel_client.py` | NAT 内から Proxy へ outbound WebSocket 接続、ローカルサーバーへリクエスト中継 |
| `~/git/Qwen3-TTS-streaming/server-design.py` | TTS ストリーミングサーバー。PCM 音声生成・配信 |


<div style="page-break-before: always;"></div>

## API エンドポイント
### チャット・設定

| Method | Path | 説明 |
|--------|------|------|
| POST | `/api/chat` | LLM によるステージ別テキスト生成 (waiting/presenting/qa/closing) |
| GET | `/api/config` | TTS URL 等のブートストラップ設定 |
| GET | `/api/deck` | 静的スライドデッキ JSON (deck.json) |
| GET | `/api/history` | ユーザー別会話履歴 |
| POST | `/api/reset` | セッション履歴クリア |

### 認証

| Method | Path | 説明 |
|--------|------|------|
| POST | `/api/auth/google` | Google OAuth サインイン (mixi.co.jp + カスタム許可リスト) |
| GET | `/api/auth/check` | セッション検証 |
| POST | `/api/auth/logout` | サインアウト |

### プレゼンス・Q&A

| Method | Path | 説明 |
|--------|------|------|
| GET | `/api/presence/users` | 参加者一覧 (オンライン状態) |
| POST | `/api/presence/heartbeat` | ハートビート (10 秒ポーリング) |
| GET | `/api/qa/timeline` | Q&A タイムライン取得 |
| PUT | `/api/qa/{qa_id}` | Q&A エントリ編集 (発表者のみ) |
| DELETE | `/api/qa/{qa_id}` | Q&A エントリ削除 |

### 論文アップロード

| Method | Path | 説明 |
|--------|------|------|
| POST | `/api/upload/presign` | S3 署名付き URL 発行 + ジョブ作成 |
| POST | `/api/upload/start` | チケット消費 → PDF ダウンロード → パイプライン開始 |
| GET | `/api/upload/jobs/{job_id}` | ジョブステータス + ログテール |
| GET | `/api/upload/jobs/{job_id}/slides` | 生成済み HTML スライド |
| GET | `/api/upload/jobs/{job_id}/script` | スピーカースクリプト JSON |
| GET | `/api/upload/papers` | 登録済み論文一覧 |
| POST | `/api/upload/papers/{job_id}/select` | 論文をアクティブに設定 |
| DELETE | `/api/upload/papers/{job_id}` | 論文削除 |
| GET | `/api/upload/eligibility` | アップロード/削除チケット残数 |
| GET | `/api/meta` | 現在の発表メタデータ (テーマ/発表者/会場) |

### TTS プロキシ

| Method | Path | 説明 |
|--------|------|------|
| POST | `/api/tts/generate_stream` | 外部 TTS バックエンドへのストリーミングプロキシ (PCM 音声) |


<div style="page-break-before: always;"></div>

## ラボ／グループの知識ノードをセキュアにシェアする

<img src="./professor-nat.svg" style="max-height: 90vh; width: auto;" />

各ラボ／グループは、論文には書かれない**局所的で尖った知識**を持っている:

| ラボ／グループ | 保有知識の例 |
|--------|-------------|
| RealTime/SFU 特化ラボ／グループ | パケットロス時の再送戦略、SFU 実装の勘所 |
| 音声認識ラボ／グループ | ノイズ環境での前処理パイプライン、失敗した手法のログ |
| ネットワーク QoS ラボ／グループ | 実測ベースの帯域制御パラメータ、ベンダー固有の挙動 |

LLM は平均化された知識を返す。ラボ／グループ RAG は**濃度の高い局所知識**を返す。
この差が価値の源泉であり、本システムはそれを**セキュアに外部へシェアする**ためのインフラである。

**設計原則:**

- **データはラボ／グループの外に出ない** — 各ノードが自身の RAG とアクセス制御を保持し、Proxy はリレーするだけ
- **Authority 付き応答** — 「A ラボ／グループの知見です」と出典が付く。出所不明の RAG 回答とは信頼性が段違い
- **ノードの自律性** — ラボ／グループごとに公開範囲・参加・離脱を自己決定できる

**実現手段 — WebSocket トンネル:**

ラボ／グループの GPU マシンは NAT/ファイアウォール内にある。
各ノードが NAT 内から **outbound WebSocket** でクラウド Proxy に接続することで、
インバウンドポート開放や VPN なしに、ラボ／グループのセキュリティポリシーを壊さず公開できる。

| 設計選択 | 効果 |
|----------|------|
| NAT 内から outbound WebSocket | ファイアウォール変更不要 |
| AWS ELB + Proxy が TLS 終端 | 証明書管理をクラウド側に集約 |
| Operator ID ルーティング | DNS 変更なしで知識ノードを動的に追加・差し替え |
| 各ノードが GPU を自前保持 | ラボ／グループの遊休 GPU を活用、クラウド GPU コストは Proxy の EC2 のみ |

**拡張: 分散 RAG ネットワークへ**

現在は単一ノード構成だが、Operator ID ルーティングにより複数ノードへ自然に拡張できる:

1. クエリが Proxy に到着
2. クエリ内容から最適な知識ノード (ラボ／グループ) にルーティング
3. 該当ノードが自身の RAG で検索・応答
4. 応答に Authority (出典ラボ／グループ) を付与

これは検索エンジンではなく、**専門知識の流通インフラ** である。

### 接続フロー

1. `tunnel_client.py` が NAT 内から `proxy_server.py` へ WebSocket 接続 (outbound)
2. `register` アクションで operator_id (UUID) + display_name を登録
3. Proxy 側の `_tunnels[operator_id]` にトンネルが登録される
4. ブラウザは Proxy の `/api/tunnel/info` でリレー先一覧を取得
5. リクエスト時に `X-Operator-ID` ヘッダー or `?operator_id=` で対象トンネルを指定


### リクエストリレー

| 種別 | 方式 | 用途 |
|------|------|------|
| 制御 (RPC) | Proxy が Future 生成 → トンネル経由で forward → レスポンスで Future 解決 | `/api/chat`, `/api/auth/*` 等 |
| ストリーム | Channel + Queue ベース、複数ブラウザで同一チャネルを共有 (マルチキャスト) | `/api/tts/generate_stream`, `/api/video/stream` |

制御リクエストのリレー:

<div style="page-break-before: always;"></div>

<img src="./professor-relay.svg" style="max-height: 90vh; width: auto;" />

### Operator ルーティング

| 優先度 | 方式 |
|--------|------|
| 1 | `X-Operator-ID` ヘッダー |
| 2 | `?operator_id=` クエリパラメータ |
| 3 | パスベースの role マッチング (`/api/video/*` → device, その他 → gameserver) |
| 4 | フォールバック: 最初の利用可能なトンネル |


<div style="page-break-before: always;"></div>

## コアデータフロー
### ライブ発表フロー

<img src="./professor-live-flow.svg" style="max-height: 90vh; width: auto;" />

<div style="page-break-before: always;"></div>

### 挙手 (Q&A) フロー

<img src="./professor-qa-flow.svg" style="max-height: 90vh; width: auto;" />

<div style="page-break-before: always;"></div>

### 論文アップロードパイプライン

<img src="./professor-upload-pipeline.svg" style="max-height: 90vh; width: auto;" />

<div style="page-break-before: always;"></div>

### RAG インデックス再構築

<img src="./professor-rag-index-rebuild.svg" style="max-height: 90vh; width: auto;" />


<div style="page-break-before: always;"></div>

## RAG アーキテクチャ

### インデックス構築 (build_paper_index.py)

| ステップ | 詳細 |
|---------|------|
| 発見 | `~/git/paper` (自分の論文・メモ + `external-pdf-for-rag/` 内の外部論文) + `professor_data/uploads` を再帰走査。PDF/MD/TeX/TXT 対象 |
| 分類 | パスからメタデータ推定: doc_type (paper/preprint/patent/memo/tool/other), topic, title |
| テキスト抽出 | PDF: PyMuPDF、MD/TeX/TXT: UTF-8 読み込み |
| チャンク分割 | 段落認識、~600 トークン/チャンク、~100 トークンオーバーラップ |
| 埋め込み | Qwen/Qwen3-Embedding-0.6B、バッチサイズ 8、正規化あり |
| 永続化 | LanceDB `chunks` テーブル (upsert) + `meta.json` マニフェスト |

### 検索 (paper_rag.py)

```
search(query, top_k=5)
  ├── Qwen3-Embedding-0.6B でクエリ埋め込み (CPU)
  ├── LanceDB ベクトル検索 (top_k=10 オーバーサンプル)
  ├── CrossEncoder リランク (BAAI/bge-reranker-v2-m3, CPU)
  └── 上位 top_k 件を返却: [{title, source, text, score}]
```

- **デバイス戦略**: 埋め込み・リランクともに CPU (vLLM GPU と競合回避)
- **スレッド安全**: `_swap_lock` でリロード中の検索は旧スナップショットを参照
- **グレースフルデグラデーション**: インデックス未構築時は空リスト返却、LLM はスライド内容のみで生成


<div style="page-break-before: always;"></div>

## LLM プロンプト設計

### システムプロンプト

ペルソナは論文ごとに動的に切り替わる。essence メタデータから `著者` / `所属` を抽出し、
システムプロンプトの冒頭に注入する。

| 条件 | プロンプト例 |
|------|-------------|
| 著者+所属あり | 「あなたは 山田太郎(東京大学 所属)本人として、研究発表(LT)をおこなう。」 |
| 著者のみ | 「あなたは 山田太郎 本人として、研究発表(LT)をおこなう。」 |
| 未設定 (フォールバック) | 「あなたは D.Sugisawa(杉澤 大輔, mixi.co.jp 所属)本人として、研究発表(LT)をおこなう。」 |

`current_meta` には `theme` / `presenter` / `affiliation` / `venue` の 4 フィールドがあり、
論文選択 (`/api/upload/papers/{job_id}/select`) のたびに更新される。

ステージ別指示:

| ステージ | 指示 |
|---------|------|
| waiting | タイトル案内、「もう少ししたら開始」(1-2 文) |
| presenting | スライド内容を順序立てて説明 (3-6 文)、Knowledge Context 参照可 |
| qa | 質問意図を捉え直接応答 (3-5 文)、範囲外は率直に申告 |
| closing | 「ご清聴ありがとうございました」(1 文) |

出力ルール (TTS 前提):
- 自然な日本語話し言葉
- マークダウン記号・箇条書き・絵文字・コードブロック禁止
- 数式は言葉で説明
- 定着略語 (TCP, RTT 等) はそのまま、馴染みない記号は言い換え

知識ソース:
- `[Knowledge Context]` は RAG コーパスからの検索論文粋として参照
  - コーパスには自分の論文・メモだけでなく、過去にアップロードされた外部論文 (IPSJ 研究会等) も含まれる
  - アップロード時に `external-pdf-for-rag/` へ自動ミラーされ、次回インデックス再構築で取込まれる
- 論文粋にない数値・主張の捏造禁止
- `[Slide]` と矛盾時はスライド優先

### RAG クエリリライト (アップロード時)

gpt-5.2 で PDF 冒頭テキストから 6 クエリ生成:
- 3 観点 (methods / domain / evaluation) x 2 言語 (日本語 / 英語)
- 出力: `{summary, summary_en, queries_ja: [3], queries_en: [3]}`


<div style="page-break-before: always;"></div>

## フロントエンド設計

### レイアウト (3 カラム)

| カラム | 幅比 | 内容 |
|-------|------|------|
| 左 | 40% | 3D アバター (Three.js GLB モデル + 背景パターンアニメーション) |
| 中央 | 238% | スライドビューア (deck.json or iframe) + スピーカースクリプトペイン |
| 右 | 72% | 参加者サイドバー + Q&A タイムライン |

<div style="page-break-before: always;"></div>

### TTS パイプライン

<img src="./professor-tts.svg" style="max-height: 90vh; width: auto;" />

### アバターアニメーション

- Three.js AnimationMixer + GLB クリップ
- 2 プール: idle / speaking (フェード遷移 250ms)
- `finished` イベントで現在のプールから次クリップ選択

### 自動進行ロジック

<img src="./professor-auto-paging.svg" style="max-height: 90vh; width: auto;" />


<div style="page-break-before: always;"></div>

## 認証・アクセス制御

- **Google OAuth**: mixi.co.jp ドメイン制限 + `.custom-auth.txt` カスタム許可リスト
- **チケットシステム**: アップロード/削除はシングルユースチケット (ファイルリネームによるアトミック消費)
  - `.ticket.available` → `.ticket.consumed` (成功時)
  - 同時消費はリネーム競合で一方のみ成功 (レースセーフ)

## 状態管理

| データ | 保存先 | 永続性 |
|--------|--------|--------|
| 会話履歴 | `professor_data/sessions/{user_id}.json` + メモリ | 永続 (presenting のみ) |
| 参加者 | メモリ (`participants` dict) | プロセス内のみ |
| Q&A | メモリ (`_qa_id_counter` + participants[].qa) | プロセス内のみ |
| アップロードジョブ | メモリ + ディスク成果物 | 再起動時 rehydrate |
| 発表メタ | `current_meta` dict (theme/presenter/affiliation/venue) | プロセス内 (select 時更新) |
| ユーザー登録 | `professor_data/users.json` | 永続 |
| RAG インデックス | `professor_data/index/` (LanceDB + meta.json) | 永続 |
| チケット | `professor_data/tickets/` | ファイルベース永続 |


## ステートフル/ステートレス設計判断

| ステージ | 種別 | 理由 |
|---------|------|------|
| presenting | ステートフル (履歴永続化) | スライド間で重複説明を避けるため |
| waiting / qa / closing | ステートレス (エフェメラル) | 独立生成、長い回答が挨拶にエコーされるのを防止 |


<div style="page-break-before: always;"></div>

## 付録: 運用メモ

### データディレクトリ

```
professor_data/
├── sessions/{user_id}.json     # 会話履歴
├── index/
│   ├── lance/                  # LanceDB ベクトルストア
│   └── meta.json               # インデックスメタデータ
├── uploads/{job_id}/
│   ├── {filename}.pdf          # アップロード PDF
│   ├── presentation.html       # 生成スライド
│   ├── presentation.pdf        # 中間ファイル (スクリプト生成用)
│   ├── presentation_script.md  # スピーカースクリプト
│   ├── presentation.essence.txt
│   └── knowledge_context.md    # RAG コンテキスト
├── tickets/
│   ├── {uuid}.ticket.available
│   ├── {uuid}.ticket.remove
│   └── {uuid}.ticket.consumed
├── users.json                  # ユーザーレジストリ
└── deck.json                   # 静的スライドデッキ
```
