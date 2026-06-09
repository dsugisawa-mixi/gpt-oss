#!/usr/bin/env python3
"""facr_batch.py — 570 クエリの自動 ASK + FACR バッチ実行スクリプト.

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

# ── 120 クエリ定義 ────────────────────────────────────────────
QUERIES = [
    # ═══ Group A: 強α (70本) — RFC/WebRTCコーパスとほぼ無関係 ═══
    # ── AI・機械学習 (10) ──
    {"group": "A", "idx": 1,  "text": "Transformerの自己注意機構とは何か"},
    {"group": "A", "idx": 2,  "text": "Mixture of Expertsの利点は何か"},
    {"group": "A", "idx": 3,  "text": "LoRAによる微調整とは何か"},
    {"group": "A", "idx": 4,  "text": "RLHFの課題は何か"},
    {"group": "A", "idx": 5,  "text": "EmbeddingとTokenizationの違い"},
    {"group": "A", "idx": 6,  "text": "RAGとFine-Tuningの比較"},
    {"group": "A", "idx": 7,  "text": "Diffusion Modelの原理"},
    {"group": "A", "idx": 8,  "text": "LLM Hallucinationの原因"},
    {"group": "A", "idx": 9,  "text": "Knowledge Distillationとは何か"},
    {"group": "A", "idx": 10, "text": "Agentic AIの課題"},
    # ── データベース (10) ──
    {"group": "A", "idx": 11, "text": "PostgreSQLのMVCCとは何か"},
    {"group": "A", "idx": 12, "text": "B+Treeの利点"},
    {"group": "A", "idx": 13, "text": "Redis Clusterの設計"},
    {"group": "A", "idx": 14, "text": "BigQueryの列指向アーキテクチャ"},
    {"group": "A", "idx": 15, "text": "Graph Databaseの用途"},
    {"group": "A", "idx": 16, "text": "CAP定理とは何か"},
    {"group": "A", "idx": 17, "text": "OLTPとOLAPの違い"},
    {"group": "A", "idx": 18, "text": "Snowflakeアーキテクチャ"},
    {"group": "A", "idx": 19, "text": "Vector DBの検索手法"},
    {"group": "A", "idx": 20, "text": "ACID特性とは何か"},
    # ── モバイルアプリ (10) ──
    {"group": "A", "idx": 21, "text": "iOS App Sandboxとは何か"},
    {"group": "A", "idx": 22, "text": "Android Binderの役割"},
    {"group": "A", "idx": 23, "text": "SwiftUIのState管理"},
    {"group": "A", "idx": 24, "text": "Jetpack Composeの特徴"},
    {"group": "A", "idx": 25, "text": "App Store審査の課題"},
    {"group": "A", "idx": 26, "text": "Push通知の実装方式"},
    {"group": "A", "idx": 27, "text": "バッテリー最適化手法"},
    {"group": "A", "idx": 28, "text": "オフライン同期設計"},
    {"group": "A", "idx": 29, "text": "モバイルA/Bテスト"},
    {"group": "A", "idx": 30, "text": "In-App Purchase設計"},
    # ── ゲーム開発 (10) ──
    {"group": "A", "idx": 31, "text": "ECSアーキテクチャとは何か"},
    {"group": "A", "idx": 32, "text": "Unity DOTSの利点"},
    {"group": "A", "idx": 33, "text": "Unreal Engine Naniteとは何か"},
    {"group": "A", "idx": 34, "text": "ゲームサーバ同期方式"},
    {"group": "A", "idx": 35, "text": "ラグ補償手法"},
    {"group": "A", "idx": 36, "text": "MMOスケーリング手法"},
    {"group": "A", "idx": 37, "text": "アセットバンドル設計"},
    {"group": "A", "idx": 38, "text": "シェーダーパイプライン"},
    {"group": "A", "idx": 39, "text": "レイトレーシングの課題"},
    {"group": "A", "idx": 40, "text": "ゲームチート対策"},
    # ── 金融 (10) ──
    {"group": "A", "idx": 41, "text": "金融市場における流動性とは何か"},
    {"group": "A", "idx": 42, "text": "ETFの仕組み"},
    {"group": "A", "idx": 43, "text": "金利上昇の影響"},
    {"group": "A", "idx": 44, "text": "オプション取引とは何か"},
    {"group": "A", "idx": 45, "text": "VaRとは何か"},
    {"group": "A", "idx": 46, "text": "Basel IIIの目的"},
    {"group": "A", "idx": 47, "text": "ステーブルコインの課題"},
    {"group": "A", "idx": 48, "text": "CBDCとは何か"},
    {"group": "A", "idx": 49, "text": "高頻度取引の利点と問題"},
    {"group": "A", "idx": 50, "text": "信用リスク評価手法"},
    # ── 医療 (10) ──
    {"group": "A", "idx": 51, "text": "MRIの原理"},
    {"group": "A", "idx": 52, "text": "CTスキャンとの違い"},
    {"group": "A", "idx": 53, "text": "電子カルテの標準規格"},
    {"group": "A", "idx": 54, "text": "遠隔医療の課題"},
    {"group": "A", "idx": 55, "text": "医療AIの倫理問題"},
    {"group": "A", "idx": 56, "text": "mRNAワクチンの仕組み"},
    {"group": "A", "idx": 57, "text": "医療画像診断支援"},
    {"group": "A", "idx": 58, "text": "医療データ匿名化"},
    {"group": "A", "idx": 59, "text": "FHIR標準とは何か"},
    {"group": "A", "idx": 60, "text": "臨床試験フェーズの違い"},
    # ── 法律 (10) ──
    {"group": "A", "idx": 61, "text": "GDPRの概要"},
    {"group": "A", "idx": 62, "text": "著作権と特許の違い"},
    {"group": "A", "idx": 63, "text": "個人情報保護法の要件"},
    {"group": "A", "idx": 64, "text": "OSSライセンスの種類"},
    {"group": "A", "idx": 65, "text": "AI規制法案の課題"},
    {"group": "A", "idx": 66, "text": "契約不履行とは何か"},
    {"group": "A", "idx": 67, "text": "独占禁止法の目的"},
    {"group": "A", "idx": 68, "text": "電子署名法の要件"},
    {"group": "A", "idx": 69, "text": "越境データ移転の課題"},
    {"group": "A", "idx": 70, "text": "プライバシー権とは何か"},
    # ── 料理 (50) ──
    {"group": "A", "idx": 71,  "text": "料理におけるメイラード反応とは何か"},
    {"group": "A", "idx": 72,  "text": "低温調理の利点は何か"},
    {"group": "A", "idx": 73,  "text": "真空調理法の特徴は何か"},
    {"group": "A", "idx": 74,  "text": "乳化とは何か"},
    {"group": "A", "idx": 75,  "text": "ベシャメルソースの作り方"},
    {"group": "A", "idx": 76,  "text": "出汁の旨味成分とは何か"},
    {"group": "A", "idx": 77,  "text": "昆布出汁と鰹出汁の違い"},
    {"group": "A", "idx": 78,  "text": "発酵食品の代表例"},
    {"group": "A", "idx": 79,  "text": "パンが膨らむ仕組み"},
    {"group": "A", "idx": 80,  "text": "サワードウとは何か"},
    {"group": "A", "idx": 81,  "text": "酵母発酵と乳酸発酵の違い"},
    {"group": "A", "idx": 82,  "text": "味噌の製造工程"},
    {"group": "A", "idx": 83,  "text": "醤油の発酵プロセス"},
    {"group": "A", "idx": 84,  "text": "チーズ熟成の仕組み"},
    {"group": "A", "idx": 85,  "text": "オリーブオイルの分類"},
    {"group": "A", "idx": 86,  "text": "エクストラバージンオイルとは何か"},
    {"group": "A", "idx": 87,  "text": "フランス料理の五大ソースとは"},
    {"group": "A", "idx": 88,  "text": "ミシュラン評価基準とは"},
    {"group": "A", "idx": 89,  "text": "パスタアルデンテの意味"},
    {"group": "A", "idx": 90,  "text": "リゾットの特徴"},
    {"group": "A", "idx": 91,  "text": "スパイスとハーブの違い"},
    {"group": "A", "idx": 92,  "text": "タンドール調理法とは何か"},
    {"group": "A", "idx": 93,  "text": "寿司の歴史"},
    {"group": "A", "idx": 94,  "text": "天ぷらの起源"},
    {"group": "A", "idx": 95,  "text": "ラーメンスープの種類"},
    {"group": "A", "idx": 96,  "text": "中華鍋の特徴"},
    {"group": "A", "idx": 97,  "text": "北京ダックとは何か"},
    {"group": "A", "idx": 98,  "text": "韓国キムチの発酵"},
    {"group": "A", "idx": 99,  "text": "ナポリピザの定義"},
    {"group": "A", "idx": 100, "text": "フードペアリング理論とは何か"},
    {"group": "A", "idx": 101, "text": "食品保存技術の歴史"},
    {"group": "A", "idx": 102, "text": "冷凍保存の原理"},
    {"group": "A", "idx": 103, "text": "急速冷凍の利点"},
    {"group": "A", "idx": 104, "text": "燻製の種類"},
    {"group": "A", "idx": 105, "text": "食品乾燥技術"},
    {"group": "A", "idx": 106, "text": "砂糖の役割"},
    {"group": "A", "idx": 107, "text": "塩漬け保存法"},
    {"group": "A", "idx": 108, "text": "食品衛生管理HACCPとは"},
    {"group": "A", "idx": 109, "text": "食中毒の代表例"},
    {"group": "A", "idx": 110, "text": "ボツリヌス菌の危険性"},
    {"group": "A", "idx": 111, "text": "アレルギー表示制度"},
    {"group": "A", "idx": 112, "text": "食品添加物の目的"},
    {"group": "A", "idx": 113, "text": "代替肉とは何か"},
    {"group": "A", "idx": 114, "text": "培養肉の課題"},
    {"group": "A", "idx": 115, "text": "グルテンとは何か"},
    {"group": "A", "idx": 116, "text": "ビーガン料理の特徴"},
    {"group": "A", "idx": 117, "text": "地中海食の特徴"},
    {"group": "A", "idx": 118, "text": "和食が無形文化遺産になった理由"},
    {"group": "A", "idx": 119, "text": "分子ガストロノミーとは何か"},
    {"group": "A", "idx": 120, "text": "料理科学とは何か"},
    # ── 歴史 (50) ──
    {"group": "A", "idx": 121, "text": "ローマ帝国滅亡の要因"},
    {"group": "A", "idx": 122, "text": "ビザンツ帝国とは何か"},
    {"group": "A", "idx": 123, "text": "十字軍遠征の目的"},
    {"group": "A", "idx": 124, "text": "百年戦争の原因"},
    {"group": "A", "idx": 125, "text": "薔薇戦争とは何か"},
    {"group": "A", "idx": 126, "text": "ルネサンスの特徴"},
    {"group": "A", "idx": 127, "text": "宗教改革の背景"},
    {"group": "A", "idx": 128, "text": "三十年戦争の影響"},
    {"group": "A", "idx": 129, "text": "産業革命の意義"},
    {"group": "A", "idx": 130, "text": "ナポレオン戦争とは何か"},
    {"group": "A", "idx": 131, "text": "ウィーン会議の目的"},
    {"group": "A", "idx": 132, "text": "アヘン戦争の原因"},
    {"group": "A", "idx": 133, "text": "明治維新の意義"},
    {"group": "A", "idx": 134, "text": "第一次世界大戦の発端"},
    {"group": "A", "idx": 135, "text": "ベルサイユ条約の内容"},
    {"group": "A", "idx": 136, "text": "第二次世界大戦の要因"},
    {"group": "A", "idx": 137, "text": "冷戦構造とは何か"},
    {"group": "A", "idx": 138, "text": "キューバ危機とは何か"},
    {"group": "A", "idx": 139, "text": "ソ連崩壊の要因"},
    {"group": "A", "idx": 140, "text": "EU成立の経緯"},
    {"group": "A", "idx": 141, "text": "秦の始皇帝とは何者か"},
    {"group": "A", "idx": 142, "text": "漢帝国の特徴"},
    {"group": "A", "idx": 143, "text": "三国時代とは何か"},
    {"group": "A", "idx": 144, "text": "唐王朝の繁栄理由"},
    {"group": "A", "idx": 145, "text": "モンゴル帝国の拡大"},
    {"group": "A", "idx": 146, "text": "大航海時代とは何か"},
    {"group": "A", "idx": 147, "text": "コロンブス航海の影響"},
    {"group": "A", "idx": 148, "text": "アステカ帝国とは何か"},
    {"group": "A", "idx": 149, "text": "インカ帝国とは何か"},
    {"group": "A", "idx": 150, "text": "フランス革命の原因"},
    {"group": "A", "idx": 151, "text": "アメリカ独立戦争の背景"},
    {"group": "A", "idx": 152, "text": "南北戦争とは何か"},
    {"group": "A", "idx": 153, "text": "帝国主義とは何か"},
    {"group": "A", "idx": 154, "text": "日露戦争の意義"},
    {"group": "A", "idx": 155, "text": "関東大震災の影響"},
    {"group": "A", "idx": 156, "text": "世界恐慌の原因"},
    {"group": "A", "idx": 157, "text": "ニューディール政策とは何か"},
    {"group": "A", "idx": 158, "text": "朝鮮戦争とは何か"},
    {"group": "A", "idx": 159, "text": "ベトナム戦争とは何か"},
    {"group": "A", "idx": 160, "text": "文化大革命とは何か"},
    {"group": "A", "idx": 161, "text": "湾岸戦争とは何か"},
    {"group": "A", "idx": 162, "text": "アラブの春とは何か"},
    {"group": "A", "idx": 163, "text": "縄文時代の特徴"},
    {"group": "A", "idx": 164, "text": "弥生時代の特徴"},
    {"group": "A", "idx": 165, "text": "平安文化とは何か"},
    {"group": "A", "idx": 166, "text": "鎌倉幕府成立の経緯"},
    {"group": "A", "idx": 167, "text": "戦国時代とは何か"},
    {"group": "A", "idx": 168, "text": "江戸幕府の統治制度"},
    {"group": "A", "idx": 169, "text": "明治憲法の特徴"},
    {"group": "A", "idx": 170, "text": "日本国憲法制定経緯"},
    # ── 生物学 (50) ──
    {"group": "A", "idx": 171, "text": "DNAの構造とは何か"},
    {"group": "A", "idx": 172, "text": "RNAの役割"},
    {"group": "A", "idx": 173, "text": "転写と翻訳の違い"},
    {"group": "A", "idx": 174, "text": "細胞分裂の仕組み"},
    {"group": "A", "idx": 175, "text": "有糸分裂とは何か"},
    {"group": "A", "idx": 176, "text": "減数分裂とは何か"},
    {"group": "A", "idx": 177, "text": "遺伝子とは何か"},
    {"group": "A", "idx": 178, "text": "染色体とは何か"},
    {"group": "A", "idx": 179, "text": "エピジェネティクスとは何か"},
    {"group": "A", "idx": 180, "text": "CRISPR-Cas9とは何か"},
    {"group": "A", "idx": 181, "text": "自然選択説とは何か"},
    {"group": "A", "idx": 182, "text": "進化論の概要"},
    {"group": "A", "idx": 183, "text": "ミトコンドリアの役割"},
    {"group": "A", "idx": 184, "text": "葉緑体の役割"},
    {"group": "A", "idx": 185, "text": "光合成の仕組み"},
    {"group": "A", "idx": 186, "text": "呼吸代謝とは何か"},
    {"group": "A", "idx": 187, "text": "ATPとは何か"},
    {"group": "A", "idx": 188, "text": "タンパク質合成とは何か"},
    {"group": "A", "idx": 189, "text": "酵素の働き"},
    {"group": "A", "idx": 190, "text": "免疫系の仕組み"},
    {"group": "A", "idx": 191, "text": "抗体とは何か"},
    {"group": "A", "idx": 192, "text": "ワクチンの原理"},
    {"group": "A", "idx": 193, "text": "ウイルスと細菌の違い"},
    {"group": "A", "idx": 194, "text": "細菌叢とは何か"},
    {"group": "A", "idx": 195, "text": "腸内細菌の役割"},
    {"group": "A", "idx": 196, "text": "神経伝達物質とは何か"},
    {"group": "A", "idx": 197, "text": "ホルモンとは何か"},
    {"group": "A", "idx": 198, "text": "内分泌系の役割"},
    {"group": "A", "idx": 199, "text": "植物ホルモンとは何か"},
    {"group": "A", "idx": 200, "text": "生態系とは何か"},
    {"group": "A", "idx": 201, "text": "食物連鎖とは何か"},
    {"group": "A", "idx": 202, "text": "生物多様性の意義"},
    {"group": "A", "idx": 203, "text": "絶滅危惧種とは何か"},
    {"group": "A", "idx": 204, "text": "適応進化とは何か"},
    {"group": "A", "idx": 205, "text": "共進化とは何か"},
    {"group": "A", "idx": 206, "text": "寄生生物とは何か"},
    {"group": "A", "idx": 207, "text": "共生関係とは何か"},
    {"group": "A", "idx": 208, "text": "幹細胞とは何か"},
    {"group": "A", "idx": 209, "text": "ES細胞とは何か"},
    {"group": "A", "idx": 210, "text": "iPS細胞とは何か"},
    {"group": "A", "idx": 211, "text": "ゲノム編集の課題"},
    {"group": "A", "idx": 212, "text": "老化研究の現状"},
    {"group": "A", "idx": 213, "text": "アルツハイマー病とは何か"},
    {"group": "A", "idx": 214, "text": "癌発生のメカニズム"},
    {"group": "A", "idx": 215, "text": "遺伝性疾患とは何か"},
    {"group": "A", "idx": 216, "text": "血液型の遺伝"},
    {"group": "A", "idx": 217, "text": "ヒトゲノム計画とは何か"},
    {"group": "A", "idx": 218, "text": "マイクロRNAとは何か"},
    {"group": "A", "idx": 219, "text": "オートファジーとは何か"},
    {"group": "A", "idx": 220, "text": "生物学的進化の証拠"},
    # ── 天文学 (50) ──
    {"group": "A", "idx": 221, "text": "太陽系形成理論とは何か"},
    {"group": "A", "idx": 222, "text": "原始惑星系円盤とは何か"},
    {"group": "A", "idx": 223, "text": "恒星の一生を説明せよ"},
    {"group": "A", "idx": 224, "text": "赤色巨星とは何か"},
    {"group": "A", "idx": 225, "text": "白色矮星とは何か"},
    {"group": "A", "idx": 226, "text": "中性子星とは何か"},
    {"group": "A", "idx": 227, "text": "ブラックホールとは何か"},
    {"group": "A", "idx": 228, "text": "事象の地平面とは何か"},
    {"group": "A", "idx": 229, "text": "一般相対性理論とは何か"},
    {"group": "A", "idx": 230, "text": "重力波とは何か"},
    {"group": "A", "idx": 231, "text": "銀河とは何か"},
    {"group": "A", "idx": 232, "text": "天の川銀河の特徴"},
    {"group": "A", "idx": 233, "text": "アンドロメダ銀河とは何か"},
    {"group": "A", "idx": 234, "text": "ダークマターとは何か"},
    {"group": "A", "idx": 235, "text": "ダークエネルギーとは何か"},
    {"group": "A", "idx": 236, "text": "ビッグバン理論とは何か"},
    {"group": "A", "idx": 237, "text": "宇宙背景放射とは何か"},
    {"group": "A", "idx": 238, "text": "ハッブル定数とは何か"},
    {"group": "A", "idx": 239, "text": "宇宙膨張とは何か"},
    {"group": "A", "idx": 240, "text": "赤方偏移とは何か"},
    {"group": "A", "idx": 241, "text": "系外惑星の探査方法"},
    {"group": "A", "idx": 242, "text": "トランジット法とは何か"},
    {"group": "A", "idx": 243, "text": "ドップラー法とは何か"},
    {"group": "A", "idx": 244, "text": "地球型惑星とは何か"},
    {"group": "A", "idx": 245, "text": "ハビタブルゾーンとは何か"},
    {"group": "A", "idx": 246, "text": "火星探査の目的"},
    {"group": "A", "idx": 247, "text": "木星の特徴"},
    {"group": "A", "idx": 248, "text": "土星の環はなぜ存在するか"},
    {"group": "A", "idx": 249, "text": "天王星の特徴"},
    {"group": "A", "idx": 250, "text": "海王星の特徴"},
    {"group": "A", "idx": 251, "text": "冥王星が準惑星になった理由"},
    {"group": "A", "idx": 252, "text": "小惑星帯とは何か"},
    {"group": "A", "idx": 253, "text": "彗星とは何か"},
    {"group": "A", "idx": 254, "text": "流星と隕石の違い"},
    {"group": "A", "idx": 255, "text": "オールトの雲とは何か"},
    {"group": "A", "idx": 256, "text": "ケプラーの法則とは何か"},
    {"group": "A", "idx": 257, "text": "ニュートンの万有引力とは何か"},
    {"group": "A", "idx": 258, "text": "ラグランジュ点とは何か"},
    {"group": "A", "idx": 259, "text": "ジェームズウェッブ宇宙望遠鏡の特徴"},
    {"group": "A", "idx": 260, "text": "ハッブル宇宙望遠鏡の成果"},
    {"group": "A", "idx": 261, "text": "電波天文学とは何か"},
    {"group": "A", "idx": 262, "text": "X線天文学とは何か"},
    {"group": "A", "idx": 263, "text": "パルサーとは何か"},
    {"group": "A", "idx": 264, "text": "クエーサーとは何か"},
    {"group": "A", "idx": 265, "text": "超新星爆発とは何か"},
    {"group": "A", "idx": 266, "text": "Ia型超新星とは何か"},
    {"group": "A", "idx": 267, "text": "宇宙論標準モデルとは何か"},
    {"group": "A", "idx": 268, "text": "フェルミのパラドックスとは何か"},
    {"group": "A", "idx": 269, "text": "SETIとは何か"},
    {"group": "A", "idx": 270, "text": "宇宙エレベーターの課題"},
    # ── 経済学 (50) ──
    {"group": "A", "idx": 271, "text": "需要と供給の法則とは何か"},
    {"group": "A", "idx": 272, "text": "価格弾力性とは何か"},
    {"group": "A", "idx": 273, "text": "限界効用とは何か"},
    {"group": "A", "idx": 274, "text": "機会費用とは何か"},
    {"group": "A", "idx": 275, "text": "比較優位とは何か"},
    {"group": "A", "idx": 276, "text": "GDPとは何か"},
    {"group": "A", "idx": 277, "text": "GNPとは何か"},
    {"group": "A", "idx": 278, "text": "インフレーションとは何か"},
    {"group": "A", "idx": 279, "text": "デフレーションとは何か"},
    {"group": "A", "idx": 280, "text": "中央銀行の役割"},
    {"group": "A", "idx": 281, "text": "金融政策とは何か"},
    {"group": "A", "idx": 282, "text": "財政政策とは何か"},
    {"group": "A", "idx": 283, "text": "量的緩和とは何か"},
    {"group": "A", "idx": 284, "text": "金利政策とは何か"},
    {"group": "A", "idx": 285, "text": "マクロ経済学とは何か"},
    {"group": "A", "idx": 286, "text": "ミクロ経済学とは何か"},
    {"group": "A", "idx": 287, "text": "市場の失敗とは何か"},
    {"group": "A", "idx": 288, "text": "外部性とは何か"},
    {"group": "A", "idx": 289, "text": "公共財とは何か"},
    {"group": "A", "idx": 290, "text": "独占市場とは何か"},
    {"group": "A", "idx": 291, "text": "寡占市場とは何か"},
    {"group": "A", "idx": 292, "text": "ゲーム理論とは何か"},
    {"group": "A", "idx": 293, "text": "ナッシュ均衡とは何か"},
    {"group": "A", "idx": 294, "text": "情報の非対称性とは何か"},
    {"group": "A", "idx": 295, "text": "モラルハザードとは何か"},
    {"group": "A", "idx": 296, "text": "逆選択とは何か"},
    {"group": "A", "idx": 297, "text": "労働市場とは何か"},
    {"group": "A", "idx": 298, "text": "失業率の意味"},
    {"group": "A", "idx": 299, "text": "フィリップス曲線とは何か"},
    {"group": "A", "idx": 300, "text": "IS-LMモデルとは何か"},
    {"group": "A", "idx": 301, "text": "為替レートとは何か"},
    {"group": "A", "idx": 302, "text": "固定相場制とは何か"},
    {"group": "A", "idx": 303, "text": "変動相場制とは何か"},
    {"group": "A", "idx": 304, "text": "国際収支とは何か"},
    {"group": "A", "idx": 305, "text": "貿易黒字とは何か"},
    {"group": "A", "idx": 306, "text": "経常収支とは何か"},
    {"group": "A", "idx": 307, "text": "自由貿易の利点"},
    {"group": "A", "idx": 308, "text": "保護貿易の利点"},
    {"group": "A", "idx": 309, "text": "関税の効果"},
    {"group": "A", "idx": 310, "text": "世界銀行の役割"},
    {"group": "A", "idx": 311, "text": "IMFの役割"},
    {"group": "A", "idx": 312, "text": "BRICSとは何か"},
    {"group": "A", "idx": 313, "text": "スタグフレーションとは何か"},
    {"group": "A", "idx": 314, "text": "行動経済学とは何か"},
    {"group": "A", "idx": 315, "text": "プロスペクト理論とは何か"},
    {"group": "A", "idx": 316, "text": "所得格差の要因"},
    {"group": "A", "idx": 317, "text": "ジニ係数とは何か"},
    {"group": "A", "idx": 318, "text": "ベーシックインカムとは何か"},
    {"group": "A", "idx": 319, "text": "暗号資産の経済的課題"},
    {"group": "A", "idx": 320, "text": "サステナブル経済とは何か"},
    # ── 自動車工学 (50) ──
    {"group": "A", "idx": 321, "text": "4ストロークエンジンとは何か"},
    {"group": "A", "idx": 322, "text": "2ストロークエンジンとは何か"},
    {"group": "A", "idx": 323, "text": "ディーゼルエンジンの特徴"},
    {"group": "A", "idx": 324, "text": "ガソリンエンジンの特徴"},
    {"group": "A", "idx": 325, "text": "圧縮比とは何か"},
    {"group": "A", "idx": 326, "text": "ターボチャージャーとは何か"},
    {"group": "A", "idx": 327, "text": "スーパーチャージャーとは何か"},
    {"group": "A", "idx": 328, "text": "ハイブリッド車の仕組み"},
    {"group": "A", "idx": 329, "text": "EVの構造"},
    {"group": "A", "idx": 330, "text": "燃料電池車とは何か"},
    {"group": "A", "idx": 331, "text": "回生ブレーキとは何か"},
    {"group": "A", "idx": 332, "text": "差動装置とは何か"},
    {"group": "A", "idx": 333, "text": "トランスミッションの役割"},
    {"group": "A", "idx": 334, "text": "CVTとは何か"},
    {"group": "A", "idx": 335, "text": "ATとMTの違い"},
    {"group": "A", "idx": 336, "text": "サスペンションの役割"},
    {"group": "A", "idx": 337, "text": "ダブルウィッシュボーンとは何か"},
    {"group": "A", "idx": 338, "text": "マクファーソンストラットとは何か"},
    {"group": "A", "idx": 339, "text": "ABSとは何か"},
    {"group": "A", "idx": 340, "text": "ESCとは何か"},
    {"group": "A", "idx": 341, "text": "トラクションコントロールとは何か"},
    {"group": "A", "idx": 342, "text": "自動運転レベル分類"},
    {"group": "A", "idx": 343, "text": "LiDARとは何か"},
    {"group": "A", "idx": 344, "text": "ミリ波レーダーとは何か"},
    {"group": "A", "idx": 345, "text": "ADASとは何か"},
    {"group": "A", "idx": 346, "text": "車両運動制御とは何か"},
    {"group": "A", "idx": 347, "text": "アンダーステアとは何か"},
    {"group": "A", "idx": 348, "text": "オーバーステアとは何か"},
    {"group": "A", "idx": 349, "text": "空力特性とは何か"},
    {"group": "A", "idx": 350, "text": "ダウンフォースとは何か"},
    {"group": "A", "idx": 351, "text": "Cd値とは何か"},
    {"group": "A", "idx": 352, "text": "EVバッテリーの種類"},
    {"group": "A", "idx": 353, "text": "リチウムイオン電池とは何か"},
    {"group": "A", "idx": 354, "text": "全固体電池とは何か"},
    {"group": "A", "idx": 355, "text": "急速充電の課題"},
    {"group": "A", "idx": 356, "text": "V2Gとは何か"},
    {"group": "A", "idx": 357, "text": "水素エネルギーの課題"},
    {"group": "A", "idx": 358, "text": "排出ガス規制とは何か"},
    {"group": "A", "idx": 359, "text": "ユーロ7規制とは何か"},
    {"group": "A", "idx": 360, "text": "カーボンニュートラルとは何か"},
    {"group": "A", "idx": 361, "text": "エンジンノッキングとは何か"},
    {"group": "A", "idx": 362, "text": "点火時期とは何か"},
    {"group": "A", "idx": 363, "text": "燃焼効率とは何か"},
    {"group": "A", "idx": 364, "text": "プラグインハイブリッドとは何か"},
    {"group": "A", "idx": 365, "text": "車載ネットワークCANとは何か"},
    {"group": "A", "idx": 366, "text": "FlexRayとは何か"},
    {"group": "A", "idx": 367, "text": "Automotive Ethernetとは何か"},
    {"group": "A", "idx": 368, "text": "OTAアップデートとは何か"},
    {"group": "A", "idx": 369, "text": "自動車機能安全ISO26262とは何か"},
    {"group": "A", "idx": 370, "text": "SDVとは何か"},
    # ── 建築 (50) ──
    {"group": "A", "idx": 371, "text": "建築基準法とは何か"},
    {"group": "A", "idx": 372, "text": "耐震設計とは何か"},
    {"group": "A", "idx": 373, "text": "免震構造とは何か"},
    {"group": "A", "idx": 374, "text": "制振構造とは何か"},
    {"group": "A", "idx": 375, "text": "RC造とは何か"},
    {"group": "A", "idx": 376, "text": "SRC造とは何か"},
    {"group": "A", "idx": 377, "text": "S造とは何か"},
    {"group": "A", "idx": 378, "text": "木造建築の特徴"},
    {"group": "A", "idx": 379, "text": "ツーバイフォー工法とは何か"},
    {"group": "A", "idx": 380, "text": "ラーメン構造とは何か"},
    {"group": "A", "idx": 381, "text": "トラス構造とは何か"},
    {"group": "A", "idx": 382, "text": "シェル構造とは何か"},
    {"group": "A", "idx": 383, "text": "超高層建築の課題"},
    {"group": "A", "idx": 384, "text": "基礎工事の種類"},
    {"group": "A", "idx": 385, "text": "杭基礎とは何か"},
    {"group": "A", "idx": 386, "text": "地盤改良工法とは何か"},
    {"group": "A", "idx": 387, "text": "建築確認申請とは何か"},
    {"group": "A", "idx": 388, "text": "BIMとは何か"},
    {"group": "A", "idx": 389, "text": "LEED認証とは何か"},
    {"group": "A", "idx": 390, "text": "ZEBとは何か"},
    {"group": "A", "idx": 391, "text": "パッシブデザインとは何か"},
    {"group": "A", "idx": 392, "text": "断熱性能とは何か"},
    {"group": "A", "idx": 393, "text": "気密性能とは何か"},
    {"group": "A", "idx": 394, "text": "換気システムの役割"},
    {"group": "A", "idx": 395, "text": "ヒートアイランド現象とは何か"},
    {"group": "A", "idx": 396, "text": "都市計画とは何か"},
    {"group": "A", "idx": 397, "text": "用途地域とは何か"},
    {"group": "A", "idx": 398, "text": "容積率とは何か"},
    {"group": "A", "idx": 399, "text": "建ぺい率とは何か"},
    {"group": "A", "idx": 400, "text": "ランドマーク建築とは何か"},
    {"group": "A", "idx": 401, "text": "ゴシック建築とは何か"},
    {"group": "A", "idx": 402, "text": "ロマネスク建築とは何か"},
    {"group": "A", "idx": 403, "text": "バロック建築とは何か"},
    {"group": "A", "idx": 404, "text": "近代建築とは何か"},
    {"group": "A", "idx": 405, "text": "モダニズム建築とは何か"},
    {"group": "A", "idx": 406, "text": "ポストモダン建築とは何か"},
    {"group": "A", "idx": 407, "text": "ル・コルビュジエとは何者か"},
    {"group": "A", "idx": 408, "text": "フランク・ロイド・ライトとは何者か"},
    {"group": "A", "idx": 409, "text": "丹下健三とは何者か"},
    {"group": "A", "idx": 410, "text": "隈研吾とは何者か"},
    {"group": "A", "idx": 411, "text": "木材利用促進の利点"},
    {"group": "A", "idx": 412, "text": "建築音響設計とは何か"},
    {"group": "A", "idx": 413, "text": "防火設計とは何か"},
    {"group": "A", "idx": 414, "text": "避難計画とは何か"},
    {"group": "A", "idx": 415, "text": "スマートビルとは何か"},
    {"group": "A", "idx": 416, "text": "建築設備とは何か"},
    {"group": "A", "idx": 417, "text": "空調システムの種類"},
    {"group": "A", "idx": 418, "text": "BEMSとは何か"},
    {"group": "A", "idx": 419, "text": "サステナブル建築とは何か"},
    {"group": "A", "idx": 420, "text": "都市再開発とは何か"},
    # ── スポーツ科学 (50) ──
    {"group": "A", "idx": 421, "text": "VO2maxとは何か"},
    {"group": "A", "idx": 422, "text": "有酸素運動とは何か"},
    {"group": "A", "idx": 423, "text": "無酸素運動とは何か"},
    {"group": "A", "idx": 424, "text": "乳酸閾値とは何か"},
    {"group": "A", "idx": 425, "text": "筋肥大の仕組み"},
    {"group": "A", "idx": 426, "text": "速筋と遅筋の違い"},
    {"group": "A", "idx": 427, "text": "超回復とは何か"},
    {"group": "A", "idx": 428, "text": "筋力トレーニングの原理"},
    {"group": "A", "idx": 429, "text": "ピリオダイゼーションとは何か"},
    {"group": "A", "idx": 430, "text": "スポーツ栄養学とは何か"},
    {"group": "A", "idx": 431, "text": "グリコーゲンローディングとは何か"},
    {"group": "A", "idx": 432, "text": "脱水症状の影響"},
    {"group": "A", "idx": 433, "text": "電解質の役割"},
    {"group": "A", "idx": 434, "text": "スポーツ心理学とは何か"},
    {"group": "A", "idx": 435, "text": "モチベーション理論とは何か"},
    {"group": "A", "idx": 436, "text": "集中力向上手法"},
    {"group": "A", "idx": 437, "text": "反応時間とは何か"},
    {"group": "A", "idx": 438, "text": "アジリティとは何か"},
    {"group": "A", "idx": 439, "text": "バイオメカニクスとは何か"},
    {"group": "A", "idx": 440, "text": "運動連鎖とは何か"},
    {"group": "A", "idx": 441, "text": "投球動作解析とは何か"},
    {"group": "A", "idx": 442, "text": "ランニングフォーム分析"},
    {"group": "A", "idx": 443, "text": "スポーツ障害とは何か"},
    {"group": "A", "idx": 444, "text": "ACL損傷とは何か"},
    {"group": "A", "idx": 445, "text": "肉離れとは何か"},
    {"group": "A", "idx": 446, "text": "オーバートレーニング症候群とは何か"},
    {"group": "A", "idx": 447, "text": "スポーツドーピングとは何か"},
    {"group": "A", "idx": 448, "text": "WADAの役割"},
    {"group": "A", "idx": 449, "text": "スポーツ医学とは何か"},
    {"group": "A", "idx": 450, "text": "リカバリー戦略とは何か"},
    {"group": "A", "idx": 451, "text": "アイシングの効果"},
    {"group": "A", "idx": 452, "text": "睡眠と競技力の関係"},
    {"group": "A", "idx": 453, "text": "持久力向上方法"},
    {"group": "A", "idx": 454, "text": "瞬発力向上方法"},
    {"group": "A", "idx": 455, "text": "ジャンプ力向上方法"},
    {"group": "A", "idx": 456, "text": "競泳の流体力学"},
    {"group": "A", "idx": 457, "text": "サッカー戦術分析とは何か"},
    {"group": "A", "idx": 458, "text": "野球セイバーメトリクスとは何か"},
    {"group": "A", "idx": 459, "text": "テニスサーブ速度要因"},
    {"group": "A", "idx": 460, "text": "ゴルフスイング解析"},
    {"group": "A", "idx": 461, "text": "ラグビー戦術の特徴"},
    {"group": "A", "idx": 462, "text": "バスケットボールのPACEとは何か"},
    {"group": "A", "idx": 463, "text": "スポーツデータ分析とは何か"},
    {"group": "A", "idx": 464, "text": "GPSトラッキング活用"},
    {"group": "A", "idx": 465, "text": "eSportsとスポーツ科学"},
    {"group": "A", "idx": 466, "text": "女性アスリート特有課題"},
    {"group": "A", "idx": 467, "text": "成長期トレーニングの注意点"},
    {"group": "A", "idx": 468, "text": "スポーツ傷害予防"},
    {"group": "A", "idx": 469, "text": "運動学習とは何か"},
    {"group": "A", "idx": 470, "text": "スポーツ科学の将来展望"},
    # ── 音楽理論 (50) ──
    {"group": "A", "idx": 471, "text": "音階とは何か"},
    {"group": "A", "idx": 472, "text": "長音階とは何か"},
    {"group": "A", "idx": 473, "text": "短音階とは何か"},
    {"group": "A", "idx": 474, "text": "和音とは何か"},
    {"group": "A", "idx": 475, "text": "三和音とは何か"},
    {"group": "A", "idx": 476, "text": "七の和音とは何か"},
    {"group": "A", "idx": 477, "text": "コード進行とは何か"},
    {"group": "A", "idx": 478, "text": "完全五度とは何か"},
    {"group": "A", "idx": 479, "text": "協和音とは何か"},
    {"group": "A", "idx": 480, "text": "不協和音とは何か"},
    {"group": "A", "idx": 481, "text": "転調とは何か"},
    {"group": "A", "idx": 482, "text": "対位法とは何か"},
    {"group": "A", "idx": 483, "text": "フーガとは何か"},
    {"group": "A", "idx": 484, "text": "ソナタ形式とは何か"},
    {"group": "A", "idx": 485, "text": "交響曲とは何か"},
    {"group": "A", "idx": 486, "text": "協奏曲とは何か"},
    {"group": "A", "idx": 487, "text": "オペラとは何か"},
    {"group": "A", "idx": 488, "text": "調性とは何か"},
    {"group": "A", "idx": 489, "text": "無調音楽とは何か"},
    {"group": "A", "idx": 490, "text": "十二音技法とは何か"},
    {"group": "A", "idx": 491, "text": "平均律とは何か"},
    {"group": "A", "idx": 492, "text": "純正律とは何か"},
    {"group": "A", "idx": 493, "text": "テンポとは何か"},
    {"group": "A", "idx": 494, "text": "拍子とは何か"},
    {"group": "A", "idx": 495, "text": "ポリリズムとは何か"},
    {"group": "A", "idx": 496, "text": "シンコペーションとは何か"},
    {"group": "A", "idx": 497, "text": "ブルース進行とは何か"},
    {"group": "A", "idx": 498, "text": "ジャズ理論とは何か"},
    {"group": "A", "idx": 499, "text": "モードとは何か"},
    {"group": "A", "idx": 500, "text": "ドリアンモードとは何か"},
    {"group": "A", "idx": 501, "text": "リディアンモードとは何か"},
    {"group": "A", "idx": 502, "text": "即興演奏とは何か"},
    {"group": "A", "idx": 503, "text": "耳コピとは何か"},
    {"group": "A", "idx": 504, "text": "絶対音感とは何か"},
    {"group": "A", "idx": 505, "text": "相対音感とは何か"},
    {"group": "A", "idx": 506, "text": "オーケストレーションとは何か"},
    {"group": "A", "idx": 507, "text": "弦楽器の特徴"},
    {"group": "A", "idx": 508, "text": "木管楽器の特徴"},
    {"group": "A", "idx": 509, "text": "金管楽器の特徴"},
    {"group": "A", "idx": 510, "text": "打楽器の特徴"},
    {"group": "A", "idx": 511, "text": "倍音とは何か"},
    {"group": "A", "idx": 512, "text": "音色とは何か"},
    {"group": "A", "idx": 513, "text": "音響学とは何か"},
    {"group": "A", "idx": 514, "text": "電子音楽とは何か"},
    {"group": "A", "idx": 515, "text": "シンセサイザーとは何か"},
    {"group": "A", "idx": 516, "text": "MIDIとは何か"},
    {"group": "A", "idx": 517, "text": "DAWとは何か"},
    {"group": "A", "idx": 518, "text": "サンプリングとは何か"},
    {"group": "A", "idx": 519, "text": "マスタリングとは何か"},
    {"group": "A", "idx": 520, "text": "音楽理論の歴史"},
    # ── 哲学 (50) ──
    {"group": "A", "idx": 521, "text": "ソクラテスとは何者か"},
    {"group": "A", "idx": 522, "text": "プラトンのイデア論とは何か"},
    {"group": "A", "idx": 523, "text": "アリストテレス哲学とは何か"},
    {"group": "A", "idx": 524, "text": "ストア派とは何か"},
    {"group": "A", "idx": 525, "text": "エピクロス派とは何か"},
    {"group": "A", "idx": 526, "text": "懐疑主義とは何か"},
    {"group": "A", "idx": 527, "text": "デカルトの方法的懐疑とは何か"},
    {"group": "A", "idx": 528, "text": "我思う故に我ありとは何か"},
    {"group": "A", "idx": 529, "text": "スピノザ哲学とは何か"},
    {"group": "A", "idx": 530, "text": "ライプニッツ哲学とは何か"},
    {"group": "A", "idx": 531, "text": "経験論とは何か"},
    {"group": "A", "idx": 532, "text": "合理論とは何か"},
    {"group": "A", "idx": 533, "text": "ヒューム哲学とは何か"},
    {"group": "A", "idx": 534, "text": "カント哲学とは何か"},
    {"group": "A", "idx": 535, "text": "定言命法とは何か"},
    {"group": "A", "idx": 536, "text": "ヘーゲル弁証法とは何か"},
    {"group": "A", "idx": 537, "text": "マルクス哲学とは何か"},
    {"group": "A", "idx": 538, "text": "実存主義とは何か"},
    {"group": "A", "idx": 539, "text": "キルケゴール哲学とは何か"},
    {"group": "A", "idx": 540, "text": "ニーチェ哲学とは何か"},
    {"group": "A", "idx": 541, "text": "超人思想とは何か"},
    {"group": "A", "idx": 542, "text": "現象学とは何か"},
    {"group": "A", "idx": 543, "text": "フッサール哲学とは何か"},
    {"group": "A", "idx": 544, "text": "ハイデガー哲学とは何か"},
    {"group": "A", "idx": 545, "text": "サルトル哲学とは何か"},
    {"group": "A", "idx": 546, "text": "自由意志とは何か"},
    {"group": "A", "idx": 547, "text": "決定論とは何か"},
    {"group": "A", "idx": 548, "text": "心身問題とは何か"},
    {"group": "A", "idx": 549, "text": "二元論とは何か"},
    {"group": "A", "idx": 550, "text": "唯物論とは何か"},
    {"group": "A", "idx": 551, "text": "功利主義とは何か"},
    {"group": "A", "idx": 552, "text": "義務論とは何か"},
    {"group": "A", "idx": 553, "text": "徳倫理学とは何か"},
    {"group": "A", "idx": 554, "text": "社会契約論とは何か"},
    {"group": "A", "idx": 555, "text": "ロールズの正義論とは何か"},
    {"group": "A", "idx": 556, "text": "ノージックの政治哲学とは何か"},
    {"group": "A", "idx": 557, "text": "言語哲学とは何か"},
    {"group": "A", "idx": 558, "text": "分析哲学とは何か"},
    {"group": "A", "idx": 559, "text": "論理実証主義とは何か"},
    {"group": "A", "idx": 560, "text": "科学哲学とは何か"},
    {"group": "A", "idx": 561, "text": "ポパーの反証可能性とは何か"},
    {"group": "A", "idx": 562, "text": "クーンのパラダイム論とは何か"},
    {"group": "A", "idx": 563, "text": "AIに意識は存在するか"},
    {"group": "A", "idx": 564, "text": "中国語の部屋とは何か"},
    {"group": "A", "idx": 565, "text": "テセウスの船とは何か"},
    {"group": "A", "idx": 566, "text": "トロッコ問題とは何か"},
    {"group": "A", "idx": 567, "text": "シミュレーション仮説とは何か"},
    {"group": "A", "idx": 568, "text": "存在論とは何か"},
    {"group": "A", "idx": 569, "text": "認識論とは何か"},
    {"group": "A", "idx": 570, "text": "哲学とは何を探究する学問か"},
    # ═══ Group B: 中α (30本) — ITだが別分野 ═══
    # ── Cloud (10) ──
    {"group": "B", "idx": 1,  "text": "Kubernetes Operatorとは何か"},
    {"group": "B", "idx": 2,  "text": "Service Meshの利点"},
    {"group": "B", "idx": 3,  "text": "Istioの課題"},
    {"group": "B", "idx": 4,  "text": "Serverlessの適用範囲"},
    {"group": "B", "idx": 5,  "text": "Cloud RunとGKE比較"},
    {"group": "B", "idx": 6,  "text": "Terraformの利点"},
    {"group": "B", "idx": 7,  "text": "Multi Cloud戦略"},
    {"group": "B", "idx": 8,  "text": "Observabilityとは何か"},
    {"group": "B", "idx": 9,  "text": "OpenTelemetryの役割"},
    {"group": "B", "idx": 10, "text": "FinOpsとは何か"},
    # ── Backend (10) ──
    {"group": "B", "idx": 11, "text": "CQRSとは何か"},
    {"group": "B", "idx": 12, "text": "Event Sourcingの利点"},
    {"group": "B", "idx": 13, "text": "Sagaパターンとは何か"},
    {"group": "B", "idx": 14, "text": "gRPCの特徴"},
    {"group": "B", "idx": 15, "text": "RESTとの比較"},
    {"group": "B", "idx": 16, "text": "Message Queue設計"},
    {"group": "B", "idx": 17, "text": "Kafkaのパーティション戦略"},
    {"group": "B", "idx": 18, "text": "API Gatewayの役割"},
    {"group": "B", "idx": 19, "text": "GraphQLの課題"},
    {"group": "B", "idx": 20, "text": "Rate Limiting手法"},
    # ── Frontend (10) ──
    {"group": "B", "idx": 21, "text": "React Server Components"},
    {"group": "B", "idx": 22, "text": "Next.js SSRとSSG比較"},
    {"group": "B", "idx": 23, "text": "Web Componentsとは何か"},
    {"group": "B", "idx": 24, "text": "WASM利用例"},
    {"group": "B", "idx": 25, "text": "Virtual DOMとは何か"},
    {"group": "B", "idx": 26, "text": "CSRとSSR比較"},
    {"group": "B", "idx": 27, "text": "状態管理ライブラリ比較"},
    {"group": "B", "idx": 28, "text": "Service Workerの役割"},
    {"group": "B", "idx": 29, "text": "PWAの利点"},
    {"group": "B", "idx": 30, "text": "Browser Rendering Pipeline"},
    # ═══ Group C: 弱α (20本) — 近傍 ═══
    {"group": "C", "idx": 1,  "text": "RTP Headerの構造"},
    {"group": "C", "idx": 2,  "text": "RTCP Receiver Reportの役割"},
    {"group": "C", "idx": 3,  "text": "SDP Offer/Answerとは何か"},
    {"group": "C", "idx": 4,  "text": "ICE Liteとは何か"},
    {"group": "C", "idx": 5,  "text": "STUNとTURNの違い"},
    {"group": "C", "idx": 6,  "text": "VP8とAV1比較"},
    {"group": "C", "idx": 7,  "text": "Simulcastとは何か"},
    {"group": "C", "idx": 8,  "text": "SVCとは何か"},
    {"group": "C", "idx": 9,  "text": "NACKとは何か"},
    {"group": "C", "idx": 10, "text": "PLIとは何か"},
    {"group": "C", "idx": 11, "text": "TWCCとは何か"},
    {"group": "C", "idx": 12, "text": "REMBとは何か"},
    {"group": "C", "idx": 13, "text": "GCCの仕組み"},
    {"group": "C", "idx": 14, "text": "RTP Timestampとは何か"},
    {"group": "C", "idx": 15, "text": "SSRCとは何か"},
    {"group": "C", "idx": 16, "text": "DTLS-SRTPとは何か"},
    {"group": "C", "idx": 17, "text": "RTP Retransmissionとは何か"},
    {"group": "C", "idx": 18, "text": "RTX Payloadとは何か"},
    {"group": "C", "idx": 19, "text": "JSEPとは何か"},
    {"group": "C", "idx": 20, "text": "ORTCとは何か"},
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
        description="FACR バッチ実行 (620 クエリ × ASK + FACR)")
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
