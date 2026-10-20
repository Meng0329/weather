"""离线本构语义特征提取器：生成 `h_cur`（实时灾情文本）与 `h_rag`（历史检索先验）。

设计要点
--------
1. **双模式**：优先使用本地可用的中文预训练编码器（sentence-transformers / transformers）；
   若本地无可用权重，则退化为**确定性**的 TF-IDF + SVD 领域词典表征。
   两种模式均保证：同一输入 -> 同一向量（可复现），且不依赖网络。
2. **严防特征泄露**：
   - `h_pre`  只编码**灾前可见**文本（`weather_summary` + `warning_info` + `remarks`），
     可在任意预测时刻 t 使用；
   - `h_post` 额外编码 `impact_description`（灾情发生后的描述），
     **只允许在 Task 3 的观测到达时刻 t_k 注入**；
   - `h_rag`  只在**训练集**范围内检索相似历史，避免用未来样本做先验。
3. **产物**：`<region>/llm_embeddings.pt`，含
   `record_ids / node_idx / date / h_pre / h_post / h_rag / dim / method`。

运行：
  python3 -I src/llm/extract_constitutive_emb.py --regions SW_Cascade SE_Typhoon North_Agro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import common as C  # noqa: E402

log = logging.getLogger("extract_constitutive_emb")

# 灾前可见文本（预测时刻 t 可用）
PRE_FIELDS = ["weather_summary", "warning_info", "remarks"]
# 灾后文本（仅 Task 3 观测到达时刻可用）
POST_FIELDS = ["impact_description"]

# 本地中文编码器候选（按优先级）
LOCAL_ENCODER_CANDIDATES = [
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-base-zh-v1.5",
    "shibing624/text2vec-base-chinese",
    "bert-base-chinese",
]

# 内网模型服务（OpenAI 兼容）：高并发嵌入 / rerank
# ⚠️ 凭据绝不入库：优先读环境变量，其次读仓库外的本地 env 文件
_KEY_FILE = Path(os.environ.get("PTGNO_ENV_FILE", "~/.config/ptgno/env")).expanduser()


def _load_local_env(path: Path) -> None:
    """把本地密钥文件逐行注入 os.environ（已存在的环境变量优先，不覆盖）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_local_env(_KEY_FILE)

API_BASE = os.environ.get("LLM_API_BASE", "http://10.200.37.71:8801")
API_KEY = os.environ.get("LLM_API_KEY", "")
API_EMBED_MODEL = os.environ.get("LLM_EMBED_MODEL", "Qwen3-Embedding-0.6B")
API_RERANK_MODEL = os.environ.get("LLM_RERANK_MODEL", "bge-reranker-v2-m3")

# 领域灾害词典：TF-IDF 退化模式下作为确定性补充特征
DOMAIN_LEXICON = {
    "暴雨": 3.0, "洪涝": 3.0, "内涝": 2.5, "洪水": 3.0, "山洪": 2.5,
    "台风": 3.0, "大风": 2.5, "冰雹": 2.5, "雷电": 2.5, "干旱": 2.5,
    "高温": 2.0, "低温": 2.0, "寒潮": 2.0, "雪灾": 2.0, "暴雪": 2.0,
    "冰冻": 2.0, "冻害": 2.0, "沙尘暴": 2.0, "雾霾": 2.0, "大雾": 1.5,
    "连阴雨": 2.0, "地质灾害": 3.0, "滑坡": 3.0, "泥石流": 3.0, "崩塌": 3.0,
    "红色预警": 3.0, "橙色预警": 2.5, "黄色预警": 2.0, "蓝色预警": 1.5,
    "预警": 1.5, "转移": 2.0, "安置": 1.5, "救援": 1.5, "抢险": 1.5,
    "停电": 1.5, "停课": 2.0, "停产": 2.0, "溃坝": 2.5, "决口": 2.5, "漫堤": 2.5,
    "淹没": 2.0, "积水": 1.5, "农田": 1.5, "农作物": 1.5, "房屋": 1.5,
    "伤亡": 1.5, "死亡": 1.5, "失踪": 1.5,
}


# ==========================================================================
# 文本组装
# ==========================================================================
def _clean_text(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def build_text_frame(records: pd.DataFrame) -> pd.DataFrame:
    """构造 pre / post / rag 三路文本。"""
    out = pd.DataFrame(index=records.index)
    pre_parts = []
    for f in PRE_FIELDS:
        if f in records.columns:
            pre_parts.append(records[f].map(_clean_text))
    pre = pd.Series([" ".join(p) for p in zip(*pre_parts)] if pre_parts else [""] * len(records),
                    index=records.index)
    post_parts = []
    for f in POST_FIELDS:
        if f in records.columns:
            post_parts.append(records[f].map(_clean_text))
    post = pd.Series([" ".join(p) for p in zip(*post_parts)] if post_parts else [""] * len(records),
                     index=records.index)
    out["text_pre"] = pre
    out["text_post"] = post
    if "search_text" in records.columns:
        out["text_rag"] = records["search_text"].map(_clean_text)
    else:
        out["text_rag"] = out["text_pre"]
    return out


# ==========================================================================
# 编码器
# ==========================================================================
class DeterministicEncoder:
    """TF-IDF + 截断 SVD + 领域词典哈希。完全确定性，无外部依赖。"""

    name = "tfidf-svd-lexicon"

    def __init__(self, dim: int = 256, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.vectorizer = None
        self.svd = None

    def fit(self, texts: list[str]) -> "DeterministicEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(1, 3), max_features=60000,
            min_df=2, sublinear_tf=True, lowercase=False,
        )
        X = self.vectorizer.fit_transform(texts)
        k = min(self.dim - 1, max(2, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self.svd.fit(X)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD  # noqa: F401

        X = self.vectorizer.transform(texts)
        Z = self.svd.transform(X)
        # 领域词典哈希特征：把灾害术语与预警级别显式编码进向量
        lex = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            if not t:
                continue
            for term, w in DOMAIN_LEXICON.items():
                if term in t:
                    h = int(hashlib.md5(term.encode("utf-8")).hexdigest()[:8], 16) % 64
                    lex[i, h] += w
        nrm = np.linalg.norm(lex, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        lex = lex / nrm
        out = np.hstack([Z.astype(np.float32), lex]).astype(np.float32)
        if out.shape[1] < self.dim:
            out = np.hstack([out, np.zeros((len(texts), self.dim - out.shape[1]), np.float32)])
        return out[:, : self.dim]



class APIEncoder:
    """通过内网 OpenAI 兼容服务做嵌入（Qwen3-Embedding-0.6B）。

    - 线程池并发（默认 32），batch 32 条/请求，失败自动重试 3 次；
    - 结果按文本内容做**内容寻址缓存**（同一文本全局只请求一次），
      既省算力又保证同一输入 -> 同一向量；
    - 归一化后返回，与本地编码器语义一致（余弦检索）。
    """

    name = "api:Qwen3-Embedding-0.6B"

    def __init__(self, dim: int = 1024, batch: int = 128, workers: int = 96,
                 base: str = API_BASE, key: str = API_KEY, model: str = API_EMBED_MODEL):
        self.dim = dim                 # 目标（MRL 截断后）维度
        self.raw_dim = dim             # 服务原生维度，fit 时探测
        self.batch = batch
        self.workers = workers
        if not key:
            raise RuntimeError(
                "缺少 LLM_API_KEY：请设置环境变量或写入 ~/.config/ptgno/env（该文件不入库）")
        self.base, self.key, self.model = base.rstrip("/"), key, model
        self.cache: dict[str, np.ndarray] = {}
        self.n_calls = 0
        self._last_save = 0.0
        self.save_interval = float(os.environ.get("LLM_CACHE_SAVE_SEC", "180"))
        self._lock = threading.Lock()
        # 磁盘缓存：文本 -> 向量，跨进程/跨区域复用，避免重复付费与限流
        self.cache_path = Path(os.environ.get(
            "LLM_EMBED_CACHE", "datasets/processed/llm_embed_cache.pkl"))
        if self.cache_path.exists():
            try:
                with self.cache_path.open("rb") as fh:
                    self.cache = pickle.load(fh)
                log.info("载入嵌入磁盘缓存 %d 条：%s", len(self.cache), self.cache_path)
            except Exception as e:                       # noqa: BLE001
                log.warning("嵌入缓存损坏，忽略（%s）", e)
                self.cache = {}

    def save_cache(self, force: bool = False) -> None:
        """原子写盘；并发/多进程场景下失败也不致命（缓存可重建）。

        缓存可能上百 MB，逐批全量重写会把 IO 打满、反而拖慢提取，
        因此默认按 `save_interval` 限频，进程退出时再强制写一次。
        """
        now = time.time()
        if not force and now - self._last_save < self.save_interval:
            return
        self._last_save = now
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
            with tmp.open("wb") as fh:
                pickle.dump(self.cache, fh, protocol=4)
            os.replace(tmp, self.cache_path)
        except Exception as e:                           # noqa: BLE001
            log.warning("嵌入缓存写盘失败（忽略）：%s", e)

    # -- 单请求 -------------------------------------------------------
    MAX_CHARS = 1200          # 单条文本截断长度（超出模型上下文的记录会 500）

    def _post(self, texts: list[str]) -> np.ndarray:
        import urllib.request

        texts = [t[: self.MAX_CHARS] if t else "无" for t in texts]
        payload = json.dumps({"model": self.model, "input": texts,
                              "encoding_format": "float"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/v1/embeddings", data=payload,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    body = json.loads(r.read().decode("utf-8"))
                data = sorted(body["data"], key=lambda d: d["index"])
                return np.asarray([d["embedding"] for d in data], dtype=np.float32)
            except Exception as e:                      # noqa: BLE001
                last = e
                time.sleep(0.4 * (attempt + 1))
        # 整批失败：对半拆分递归（定位到单条脏数据），保证长跑不中断
        if len(texts) > 1:
            mid = len(texts) // 2
            return np.vstack([self._post(texts[:mid]), self._post(texts[mid:])])
        log.warning("单条文本嵌入失败，置零向量：%s", last)
        return np.zeros((1, self.raw_dim), dtype=np.float32)

    def fit(self, texts: list[str]) -> "APIEncoder":
        # 探测：拿一条真实文本确认维度与协议
        probe = [t if t else "无" for t in (texts[:1] or ["测试"])]
        v = self._post(probe)
        self.raw_dim = int(v.shape[-1])
        return self

    def _normalize(self, V: np.ndarray) -> np.ndarray:
        """Matryoshka 截断（Qwen3-Embedding 原生支持 MRL）+ L2 归一化。

        截断后重新归一化，保证余弦检索与精排语义一致，同时把每区域的
        文本张量体积压到 1/2（1024 -> 512 维）。
        """
        if V.shape[-1] > self.dim:
            V = V[:, : self.dim]
        n = np.linalg.norm(V, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return (V / n).astype(np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        texts = [t if t else "无" for t in texts]
        todo = list(dict.fromkeys(t for t in texts if t not in self.cache))
        chunks = [todo[i:i + self.batch] for i in range(0, len(todo), self.batch)]

        def work(chunk):
            V = self._post(chunk)
            with self._lock:
                self.n_calls += 1
                for t, v in zip(chunk, V):
                    self.cache[t] = v
            return len(chunk)

        if chunks:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                done = sum(ex.map(work, chunks))
            log.info("嵌入服务：%d 条新文本，%d 次请求（缓存命中 %d 条）",
                     done, len(chunks), len(texts) - done)
            self.save_cache()
        V = np.stack([self.cache[t] for t in texts]).astype(np.float32)
        return self._normalize(V)


class APIReranker:
    """内网 rerank 服务（bge-reranker-v2-m3），用于 RAG 先验的精排。

    - **自适应降级**：首次调用测延迟，服务被压满（> `max_latency` 秒）时
      自动关闭精排、退回稠密余弦 Top-k；
    - 结果按 `(query, docs)` 内容寻址缓存到磁盘，跨区域复用。
    """

    def __init__(self, base: str = API_BASE, key: str = API_KEY,
                 model: str = API_RERANK_MODEL, max_latency: float = 1.0):
        self.base, self.key, self.model = base.rstrip("/"), key, model
        self.max_latency = max_latency
        self.enabled = True
        self._lat_window: list[float] = []      # 最近若干次调用的延迟
        self._last_save = 0.0
        self.save_interval = float(os.environ.get("LLM_CACHE_SAVE_SEC", "180"))
        self._lock = threading.Lock()
        self.cache_path = Path(os.environ.get(
            "LLM_RERANK_CACHE", "datasets/processed/llm_rerank_cache.pkl"))
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            try:
                with self.cache_path.open("rb") as fh:
                    self.cache = pickle.load(fh)
                log.info("载入精排磁盘缓存 %d 条", len(self.cache))
            except Exception:                            # noqa: BLE001
                self.cache = {}

    def save_cache(self, force: bool = False) -> None:
        """原子写盘；并发/多进程场景下失败也不致命（缓存可重建）。

        缓存可能上百 MB，逐批全量重写会把 IO 打满、反而拖慢提取，
        因此默认按 `save_interval` 限频，进程退出时再强制写一次。
        """
        now = time.time()
        if not force and now - self._last_save < self.save_interval:
            return
        self._last_save = now
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
            with tmp.open("wb") as fh:
                pickle.dump(self.cache, fh, protocol=4)
            os.replace(tmp, self.cache_path)
        except Exception as e:                           # noqa: BLE001
            log.warning("嵌入缓存写盘失败（忽略）：%s", e)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        import urllib.request

        if not docs or not self.enabled:
            return np.zeros(len(docs), dtype=np.float32)
        key = hashlib.md5((query + "\u0001" + "\u0002".join(docs)).encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        payload = json.dumps({"model": self.model, "query": query,
                              "documents": docs[:64]}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/v1/rerank", data=payload,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        for attempt in range(3):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=45) as r:
                    body = json.loads(r.read().decode("utf-8"))
                dt = time.time() - t0
                rows = sorted(body, key=lambda d: d["index"])
                out = np.zeros(len(docs), dtype=np.float32)
                out[: len(rows)] = [float(d["score"]) for d in rows]
                # 只看首次探针会被单条请求误导（并发后延迟会涨 10 倍），
                # 因此用最近 32 次调用的滑动平均判定是否降级。
                with self._lock:
                    self._lat_window.append(dt)
                    if len(self._lat_window) > 32:
                        self._lat_window.pop(0)
                    avg = sum(self._lat_window) / len(self._lat_window)
                    if self.enabled and len(self._lat_window) >= 16 and avg > self.max_latency:
                        self.enabled = False
                        log.warning("精排并发延迟均值 %.2fs > %.1fs，自动关闭精排"
                                    "（退回稠密余弦 Top-k）", avg, self.max_latency)
                with self._lock:
                    self.cache[key] = out
                return out
            except Exception:                           # noqa: BLE001
                time.sleep(0.3 * (attempt + 1))
        return np.zeros(len(docs), dtype=np.float32)


class LocalTransformerEncoder:
    """本地预训练中文编码器：transformers + 注意力掩码均值池化（完全离线）。

    只使用 HuggingFace 本地缓存（`local_files_only=True`），不发起任何网络请求；
    若候选权重均不在本地缓存中则抛错，由 `build_encoder` 退化到确定性模式。
    """

    def __init__(self, model_name: str, dim: int = 256, max_length: int = 256):
        self.model_name = model_name
        self.dim = dim
        self.max_length = max_length
        self.model = None
        self.tok = None
        self.backend = None
        self.device = "cpu"

    def _available(self) -> bool:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if torch.cuda.is_available():
            self.device = "cuda"
        for cand in LOCAL_ENCODER_CANDIDATES:
            try:
                tok = AutoTokenizer.from_pretrained(cand, local_files_only=True)
                mdl = AutoModel.from_pretrained(cand, local_files_only=True)
            except Exception:
                continue
            mdl.eval().to(self.device)
            for p in mdl.parameters():
                p.requires_grad_(False)
            self.tok, self.model = tok, mdl
            self.model_name = cand
            self.backend = "transformers-meanpool"
            return True
        return False

    def fit(self, texts: list[str]) -> "LocalTransformerEncoder":
        if not self._available():
            raise RuntimeError("本地无可用中文编码器")
        return self

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        import torch

        m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def transform(self, texts: list[str]) -> np.ndarray:
        import torch

        out = []
        B = 128
        with torch.inference_mode():
            for i in range(0, len(texts), B):
                chunk = [t if t else "无" for t in texts[i:i + B]]
                enc = self.tok(chunk, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hidden = self.model(**enc).last_hidden_state
                vec = self._mean_pool(hidden, enc["attention_mask"])
                vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
                out.append(vec.float().cpu().numpy())
        V = np.vstack(out).astype(np.float32)
        if V.shape[1] > self.dim:
            V = V[:, : self.dim]
            # 截断后重新归一化，保证余弦检索语义一致
            n = np.linalg.norm(V, axis=1, keepdims=True)
            n[n == 0] = 1.0
            V = (V / n).astype(np.float32)
        elif V.shape[1] < self.dim:
            V = np.hstack([V, np.zeros((len(V), self.dim - V.shape[1]), np.float32)])
        return V


def build_encoder(dim: int, prefer_transformer: bool = True, mode: str = "auto"):
    """返回 (encoder, method)。

    mode:
      - "api"    : 强制内网嵌入服务（Qwen3-Embedding-0.6B，1024 维）
      - "local"  : 强制本地 transformers 编码器
      - "tfidf"  : 强制确定性退化模式
      - "auto"   : api -> local -> tfidf 依次尝试
    """
    if mode in ("auto", "api"):
        try:
            enc = APIEncoder(dim=dim)
            enc.fit([])
            log.info("使用内网嵌入服务: %s (%d->%d 维)", enc.model, enc.raw_dim, enc.dim)
            return enc, enc.name.replace("Qwen3-Embedding-0.6B", enc.model)
        except Exception as e:                              # noqa: BLE001
            if mode == "api":
                raise
            log.warning("内网嵌入服务不可用（%s），尝试本地编码器", e)
    if mode in ("auto", "local") and prefer_transformer:
        log.info("尝试加载本地中文预训练编码器 ...")
        enc = LocalTransformerEncoder(LOCAL_ENCODER_CANDIDATES[0], dim=dim)
        try:
            enc.fit([])
            log.info("使用本地编码器: %s (%s)", enc.model_name, enc.backend)
            return enc, f"{enc.backend}:{enc.model_name}"
        except Exception as e:
            log.warning("本地编码器不可用（%s），退化为确定性 TF-IDF + 领域词典", e)
    enc = DeterministicEncoder(dim=dim)
    return enc, enc.name


# ==========================================================================
# RAG 先验
# ==========================================================================
def build_rag_prior(
    encoder,
    query_texts: list[str],
    corpus_texts: list[str],
    top_k: int = 3,
    batch: int = 512,
    reranker=None,
    recall_k: int = 20,
) -> np.ndarray:
    """在给定语料（必须是训练集）中检索 Top-k 相似文本并取均值。"""
    if not corpus_texts:
        return np.zeros((len(query_texts), encoder.dim), dtype=np.float32)
    corpus_emb = []
    for i in range(0, len(corpus_texts), batch):
        corpus_emb.append(encoder.transform(corpus_texts[i:i + batch]))
    Cemb = np.vstack(corpus_emb)
    norms = np.linalg.norm(Cemb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Cemb = Cemb / norms

    # 关键：先把**全部**查询文本一次性嵌入。若逐批嵌入，每批只有 3~4 个请求
    # 在飞，服务端延迟被放大十倍；整表一次性提交才能把并发打满（后续均为命中缓存）。
    encoder.transform(list(query_texts))

    out = np.zeros((len(query_texts), encoder.dim), dtype=np.float32)
    for i in range(0, len(query_texts), batch):
        q = encoder.transform(query_texts[i:i + batch])
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        q = q / qn
        sim = q @ Cemb.T                     # (b, n_corpus)
        if reranker is not None:
            # 两阶段 RAG：先用稠密向量粗召回 recall_k，再用 bge-reranker 精排取 top_k。
            # 查询之间彼此独立，故用线程池并发打满高并发服务。
            kr = min(recall_k, sim.shape[1])
            cand = np.argpartition(-sim, kr - 1, axis=1)[:, :kr]
            rows = list(range(sim.shape[0]))

            def _one(r):
                docs = [corpus_texts[j] for j in cand[r]]
                sc = reranker.score(query_texts[i + r], docs)
                if float(np.abs(sc).sum()) == 0.0:      # 服务异常 -> 回退余弦
                    order = np.argsort(-sim[r, cand[r]])[: min(top_k, kr)]
                else:
                    order = np.argsort(-sc)[: min(top_k, len(docs))]
                return r, Cemb[cand[r][order]].mean(axis=0)

            with ThreadPoolExecutor(max_workers=64) as ex:
                for r, vec in ex.map(_one, rows):
                    out[i + r] = vec
            if hasattr(reranker, "save_cache"):
                reranker.save_cache()
            log.info("RAG 进度：%d/%d", min(i + batch, len(query_texts)), len(query_texts))
            continue
        k = min(top_k, sim.shape[1])
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        picked = Cemb[idx]                   # (b, k, dim)
        out[i:i + batch] = picked.mean(axis=1)
    return out


# ==========================================================================
# 主流程
# ==========================================================================
def extract_region(region: str, processed: Path, dim: int, prefer_transformer: bool,
                   top_k: int, mode: str = "auto", use_rerank: bool = True) -> dict:
    t0 = time.time()
    out_dir = processed / region
    records = pd.read_parquet(out_dir / "records.parquet")
    if "record_id" not in records.columns:
        raise RuntimeError(f"{region}: records.parquet 缺少 record_id")

    # 需要原始文本字段：从全局 disaster_cleaned 合并
    global_dis = pd.read_parquet(
        processed / "disaster_cleaned.parquet",
        columns=["record_id", "weather_summary", "warning_info", "remarks",
                 "impact_description", "search_text"],
    )
    rec = records.merge(global_dis, on="record_id", how="left", suffixes=("", "_g"))
    for c in ("weather_summary", "warning_info", "remarks", "impact_description", "search_text"):
        if f"{c}_g" in rec.columns:
            rec[c] = rec[c].fillna(rec[f"{c}_g"])

    texts = build_text_frame(rec)

    # 训练集语料（仅训练集，避免用未来样本构造先验）
    is_train = rec["split"] == "train"
    corpus = texts.loc[is_train, "text_rag"].tolist()
    log.info("%s: 记录 %d 条（训练语料 %d 条）", region, len(rec), len(corpus))

    encoder, method = build_encoder(dim, prefer_transformer, mode=mode)
    encoder.fit(corpus + texts["text_pre"].tolist()[: min(2000, len(texts))])

    h_pre = encoder.transform(texts["text_pre"].tolist())
    h_post = encoder.transform(texts["text_post"].tolist())
    h_rag = build_rag_prior(encoder, texts["text_rag"].tolist(), corpus, top_k=top_k,
                            reranker=APIReranker() if (use_rerank and isinstance(encoder, APIEncoder)) else None)

    payload = {
        "record_ids": rec["record_id"].to_numpy(),
        "node_idx": rec["node_idx"].to_numpy().astype(np.int64),
        "date": pd.to_datetime(rec["start_date"]).to_numpy(dtype="datetime64[D]"),
        "split": rec["split"].to_numpy(),
        "h_pre": h_pre.astype(np.float32),
        "h_post": h_post.astype(np.float32),
        "h_rag": h_rag.astype(np.float32),
        "dim": int(dim),
        "method": method,
        "top_k_rag": top_k,
    }
    torch = _torch()
    path = out_dir / "llm_embeddings.pt"
    if torch is not None:
        torch.save(payload, path)
    else:
        np.savez_compressed(path.with_suffix(".npz"), **{
            k: v for k, v in payload.items() if isinstance(v, np.ndarray)})
        (out_dir / "llm_embeddings_meta.json").write_text(
            json.dumps({k: v for k, v in payload.items()
                        if not isinstance(v, np.ndarray)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    log.info("%s: 写出 %s  dim=%d method=%s  用时 %.1fs",
             region, path.name, dim, method, time.time() - t0)
    return {
        "region": region,
        "n_records": int(len(rec)),
        "n_train_corpus": int(len(corpus)),
        "dim": int(dim),
        "method": method,
        "top_k_rag": top_k,
        "seconds": round(time.time() - t0, 1),
    }


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2: 离线本构语义特征提取")
    ap.add_argument("--processed-dir", type=Path, default=C.PROCESSED_DIR)
    ap.add_argument("--regions", nargs="*", default=list(C.REGIONS))
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--top-k-rag", type=int, default=3)
    ap.add_argument("--encoder", default="auto",
                    choices=["auto", "api", "local", "tfidf"],
                    help="编码器后端：api=内网嵌入服务，local=本地 transformers，tfidf=确定性退化")
    ap.add_argument("--no-rerank", action="store_true",
                    help="RAG 精排关闭（仅用余弦 Top-k）")
    ap.add_argument("--overwrite", action="store_true",
                    help="覆盖已存在的 llm_embeddings.pt")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout, force=True,
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    stats = []
    for region in args.regions:
        out_pt = args.processed_dir / region / "llm_embeddings.pt"
        if out_pt.exists() and not args.overwrite:
            log.info("%s: 已存在 %s，跳过（--overwrite 可覆盖）", region, out_pt.name)
            continue
        stats.append(extract_region(
            region, args.processed_dir, args.dim,
            prefer_transformer=(args.encoder in ("auto", "local")),
            top_k=args.top_k_rag, mode=args.encoder,
            use_rerank=not args.no_rerank,
        ))

    out = args.processed_dir / "llm_embeddings_report.json"
    out.write_text(json.dumps({"regions": stats}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    log.info("写出 %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
