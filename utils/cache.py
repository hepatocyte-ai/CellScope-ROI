"""
Сохранение загруженных пар и кэширование результатов пайплайна.

Структура data/
───────────────
data/
  {pair_hash}/              ← sha1(img_bytes + mask_bytes)[:12]
    meta.json               ← имена файлов, дата сохранения
    image.{ext}
    mask.{ext}
    {run_hash}/             ← sha1(class_names + config + palette + vis_params)[:12]
      full_counts.json
      full_vis.png
      roi_{x0}_{y0}_{x1}_{y1}_counts.json
      roi_{x0}_{y0}_{x1}_{y1}_vis.png
      …
"""

import datetime
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

DATA_DIR = "data"


# ── Хэширование ──────────────────────────────────────────────────────────────

def _stable_str(obj: Any) -> str:
    if isinstance(obj, dict):
        return (
            "{"
            + ",".join(
                f"{_stable_str(k)}:{_stable_str(v)}"
                for k, v in sorted(obj.items(), key=lambda x: str(x[0]))
            )
            + "}"
        )
    if isinstance(obj, (set, frozenset)):
        return "[" + ",".join(_stable_str(v) for v in sorted(obj)) + "]"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_str(v) for v in obj) + "]"
    if hasattr(obj, "__dataclass_fields__"):
        return _stable_str(
            {field: getattr(obj, field) for field in obj.__dataclass_fields__}
        )
    return repr(obj)


def _sha1(*parts: Any, length: int = 12) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else _stable_str(p).encode("utf-8"))
    return h.hexdigest()[:length]


# ── Сохранение загруженной пары ───────────────────────────────────────────────

def ensure_pair_saved(
    img_bytes: bytes,
    mask_bytes: bytes,
    img_name: str,
    mask_name: str,
) -> Tuple[str, str]:
    """
    Сохраняет пару image + mask в data/{pair_hash}/.
    Не перезаписывает файлы, если они уже лежат на диске.
    Возвращает (путь к директории, pair_hash).
    """
    pair_hash = _sha1(img_bytes, mask_bytes)
    d = os.path.join(DATA_DIR, pair_hash)
    os.makedirs(d, exist_ok=True)

    img_ext  = os.path.splitext(img_name)[1]  or ".png"
    mask_ext = os.path.splitext(mask_name)[1] or ".png"

    for fname, data in [
        (f"image{img_ext}",  img_bytes),
        (f"mask{mask_ext}",  mask_bytes),
    ]:
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            with open(path, "wb") as fh:
                fh.write(data)

    # Метаданные — сохраняются один раз при первой загрузке пары
    meta_path = os.path.join(d, "meta.json")
    if not os.path.exists(meta_path):
        meta = {
            "img_name":  img_name,
            "mask_name": mask_name,
            "saved_at":  datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    return d, pair_hash


# ── История загруженных пар ───────────────────────────────────────────────────

def list_saved_pairs() -> List[Dict[str, str]]:
    """
    Возвращает список всех сохранённых пар из data/.
    Каждый элемент: {"pair_hash", "img_name", "mask_name", "saved_at"}.
    Отсортировано по дате — новые сначала.
    """
    result: List[Dict[str, str]] = []
    if not os.path.isdir(DATA_DIR):
        return result

    for pair_hash in os.listdir(DATA_DIR):
        meta_path = os.path.join(DATA_DIR, pair_hash, "meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            meta["pair_hash"] = pair_hash
            result.append(meta)
        except Exception:
            pass

    result.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    return result


def load_pair(
    pair_hash: str,
) -> Optional[Tuple[bytes, bytes, str, str]]:
    """
    Загружает байты изображения и маски из data/{pair_hash}/.
    Возвращает (img_bytes, mask_bytes, img_name, mask_name) или None при ошибке.
    """
    d = os.path.join(DATA_DIR, pair_hash)
    meta_path = os.path.join(d, "meta.json")
    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return None

    img_name  = meta["img_name"]
    mask_name = meta["mask_name"]
    img_ext   = os.path.splitext(img_name)[1]  or ".png"
    mask_ext  = os.path.splitext(mask_name)[1] or ".png"

    img_path  = os.path.join(d, f"image{img_ext}")
    mask_path = os.path.join(d, f"mask{mask_ext}")

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return None

    try:
        with open(img_path,  "rb") as fh:
            img_bytes  = fh.read()
        with open(mask_path, "rb") as fh:
            mask_bytes = fh.read()
    except Exception:
        return None

    return img_bytes, mask_bytes, img_name, mask_name


# ── Кэш результатов пайплайна ─────────────────────────────────────────────────

def run_cache_dir(pair_hash: str, *config_parts: Any) -> str:
    run_hash = _sha1(*config_parts)
    return os.path.join(DATA_DIR, pair_hash, run_hash)


def load_cached_result(
    cache_dir: str,
    prefix: str,
) -> Optional[Tuple[Dict[str, int], np.ndarray]]:
    counts_path = os.path.join(cache_dir, f"{prefix}_counts.json")
    vis_path    = os.path.join(cache_dir, f"{prefix}_vis.png")
    if os.path.exists(counts_path) and os.path.exists(vis_path):
        try:
            with open(counts_path, encoding="utf-8") as f:
                counts = json.load(f)
            vis = np.array(Image.open(vis_path))
            return counts, vis
        except Exception:
            return None
    return None


def save_cached_result(
    cache_dir: str,
    prefix: str,
    counts: Dict[str, int],
    vis_img: np.ndarray,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(
        os.path.join(cache_dir, f"{prefix}_counts.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    Image.fromarray(vis_img).save(os.path.join(cache_dir, f"{prefix}_vis.png"))