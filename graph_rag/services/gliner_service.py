"""
GLiNER Service — Named Entity Recognition for the dashboard.
Extracts entities from indexed pages using GLiNER models.
"""

import asyncio
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Default labels for entity extraction
DEFAULT_LABELS = [
    "nombre de persona", "universidad", "empresa", "institución",
    "grado universitario", "máster universitario", "doctorado",
    "curso", "asignatura", "especialización",
    "créditos ECTS", "precio", "duración",
    "programming language", "software", "technology",
    "campo de estudio", "sector profesional",
    "ciudad", "país",
]

# ---- Postprocessing: blacklists and normalization ----

GENERIC_BLACKLIST = {
    "asignaturas", "asignatura", "curso", "cursos", "grado", "grados",
    "máster", "masters", "doctorado", "doctorados", "estudios",
    "formación", "formación continua", "programa", "programas",
    "especialización", "titulación", "competencias", "competencia",
    "habilidades", "habilidad", "créditos", "matrícula",
    "empresas", "empresa", "profesionales", "expertos", "estudiantes",
    "estudiantado", "profesorado", "alumnado", "sector", "sectores",
    "tecnología", "tecnologías", "herramienta", "herramientas",
    "software", "aplicación", "plataforma",
    "duración", "precio", "salud", "educación", "investigación",
    "sociedad", "personas", "equipo", "trabajo", "proyecto",
}

DOMAIN_BLACKLIST = {
    "uoc", "uoc.edu", "la uoc", "universitat oberta de catalunya",
    "universitat oberta", "universidad oberta",
}

NOISE_PATTERNS = [
    r"^https?://",
    r"^\d+$",
    r"^[\s\-_]+$",
    r"^.{1,2}$",
    r"^(el|la|los|las|un|una|de|del|en|por|para|con|sin)\s",
]


def _is_generic(text: str) -> bool:
    normalized = text.lower().strip()
    if normalized in GENERIC_BLACKLIST or normalized in DOMAIN_BLACKLIST:
        return True
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, normalized):
            return True
    return False


def _normalize_entity_text(text: str) -> str:
    t = text.strip().strip(".,;:!?()[]{}\"'")
    return re.sub(r"\s+", " ", t)


def _normalize_key(text: str) -> str:
    t = _normalize_entity_text(text).lower()
    t = re.sub(r"^(el|la|los|las|l'|d')\s+", "", t)
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
        "ä": "a", "ë": "e", "ï": "i", "ö": "o", "ü": "u",
        "ñ": "n", "ç": "c",
    }
    for old, new in replacements.items():
        t = t.replace(old, new)
    return t


def _deduplicate_entities(entities: list[dict]) -> list[dict]:
    groups = {}
    for ent in entities:
        key = (_normalize_key(ent["text"]), ent["label"])
        if key not in groups:
            groups[key] = {
                "text": _normalize_entity_text(ent["text"]),
                "label": ent["label"],
                "best_score": ent["score"],
                "count": 1,
                "variants": {ent["text"]},
                "urls": {ent.get("url", "")},
            }
        else:
            g = groups[key]
            g["count"] += 1
            g["variants"].add(ent["text"])
            g["urls"].add(ent.get("url", ""))
            if ent["score"] > g["best_score"]:
                g["best_score"] = ent["score"]
                g["text"] = _normalize_entity_text(ent["text"])

    result = []
    for g in groups.values():
        result.append({
            "text": g["text"],
            "label": g["label"],
            "score": g["best_score"],
            "count": g["count"],
            "variants": sorted(g["variants"]),
            "pages": len(g["urls"] - {""}),
        })
    return sorted(result, key=lambda x: (-x["count"], -x["score"]))


def _postprocess(all_entities: list[dict], min_score: float = 0.6, min_count: int = 1) -> dict:
    total_raw = len(all_entities)
    filtered = [e for e in all_entities if not _is_generic(e["text"])]
    removed_generic = total_raw - len(filtered)
    filtered = [e for e in filtered if e["score"] >= min_score]
    removed_low_score = (total_raw - removed_generic) - len(filtered)
    deduped = _deduplicate_entities(filtered)
    if min_count > 1:
        deduped = [e for e in deduped if e["count"] >= min_count]

    return {
        "entities": deduped,
        "stats": {
            "total_raw": total_raw,
            "removed_generic": removed_generic,
            "removed_low_score": removed_low_score,
            "after_filter": total_raw - removed_generic - removed_low_score,
            "unique_entities": len(deduped),
            "min_score": min_score,
            "min_count": min_count,
        },
    }


# ---- GLiNER model singleton ----

_gliner_model = None
_gliner_model_name = None


def _get_model(model_name: str = "urchade/gliner_large-v2.1"):
    global _gliner_model, _gliner_model_name
    if _gliner_model is None or _gliner_model_name != model_name:
        from gliner import GLiNER
        logger.info(f"Loading GLiNER model: {model_name}")
        _gliner_model = GLiNER.from_pretrained(model_name)
        _gliner_model_name = model_name
        logger.info("GLiNER model loaded")
    return _gliner_model


def _extract_from_text(model, text: str, labels: list[str], threshold: float, max_length: int = 1500):
    truncated = text[:max_length]
    entities = model.predict_entities(truncated, labels, threshold=threshold)
    return [
        {"text": ent["text"], "label": ent["label"], "score": round(ent["score"], 3)}
        for ent in entities
    ]


# ---- Public async API ----

async def extract_entities(
    supabase_client,
    client_id: str,
    sample_size: int | None = None,
    threshold: float = 0.5,
    labels: list[str] | None = None,
    min_score: float = 0.6,
    min_count: int = 1,
    model_name: str = "urchade/gliner_large-v2.1",
) -> dict:
    """
    Extract entities from client pages using GLiNER.

    Returns dict with 'entities', 'stats', and 'pages_processed'.
    """
    if labels is None:
        labels = DEFAULT_LABELS

    # Load pages from Supabase
    query = """
        SELECT url, title, content
        FROM rag_pages
        WHERE client_id = $1 AND content IS NOT NULL AND length(content) > 100
    """
    if sample_size:
        query += f" ORDER BY random() LIMIT {sample_size}"

    async with supabase_client.get_connection() as conn:
        rows = await conn.fetch(query, client_id)
    pages = [dict(r) for r in rows]

    if not pages:
        return {"entities": [], "stats": {"total_raw": 0, "unique_entities": 0}, "pages_processed": 0}

    # Run GLiNER extraction in a thread (synchronous model)
    def _run_extraction():
        model = _get_model(model_name)
        all_entities = []
        for page in pages:
            ents = _extract_from_text(model, page["content"], labels, threshold)
            for ent in ents:
                ent["url"] = page["url"]
            all_entities.extend(ents)
        return all_entities

    all_entities = await asyncio.to_thread(_run_extraction)

    # Postprocess
    result = _postprocess(all_entities, min_score=min_score, min_count=min_count)
    result["pages_processed"] = len(pages)
    return result


def get_model_status() -> dict:
    """Check if GLiNER model is loaded."""
    return {
        "model_loaded": _gliner_model is not None,
        "model_name": _gliner_model_name,
    }
