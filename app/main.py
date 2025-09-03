from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Tuple
import os, csv, json, re, unicodedata
from rapidfuzz import process, fuzz

# =========================
# Loaders
# =========================

def load_codeframes(path: str) -> Dict[str, Dict[str, Tuple[str, int]]]:
    """
    Reads all CSVs in ./codeframes and returns:
      {question_id: {canonical_label: (canonical_label, code)}}
    CSVs must have headers: label,code
    """
    out: Dict[str, Dict[str, Tuple[str, int]]] = {}
    if not os.path.isdir(path):
        print(f"[load_codeframes] Directory not found: {path}")
        return out

    for fname in os.listdir(path):
        if not fname.lower().endswith(".csv"):
            continue
        qid = os.path.splitext(fname)[0]
        fpath = os.path.join(path, fname)

        rows = 0
        out[qid] = {}
        with open(fpath, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "label" not in reader.fieldnames or "code" not in reader.fieldnames:
                raise RuntimeError(
                    f"{fname}: CSV must have header 'label,code' (found {reader.fieldnames})"
                )
            for row in reader:
                lbl = (row.get("label") or "").strip()
                code_str = (row.get("code") or "").strip()
                if not lbl or not code_str:
                    continue
                try:
                    code = int(float(code_str))
                except ValueError:
                    raise RuntimeError(f"{fname}: non-integer 'code' value: {code_str}")
                out[qid][lbl] = (lbl, code)
                rows += 1
        print(f"[load_codeframes] {fname}: loaded {rows} rows")
    if not out:
        print("[load_codeframes] No CSVs loaded")
    return out


def load_aliases(fp: str) -> Dict[str, Dict[str, str]]:
    """
    Loads manual aliases JSON:
      { "Q1a": { "volkswagen": "VW", ... }, "F3": { ... } }
    RHS (canonical) MUST exactly match a label in the corresponding CSV.
    """
    if not os.path.exists(fp):
        print(f"[load_aliases] No alias file at {fp} (skipping)")
        return {}
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[load_aliases] JSON decode error in {fp}: {e}")
        return {}  # fail-soft: run without aliases
    if not isinstance(data, dict):
        print("[load_aliases] aliases.json root must be an object; ignoring")
        return {}
    return data

# =========================
# Normalization helpers
# =========================

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[()]", " ", s)      # drop parentheses for matching
    s = re.sub(r"[\.\-_/]", " ", s)  # break punctuation
    s = re.sub(r"\b&\b", " and ", s) # & -> and
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compact(s: str) -> str:
    return re.sub(r"\s+", "", s)

# =========================
# Matcher
# =========================

class Matcher:
    def __init__(self, codeframes: Dict[str, Dict[str, Tuple[str, int]]], manual_aliases: Optional[Dict[str, Dict[str, str]]] = None):
        self.codeframes = codeframes
        self.manual_aliases = manual_aliases or {}
        self.norm_labels: Dict[str, Dict[str, str]] = {}
        self.auto_aliases: Dict[str, Dict[str, str]] = {}

        for qid, labels in self.codeframes.items():
            self.norm_labels[qid] = {}
            self.auto_aliases[qid] = {}

            for lbl in labels.keys():
                nlbl = normalize(lbl)
                self.norm_labels[qid][lbl] = nlbl

                # auto-aliases derived from label text
                base = re.sub(r"\s*\([^)]*\)", "", lbl).strip()  # remove parentheticals like (VW)
                nbase = normalize(base)
                clbl  = compact(nlbl)
                cbase = compact(nbase)

                if nbase:
                    self.auto_aliases[qid].setdefault(nbase, lbl)
                if cbase:
                    self.auto_aliases[qid].setdefault(cbase, lbl)
                if clbl:
                    self.auto_aliases[qid].setdefault(clbl, lbl)

            # merge manual aliases (normalize keys; only keep entries with existing canonical labels)
            for raw_alias, canonical in self.manual_aliases.get(qid, {}).items():
                na = normalize(raw_alias)
                if canonical in self.codeframes[qid]:
                    self.auto_aliases[qid].setdefault(na, canonical)
                    self.auto_aliases[qid].setdefault(compact(na), canonical)

    def match(self, qid: str, verbatim: str):
        # ----- empty handling -----
        if not verbatim or not verbatim.strip():
            # Q1a: empty → 97
            if qid == "Q1a":
                return ("None", 97, 1.0, "rule-empty", None, [])
            # F3: per spec, fallback only for non-empty; empty returns None
            return None

        if qid not in self.codeframes:
            return None

        text = verbatim.strip()
        norm_text = normalize(text)
        ctext = compact(norm_text)

        # ===== Q1a special rules =====
        if qid == "Q1a":
            # 1) Cyrillic letters
            if re.search(r"[\u0400-\u04FF]", text):
                return ("Cyrillic", 95, 1.0, "rule-cyrillic", None, [])
            # 2) No / none / nothing
            if re.search(r"\b(no|none|nothing)\b", norm_text):
                return ("None", 97, 1.0, "rule-none", None, [])
            # 3) Don't know / dk / no idea
            if re.search(r"\b(dk|don.?t know|dont know|no idea)\b", norm_text):
                return ("DontKnow", 99, 1.0, "rule-dk", None, [])

        # ===== normal alias / substring / fuzzy =====
        labels_map = self.codeframes[qid]
        canon_labels = list(labels_map.keys())

        if not canon_labels:
            if qid == "Q1a":
                return ("Other", 98, 1.0, "rule-empty-frame", None, [])
            # F3 non-empty input with empty frame will be handled by fallback below

        # alias quick hits
        aliases = self.auto_aliases.get(qid, {})
        if norm_text in aliases:
            canonical = aliases[norm_text]
            pair = labels_map.get(canonical)
            if pair:
                _, code = pair
                return (canonical, code, 0.99, "alias", norm_text, [(canonical, 99)])
        if ctext in aliases:
            canonical = aliases[ctext]
            pair = labels_map.get(canonical)
            if pair:
                _, code = pair
                return (canonical, code, 0.99, "alias", ctext, [(canonical, 99)])
        for a, canonical in aliases.items():
            if a and (a in norm_text or a in ctext):
                pair = labels_map.get(canonical)
                if pair:
                    _, code = pair
                    return (canonical, code, 0.96, "alias-sub", a, [(canonical, 96)])

        # bidirectional substring on canonical labels
        for label in canon_labels:
            nlbl = self.norm_labels[qid][label]
            if not nlbl:
                continue
            if (nlbl in norm_text or norm_text in nlbl or
                compact(nlbl) in ctext or ctext in compact(nlbl)):
                _, code = labels_map[label]
                return (label, code, 0.92, "substring", nlbl, [(label, 92)])

        # fuzzy (with score_cutoff support)
        def scorer(q, choice, *, score_cutoff=0):
            s1 = fuzz.token_set_ratio(q, choice, score_cutoff=score_cutoff)
            s2 = fuzz.partial_ratio(q, choice, score_cutoff=score_cutoff)
            return 0.6 * s1 + 0.4 * s2

        if canon_labels:
            choices_map = {label: self.norm_labels[qid][label] for label in canon_labels}
            extracted = process.extract(norm_text, choices_map, scorer=scorer, limit=3)
            if extracted:
                top_label, top_score, _ = extracted[0]
                second = extracted[1][1] if len(extracted) > 1 else 0
                if top_score >= 75 and (top_score - second) >= 5:
                    _, code = labels_map[top_label]
                    return (
                        top_label,
                        code,
                        min(1.0, top_score / 100.0),
                        "fuzzy",
                        self.norm_labels[qid][top_label],
                        [(lab, int(sc)) for (lab, sc, __) in extracted],
                    )

        # =============================
        # Fallbacks (inside the method)
        # =============================

        # Q1a fallback → 98
        if qid == "Q1a":
            return ("Other", 98, 1.0, "rule-fallback", None, [])

        # F3 fallback → 998 for any non-empty input
        if qid == "F3":
            return ("Other", 998, 1.0, "rule-fallback", None, [])

        # Other frames: no forced fallback
        return None

# =========================
# FastAPI app
# =========================

app = FastAPI(title="Autocoder API", version="1.4.0")
STATE: Dict[str, object] = {"matcher": None}

def init():
    codeframes = load_codeframes("./codeframes")
    aliases = load_aliases("./aliases/aliases.json")
    STATE["matcher"] = Matcher(codeframes, aliases)

init()

# ---------- Models ----------

class AutocodeRequest(BaseModel):
    question_id: str
    verbatim: str

class AutocodeResponse(BaseModel):
    question_id: str
    input: str
    matched_label: Optional[str]
    code: Optional[int]
    confidence: float
    method: Optional[str]
    matched_alias_or_term: Optional[str]
    extras: Optional[Dict] = None

class BatchAutocodeRequest(BaseModel):
    question_id: str = Field(..., description="CSV filename without .csv (case-sensitive)")
    verbatims: List[str] = Field(..., description="Up to 10 verbatims")

    @validator("verbatims")
    def validate_verbatims(cls, v):
        if not isinstance(v, list):
            raise ValueError("verbatims must be a list of strings")
        if len(v) == 0:
            raise ValueError("verbatims must contain at least 1 item")
        if len(v) > 10:
            raise ValueError("verbatims cannot exceed 10 items")
        return v

class BatchAutocodeItem(BaseModel):
    input: str
    matched_label: Optional[str]
    code: Optional[int]
    confidence: float
    method: Optional[str]
    matched_alias_or_term: Optional[str]
    extras: Optional[Dict] = None

class BatchAutocodeResponse(BaseModel):
    question_id: str
    results: List[BatchAutocodeItem]

# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug/codeframes")
def debug_codeframes():
    m: Matcher = STATE.get("matcher")  # type: ignore
    if not m:
        return {}
    return {qid: len(labels) for qid, labels in m.codeframes.items()}

@app.get("/debug/aliases")
def debug_aliases():
    m: Matcher = STATE.get("matcher")  # type: ignore
    if not m:
        return {}
    return {qid: len(amap) for qid, amap in m.auto_aliases.items()}

# Optional: inspect alias map samples
@app.get("/debug/alias-map/{qid}")
def debug_alias_map(qid: str, n: int = 25):
    m: Matcher = STATE.get("matcher")  # type: ignore
    if not m:
        return {}
    amap = m.auto_aliases.get(qid, {})
    items = list(amap.items())[:max(0, n)]
    return {"qid": qid, "count": len(amap), "sample": items}

@app.post("/autocode", response_model=AutocodeResponse)
def autocode(payload: AutocodeRequest):
    try:
        m: Matcher = STATE["matcher"]  # type: ignore
        res = m.match(payload.question_id, payload.verbatim)
    except Exception as e:
        # fail-soft: return detail instead of 500
        return AutocodeResponse(
            question_id=payload.question_id,
            input=payload.verbatim,
            matched_label=None,
            code=998 if (payload.question_id == "F3" and (payload.verbatim or "").strip()) else (98 if payload.question_id == "Q1a" else None),
            confidence=1.0 if (payload.question_id in ("Q1a", "F3") and (payload.verbatim or "").strip()) else 0.0,
            method="rule-fallback" if (payload.question_id in ("Q1a", "F3") and (payload.verbatim or "").strip()) else None,
            matched_alias_or_term=None,
            extras={"error": "match_exception", "detail": str(e)},
        )

    if not res:
        # Final guard so F3 never returns null for non-empty inputs
        if payload.question_id == "F3" and (payload.verbatim or "").strip():
            return AutocodeResponse(
                question_id=payload.question_id,
                input=payload.verbatim,
                matched_label="Other",
                code=998,
                confidence=1.0,
                method="rule-fallback",
                matched_alias_or_term=None,
                extras={"reason": "no_match_endpoint_guard"},
            )
        return AutocodeResponse(
            question_id=payload.question_id,
            input=payload.verbatim,
            matched_label=None,
            code=None,
            confidence=0.0,
            method=None,
            matched_alias_or_term=None,
            extras={"reason": "no_match"},
        )

    label, code, conf, method, term, top = res
    return AutocodeResponse(
        question_id=payload.question_id,
        input=payload.verbatim,
        matched_label=label,
        code=code,
        confidence=conf,
        method=method,
        matched_alias_or_term=term,
        extras={"top_candidates": top},
    )

@app.post("/autocode/batch", response_model=BatchAutocodeResponse)
def autocode_batch(payload: BatchAutocodeRequest):
    m: Matcher = STATE["matcher"]  # type: ignore
    qid = payload.question_id

    if qid not in m.codeframes:
        raise HTTPException(status_code=400, detail=f"Unknown question_id '{qid}'")

    results: List[BatchAutocodeItem] = []
    for verb in payload.verbatims:
        try:
            res = m.match(qid, verb)
        except Exception as e:
            code_fallback = 998 if (qid == "F3" and (verb or "").strip()) else (98 if qid == "Q1a" else None)
            results.append(BatchAutocodeItem(
                input=verb,
                matched_label="Other" if code_fallback else None,
                code=code_fallback,
                confidence=1.0 if code_fallback else 0.0,
                method="rule-fallback" if code_fallback else None,
                matched_alias_or_term=None,
                extras={"error": "match_exception", "detail": str(e)}
            ))
            continue

        if not res:
            code_fallback = 998 if (qid == "F3" and (verb or "").strip()) else (98 if qid == "Q1a" else None)
            results.append(BatchAutocodeItem(
                input=verb,
                matched_label="Other" if code_fallback else None,
                code=code_fallback,
                confidence=1.0 if code_fallback else 0.0,
                method="rule-fallback" if code_fallback else None,
                matched_alias_or_term=None,
                extras={"reason": "no_match"}
            ))
            continue

        label, code, conf, method, term, top = res
        results.append(BatchAutocodeItem(
            input=verb,
            matched_label=label,
            code=code,
            confidence=conf,
            method=method,
            matched_alias_or_term=term,
            extras={"top_candidates": top}
        ))

    return BatchAutocodeResponse(question_id=qid, results=results)
