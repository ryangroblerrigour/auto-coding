from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import os, csv, json, re, unicodedata
from rapidfuzz import process, fuzz

# =========================
# Loaders
# =========================

def load_codeframes(path: str) -> Dict[str, Dict[str, tuple]]:
    """
    Reads all CSVs in ./codeframes and returns:
      {question_id: {canonical_label: (canonical_label, code)}}
    CSVs must have headers: label,code
    """
    out: Dict[str, Dict[str, tuple]] = {}
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
      { "Q1a": { "volkswagen": "VW", ... }, "f3": { ... } }
    RHS (canonical) MUST exactly match a label in the corresponding CSV.
    """
    if not os.path.exists(fp):
        print(f"[load_aliases] No alias file at {fp} (skipping)")
        return {}
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise RuntimeError("[load_aliases] aliases.json root must be an object")
        return data

# =========================
# Normalization helpers
# =========================

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[()]", " ", s)            # drop parentheses for matching
    s = re.sub(r"[\.\-_/]", " ", s)        # break punctuation
    s = re.sub(r"\b&\b", " and ", s)       # & -> and
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compact(s: str) -> str:
    return re.sub(r"\s+", "", s)

# =========================
# Matcher
# =========================

class Matcher:
    def __init__(self, codeframes: Dict[str, Dict[str, tuple]], manual_aliases: Optional[Dict[str, Dict[str, str]]] = None):
        self.codeframes = codeframes                      # {qid: {label: (label, code)}}
        self.manual_aliases = manual_aliases or {}        # {qid: {alias_text: canonical_label}}
        self.norm_labels: Dict[str, Dict[str, str]] = {}  # {qid: {label: normalized_label}}
        self.auto_aliases: Dict[str, Dict[str, str]] = {} # {qid: {alias_norm_or_compact: canonical_label}}

        for qid, labels in self.codeframes.items():
            self.norm_labels[qid] = {}
            self.auto_aliases[qid] = {}

            for lbl in labels.keys():
                nlbl = normalize(lbl)
                self.norm_labels[qid][lbl] = nlbl

                # Build lightweight auto-aliases from the label itself
                base = re.sub(r"\s*\([^)]*\)", "", lbl).strip()  # remove parenthetical like (VW)
                nbase = normalize(base)
                clbl  = compact(nlbl)
                cbase = compact(nbase)

                # Add normalized base label + compact forms as aliases to the canonical label
                if nbase:
                    self.auto_aliases[qid].setdefault(nbase, lbl)
                if cbase:
                    self.auto_aliases[qid].setdefault(cbase, lbl)
                if clbl:
                    self.auto_aliases[qid].setdefault(clbl,  lbl)

            # Merge manual aliases — normalize keys; only keep entries with existing canonical labels
            for raw_alias, canonical in self.manual_aliases.get(qid, {}).items():
                na = normalize(raw_alias)
                if canonical in self.codeframes[qid]:
                    self.auto_aliases[qid].setdefault(na, canonical)
                    self.auto_aliases[qid].setdefault(compact(na), canonical)

    def match(self, qid: str, verbatim: str):
        if not verbatim or not verbatim.strip():
            return None
        if qid not in self.codeframes:
            return None

        labels_map = self.codeframes[qid]  # {label: (label, code)}
        canon_labels = list(labels_map.keys())
        if not canon_labels:
            return None

        norm_text = normalize(verbatim)
        ctext = compact(norm_text)

        # 0) alias quick hits (manual + auto)
        aliases = self.auto_aliases.get(qid, {})
        if norm_text in aliases:
            canonical = aliases[norm_text]
            _, code = labels_map[canonical]
            return (canonical, code, 0.99, "alias", norm_text, [(canonical, 99)])
        if ctext in aliases:
            canonical = aliases[ctext]
            _, code = labels_map[canonical]
            return (canonical, code, 0.99, "alias", ctext, [(canonical, 99)])
        # alias substring containment (e.g., 'volks wagon' inside input)
        for a, canonical in aliases.items():
            if a and (a in norm_text or a in ctext):
                _, code = labels_map[canonical]
                return (canonical, code, 0.96, "alias-sub", a, [(canonical, 96)])

        # 1) bidirectional substring on canonical labels
        for label in canon_labels:
            nlbl = self.norm_labels[qid][label]
            if not nlbl:
                continue
            if (nlbl in norm_text or norm_text in nlbl or
                compact(nlbl) in ctext or ctext in compact(nlbl)):
                _, code = labels_map[label]
                return (label, code, 0.92, "substring", nlbl, [(label, 92)])

        # 2) fuzzy (with scorer that supports score_cutoff)
        def scorer(q, choice, *, score_cutoff=0):
            s1 = fuzz.token_set_ratio(q, choice, score_cutoff=score_cutoff)
            s2 = fuzz.partial_ratio(q, choice, score_cutoff=score_cutoff)
            return 0.6 * s1 + 0.4 * s2

        choices_map = {label: self.norm_labels[qid][label] for label in canon_labels}
        extracted = process.extract(norm_text, choices_map, scorer=scorer, limit=3)

        if extracted:
            top_label, top_score, _ = extracted[0]
            second = extracted[1][1] if len(extracted) > 1 else 0
            threshold = 75
            margin = 5
            if top_score >= threshold and (top_score - second) >= margin:
                _, code = labels_map[top_label]
                return (
                    top_label,
                    code,
                    min(1.0, top_score / 100.0),
                    "fuzzy",
                    self.norm_labels[qid][top_label],
                    [(lab, int(sc)) for (lab, sc, __) in extracted],
                )
        return None

# =========================
# FastAPI app
# =========================

app = FastAPI(title="Autocoder API", version="1.1.0")
STATE: Dict[str, object] = {"matcher": None}

def init():
    codeframes = load_codeframes("./codeframes")
    aliases = load_aliases("./aliases/aliases.json")
    STATE["matcher"] = Matcher(codeframes, aliases)

init()

class AutocodeRequest(BaseModel):
    question_id: str  # must match CSV filename without .csv (case-sensitive)
    verbatim: str     # brand text only is fine

class AutocodeResponse(BaseModel):
    question_id: str
    input: str
    matched_label: Optional[str]
    code: Optional[int]
    confidence: float
    method: Optional[str]
    matched_alias_or_term: Optional[str]
    extras: Optional[Dict] = None

@app.get("/health")
def health():
    return {"status": "ok"}

# Debug helper: counts per frame
@app.get("/debug/codeframes")
def debug_codeframes():
    m: Matcher = STATE.get("matcher")  # type: ignore
    if not m:
        return {}
    return {qid: len(labels) for qid, labels in m.codeframes.items()}

# Debug helper: alias counts per frame
@app.get("/debug/aliases")
def debug_aliases():
    m: Matcher = STATE.get("matcher")  # type: ignore
    if not m:
        return {}
    return {qid: len(amap) for qid, amap in m.auto_aliases.items()}

@app.post("/autocode", response_model=AutocodeResponse)
def autocode(payload: AutocodeRequest):
    try:
        m: Matcher = STATE["matcher"]  # type: ignore
        res = m.match(payload.question_id, payload.verbatim)
    except Exception as e:
        # fail-soft: return detail in extras rather than 500
        return AutocodeResponse(
            question_id=payload.question_id,
            input=payload.verbatim,
            matched_label=None,
            code=None,
            confidence=0.0,
            method=None,
            matched_alias_or_term=None,
            extras={"error": "match_exception", "detail": str(e)},
        )

    if not res:
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
