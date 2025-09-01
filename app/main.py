from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import csv, os, re, unicodedata
from rapidfuzz import process, fuzz

# ------------------
# loader (robust)
# ------------------
def load_codeframes(path: str):
    out = {}
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
        with open(fpath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "label" not in reader.fieldnames or "code" not in reader.fieldnames:
                raise RuntimeError(f"{fname}: CSV must have header 'label,code' (found {reader.fieldnames})")
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
    return out

# # ------------------
# matcher (improved)
# ------------------
import re, unicodedata
from rapidfuzz import process, fuzz

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[()]", " ", s)            # remove parentheses for matching
    s = re.sub(r"[\.\-_/]", " ", s)        # break punctuation
    s = re.sub(r"\b&\b", " and ", s)       # & -> and
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compact(s: str) -> str:
    return re.sub(r"\s+", "", s)

def extract_parenthetical_acronyms(label: str):
    """
    From 'Volkswagen (VW)' -> ['vw'].
    From 'Procter & Gamble (P&G)' -> ['p&g','pg'] etc. (we normalize later)
    """
    acrs = []
    for m in re.findall(r"\(([^)]+)\)", label):
        acrs.append(m)
    return acrs

class Matcher:
    def __init__(self, codeframes):
        self.codeframes = codeframes  # {qid: {label: (label, code)}}
        # Precompute normalized labels and lightweight auto-aliases
        self.norm_labels = {}
        self.auto_aliases = {}  # {qid: {alias_norm: canonical_label}}
        for qid, labels in self.codeframes.items():
            self.norm_labels[qid] = {}
            self.auto_aliases[qid] = {}
            for lbl in labels.keys():
                nlbl = normalize(lbl)
                self.norm_labels[qid][lbl] = nlbl

                # base label without parentheticals
                base = re.sub(r"\s*\([^)]*\)", "", lbl).strip()
                nbase = normalize(base)

                # compact forms
                clbl = compact(nlbl)
                cbase = compact(nbase)

                # parenthetical acronyms like (VW)
                acrs = extract_parenthetical_acronyms(lbl)
                for a in acrs:
                    na = normalize(a)
                    ca = compact(na)
                    if na:
                        self.auto_aliases[qid].setdefault(na, lbl)
                    if ca:
                        self.auto_aliases[qid].setdefault(ca, lbl)

                # add base and compact variants as aliases too
                if nbase:
                    self.auto_aliases[qid].setdefault(nbase, lbl)
                if cbase:
                    self.auto_aliases[qid].setdefault(cbase, lbl)
                if clbl:
                    self.auto_aliases[qid].setdefault(clbl, lbl)

    def match(self, qid: str, verbatim: str):
        if qid not in self.codeframes:
            return None
        labels_map = self.codeframes[qid]          # {label: (label, code)}
        canon_labels = list(labels_map.keys())
        if not canon_labels:
            return None

        norm_text = normalize(verbatim)
        ctext = compact(norm_text)

        # 0) auto-alias quick hits (exact or substring)
        aliases = self.auto_aliases.get(qid, {})
        # exact alias
        if norm_text in aliases:
            canonical = aliases[norm_text]
            _, code = labels_map[canonical]
            return (canonical, code, 0.99, "auto-alias", norm_text, [(canonical, 99)])
        if ctext in aliases:
            canonical = aliases[ctext]
            _, code = labels_map[canonical]
            return (canonical, code, 0.99, "auto-alias", ctext, [(canonical, 99)])
        # alias substring containment
        for a, canonical in aliases.items():
            if a and (a in norm_text or a in ctext):
                _, code = labels_map[canonical]
                return (canonical, code, 0.96, "auto-alias-sub", a, [(canonical, 96)])

        # 1) bidirectional substring on canonical labels
        for label in canon_labels:
            nlbl = self.norm_labels[qid][label]
            if not nlbl:
                continue
            # label-in-text OR text-in-label, in both normal and compact spaces
            if (nlbl in norm_text or norm_text in nlbl or
                compact(nlbl) in ctext or ctext in compact(nlbl)):
                _, code = labels_map[label]
                return (label, code, 0.92, "substring", nlbl, [(label, 92)])

        # 2) fuzzy (slightly more permissive to catch 1-char typos like 'cadillc')
        def scorer(q, choice, *, score_cutoff=0):
    # Pass score_cutoff through to each metric so RapidFuzz won't error
    s1 = fuzz.token_set_ratio(q, choice, score_cutoff=score_cutoff)
    s2 = fuzz.partial_ratio(q, choice, score_cutoff=score_cutoff)
    # Weighted ensemble
    return 0.6 * s1 + 0.4 * s2


        choices_map = {label: self.norm_labels[qid][label] for label in canon_labels}
        extracted = process.extract(norm_text, choices_map, scorer=scorer, limit=3)

        if extracted:
            top_label, top_score, _ = extracted[0]
            second = extracted[1][1] if len(extracted) > 1 else 0
            # Relaxed thresholds a touch
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

# ------------------
# fastapi app
# ------------------
app = FastAPI(title="Autocoder API", version="1.0.1")

STATE = {"matcher": None}

def init():
    codeframes = load_codeframes("./codeframes")
    STATE["matcher"] = Matcher(codeframes)

init()

class AutocodeRequest(BaseModel):
    question_id: str
    verbatim: str  # brand text only is fine

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

# debug helper so you can see what loaded
@app.get("/debug/codeframes")
def debug_codeframes():
    cf = STATE["matcher"].codeframes if STATE.get("matcher") else {}
    return {qid: len(labels) for qid, labels in cf.items()}

@app.post("/autocode", response_model=AutocodeResponse)
def autocode(payload: AutocodeRequest):
    try:
        res = STATE["matcher"].match(payload.question_id, payload.verbatim)
    except Exception as e:
        # clear message instead of 500
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
