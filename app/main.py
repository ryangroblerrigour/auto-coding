from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import csv, os, re, unicodedata
from rapidfuzz import process, fuzz

# ------------------
# loader
# ------------------
def load_codeframes(path: str):
    out = {}
    for fname in os.listdir(path):
        if fname.lower().endswith(".csv"):
            qid = os.path.splitext(fname)[0]
            out[qid] = {}
            with open(os.path.join(path, fname), newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    label = row["label"].strip()
                    code = int(row["code"])
                    out[qid][label] = (label, code)
    return out

# ------------------
# matcher
# ------------------
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[\.\-_/]", " ", s)
    s = re.sub(r"\b&\b", " and ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compact(s: str) -> str:
    return re.sub(r"\s+", "", s)

class Matcher:
    def __init__(self, codeframes):
        self.codeframes = codeframes
        self.norm_labels = {
            qid: {lbl: normalize(lbl) for lbl in labels}
            for qid, labels in self.codeframes.items()
        }

    def match(self, qid: str, verbatim: str):
        if qid not in self.codeframes:
            return None
        labels_map = self.codeframes[qid]
        canon_labels = list(labels_map.keys())

        norm_text = normalize(verbatim)
        ctext = compact(norm_text)

        # 1) substring check
        for label in canon_labels:
            nlbl = self.norm_labels[qid][label]
            if nlbl in norm_text or compact(nlbl) in ctext:
                _, code = labels_map[label]
                return (label, code, 0.92, "substring", nlbl, [(label, 92)])

        # 2) fuzzy
        def scorer(q, choice):
            return 0.6 * fuzz.token_set_ratio(q, choice) + 0.4 * fuzz.partial_ratio(q, choice)

        extracted = process.extract(
            norm_text,
            {label: self.norm_labels[qid][label] for label in canon_labels},
            scorer=scorer,
            limit=3
        )
        if extracted:
            top_label, top_score, _ = extracted[0]
            second = extracted[1][1] if len(extracted) > 1 else 0
            if top_score >= 80 and (top_score - second) >= 10:
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
app = FastAPI(title="Autocoder API", version="1.0.0")

STATE = {"matcher": None}

def init():
    codeframes = load_codeframes("./codeframes")
    STATE["matcher"] = Matcher(codeframes)

init()

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/autocode", response_model=AutocodeResponse)
def autocode(payload: AutocodeRequest):
    res = STATE["matcher"].match(payload.question_id, payload.verbatim)
    if not res:
        return Aut
