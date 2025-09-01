from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
from app.loader import load_codeframes
from app.match import Matcher

app = FastAPI(title="Autocoder API", version="1.0.0")

STATE = {"matcher": None}

def init():
    codeframes = load_codeframes("./codeframes")
    STATE["matcher"] = Matcher(codeframes)

init()

class AutocodeRequest(BaseModel):
    question_id: str  # e.g., "Q1a" or "f3" (file name without .csv)
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
