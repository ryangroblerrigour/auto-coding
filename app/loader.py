import csv, os
from typing import Dict, Tuple

def load_codeframes(path: str) -> Dict[str, Dict[str, Tuple[str, int]]]:
    """
    Reads all CSVs in ./codeframes and returns:
      {question_id: {canonical_label: (canonical_label, code)}}
    where question_id is the CSV filename (without .csv).
    CSVs must have headers: label,code
    """
    out = {}
    for fname in os.listdir(path):
        if fname.lower().endswith(".csv"):
            qid = os.path.splitext(fname)[0]  # e.g., Q1a -> question_id "Q1a"
            out[qid] = {}
            with open(os.path.join(path, fname), newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    label = row["label"].strip()
                    code = int(row["code"])
                    out[qid][label] = (label, code)
    return out
