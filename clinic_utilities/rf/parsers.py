from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

@dataclass(frozen=True)
class MedicationCandidate:
    source_text: str
    category: str
    canonical_key: str
    drug_name: str
    dose: str
    duration: str
    auto_selected: bool = False
    def to_dict(self) -> dict:
        return asdict(self)

_MEDICATION_PATTERNS = (
    ("nsaid","ibuprofen","Ιβουπροφαίνη",("ibuprofen","ιβουπροφαινη","brufen","nurofen")),
    ("nsaid","diclofenac","Δικλοφενάκη",("diclofenac","δικλοφενακη","dicloduo","voltaren")),
    ("nsaid","naproxen","Ναπροξένη",("naproxen","ναπροξενη","naprosyn")),
    ("nsaid","etoricoxib","Etoricoxib",("etoricoxib","arcoxia")),
    ("nsaid","celecoxib","Celecoxib",("celecoxib","celebrex")),
    ("nsaid","meloxicam","Μελοξικάμη",("meloxicam","μελοξικαμη","mobic")),
    ("nsaid","aceclofenac","Aceclofenac",("aceclofenac","aertal")),
    ("nsaid","dexketoprofen","Dexketoprofen",("dexketoprofen","arveles")),
    ("nsaid","lornoxicam","Lornoxicam",("lornoxicam","xefo")),
    ("other","paracetamol","Παρακεταμόλη",("paracetamol","παρακεταμολ","panadol","depon")),
    ("other","parcoten","Parcoten",("parcoten",)),
    ("other","tramadol","Tramadol",("tramadol","τραμαδολ","tramadex","mabron")),
    ("other","codeine","Codeine",("codeine","κωδεινη")),
    ("other","tapentadol","Tapentadol",("tapentadol","palexia")),
    ("other","pregabalin","Pregabalin",("pregabalin","πρεγκαμπαλινη","lyrica")),
    ("other","gabapentin","Gabapentin",("gabapentin","γκαμπαπεντινη","neurontin")),
    ("other","duloxetine","Duloxetine",("duloxetine","ντουλοξετινη","cymbalta")),
)
_DOSE_RE = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|µg)(?!\w)", re.I)
_DURATION_RE = re.compile(r"(?:για\s*)?(\d+)\s*(ημερ(?:α|ες|ών)?|days?|εβδοµ(?:αδα|αδες|άδα|άδες)?|εβδομ(?:αδα|αδες|άδα|άδες)?|weeks?|μην(?:α|ες|ών)?|µην(?:α|ες|ών)?|months?)(?:\b|$)", re.I)

def _norm(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(value or "").casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))

def _entries(text: str) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in re.split(r"[\n;]+", str(text or "")) if part.strip()))

def _match(entry: str):
    normalized = _norm(entry)
    for category, key, display, aliases in _MEDICATION_PATTERNS:
        if any(_norm(alias) in normalized for alias in aliases):
            return category, key, display
    return None

def _dose(entry: str) -> str:
    match = _DOSE_RE.search(entry)
    return match.group(1).replace(",", ".") + " " + match.group(2) if match else ""

def _duration(entry: str) -> str:
    match = _DURATION_RE.search(entry)
    return f"{match.group(1)} {match.group(2)}" if match else ""

def parse_medications(text: str) -> dict:
    candidates=[]
    for entry in _entries(text):
        matched=_match(entry)
        if not matched: continue
        category,key,display=matched
        candidates.append(MedicationCandidate(entry,category,key,display,_dose(entry),_duration(entry)))
    best={}
    for idx,candidate in enumerate(candidates):
        score=(1 if candidate.dose else 0)+(2 if candidate.duration else 0)
        key=(candidate.category,candidate.canonical_key)
        current=best.get(key)
        current_score=(-1 if current is None else (1 if current[1].dose else 0)+(2 if current[1].duration else 0))
        if current is None or score>current_score: best[key]=(idx,candidate)
    deduped=[item for _,item in sorted(best.values(),key=lambda pair:pair[0])]
    grouped={"nsaid":[],"other":[]}
    for item in deduped: grouped[item.category].append(item)
    def ranked(items: Iterable[MedicationCandidate]):
        items=list(items); indexed=list(enumerate(items)); indexed.sort(key=lambda p:(-(1 if p[1].duration else 0),-(1 if p[1].dose else 0),p[0])); chosen={id(c) for _,c in indexed[:3]}
        return [MedicationCandidate(**{**c.to_dict(),"auto_selected":id(c) in chosen}) for c in items]
    nsaids=ranked(grouped["nsaid"]); others=ranked(grouped["other"])
    return {"nsaid_candidates":[x.to_dict() for x in nsaids],"other_candidates":[x.to_dict() for x in others],"auto_selected_nsaids":[x.to_dict() for x in nsaids if x.auto_selected][:3],"auto_selected_others":[x.to_dict() for x in others if x.auto_selected][:3]}

_DATE_PATTERNS=(re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)"),re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)"))
_SHORT_DATE_RE=re.compile(r"(?<!\d)\d{1,2}[./-]\d{1,2}(?![./-]\d)")
def parse_physio_dates(text: str) -> dict:
    raw=str(text or ""); found=set(); invalid=[]; spans=[]
    for pidx,pattern in enumerate(_DATE_PATTERNS):
        for match in pattern.finditer(raw):
            try:
                if pidx==0: year,month,day=map(int,match.groups())
                else: day,month,year=map(int,match.groups())
                found.add(date(year,month,day)); spans.append(match.span())
            except ValueError: invalid.append(match.group(0))
    for match in _SHORT_DATE_RE.finditer(raw):
        if not any(start<=match.start() and match.end()<=end for start,end in spans): invalid.append(match.group(0))
    values=sorted(found)
    return {"dates":[v.isoformat() for v in values],"start_date":values[0].isoformat() if values else "","end_date":values[-1].isoformat() if values else "","treatment_count":len(values),"invalid_or_ambiguous_tokens":list(dict.fromkeys(invalid))}
