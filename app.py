"""
PV Literature Screening - Synovia Pharma pipeline (Streamlit, self-contained).
Five stages only: Ingestion, Screening, Extraction, Prioritization, Signals.
No audit logging, no database. Lightweight screening model for free-tier hosting.
Run with: streamlit run app.py
"""
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Synovia PV Literature Screening", layout="wide")

# --- Synovia Pharma search strings ---------------------------------------
SAFETY_TERMS = ["adverse event", "adverse reaction", "side effect", "safety"]

SYNOVIA_SEARCH_STRINGS = {
    "Anti-Cancer": ["oxaliplatin", "capecitabine", "erlotinib", "docetaxel",
                     "anti-thymocyte globulin", "paclitaxel"],
    "Anti-Diabetic": ["glimepiride", "insulin glulisine", "ertugliflozin", "insulin glargine",
                       "sitagliptin", "metformin", "tirzepatide"],
    "Anti-Infective": ["piperacillin tazobactam", "meropenem", "faropenem",
                        "amoxycillin and clavulanic acid", "amoxycillin trihydrate",
                        "metronidazole", "phenoxymethyl penicillin potassium", "cephradine",
                        "teicoplanin", "azithromycin", "cefixime", "cefuroxime axetil"],
    "Cardiac": ["apixaban", "atenolol", "furosemide", "enoxaparin sodium", "rosuvastatin",
                 "sacubitril", "atorvastatin", "pentoxifylline", "ramipril"],
    "CNS": ["sodium valproate", "clobazam", "levetiracetam", "risperidone",
             "prochlorperazine maleate", "vortioxetine hydrobromide", "escitalopram oxalate",
             "zopiclone", "quetiapine"],
    "Dermatology": ["luliconazole", "econazole nitrate", "triamcinolone acetonide", "tapinarof"],
    "Gastro": ["hyoscine butylbromide", "bacillus clausii",
                "sodium alginate sodium bicarbonate and calcium carbonate", "vonoprazan",
                "rabeprazole", "domperidone", "omeprazole", "sevelamer carbonate"],
    "Musculoskeletal": ["ketoprofen", "mirogabalin"],
    "Respiratory": ["pheniramine maleate", "doxofylline", "montelukast",
                      "promethazine", "phenylephrine", "fexofenadine hydrochloride"],
    "Vaccines": ["diphtheria tetanus pertussis hepatitis b inactivated polio and hib vaccine",
                  "meningococcal vaccine", "inactivated influenza vaccine", "hepatitis a vaccine",
                  "diphtheria tetanus pertussis and inactivated polio vaccine",
                  "polysaccharide typhoid vaccine"],
}

NEGATION_WORDS = ["no ", "not ", "without ", "non-", "absence of ", "no evidence of "]

def _term_present(haystack, term, window=20):
    for m in re.finditer(re.escape(term), haystack):
        preceding = haystack[max(0, m.start() - window):m.start()]
        if not any(neg in preceding for neg in NEGATION_WORDS):
            return True
    return False

# --- BanglaJOL ingestion -------------------------------------------------
JOURNAL_WATCHLIST = {
    "BJP": "https://www.banglajol.info/index.php/BJP",
    "BJMS": "https://www.banglajol.info/index.php/BJMS",
    "BMJK": "https://banglajol.info/index.php/BMJK",
    "BJMP": "https://www.banglajol.info/index.php/BJMP",
}
OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "dc": "http://purl.org/dc/elements/1.1/",
          "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/"}

@dataclass
class Article:
    record_id: str; title: str; abstract: str; journal: str; journal_code: str
    pub_date: str; authors: list; matched_category: str; matched_drug: str
    source: str = "banglajol"

def _parse_oai_dc_record(record_el, journal_code, journal_name):
    header = record_el.find("oai:header", OAI_NS)
    if header is not None and header.get("status") == "deleted":
        return None
    identifier = header.findtext("oai:identifier", default="", namespaces=OAI_NS)
    metadata = record_el.find(".//oai_dc:dc", OAI_NS)
    if metadata is None:
        return None
    def get_all(tag):
        return [e.text for e in metadata.findall(f"dc:{tag}", OAI_NS) if e.text]
    titles, descriptions, creators, dates = get_all("title"), get_all("description"), get_all("creator"), get_all("date")
    return Article(identifier, titles[0] if titles else "", descriptions[0] if descriptions else "",
                    journal_name, journal_code, dates[0] if dates else "", creators, "", "")

def harvest_journal(journal_code, base_url, days_back=30):
    since = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {"verb": "ListRecords", "metadataPrefix": "oai_dc", "from": since}
    articles = []
    url = f"{base_url}/oai"
    while True:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        error = root.find(".//oai:error", OAI_NS)
        if error is not None:
            break
        for record in root.findall(".//oai:record", OAI_NS):
            a = _parse_oai_dc_record(record, journal_code, journal_code)
            if a:
                articles.append(a)
        token_el = root.find(".//oai:resumptionToken", OAI_NS)
        if token_el is not None and token_el.text:
            params = {"verb": "ListRecords", "resumptionToken": token_el.text}
            time.sleep(0.5)
        else:
            break
    return articles

def filter_by_search_strings(articles, search_strings, safety_terms):
    matched = []
    for a in articles:
        haystack = f"{a.title} {a.abstract}".lower()
        if not any(_term_present(haystack, t) for t in safety_terms):
            continue
        for category, drugs in search_strings.items():
            hit_drug = next((d for d in drugs if d.lower() in haystack), None)
            if hit_drug:
                a.matched_category = category
                a.matched_drug = hit_drug
                matched.append(a)
                break
    return matched

@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def run_banglajol_ingestion(search_strings, safety_terms, days_back, journals):
    """Cache each BanglaJOL query for 15 minutes to avoid repeated downloads."""
    all_articles = []
    for code_, base_url in journals.items():
        try:
            all_articles.extend(harvest_journal(code_, base_url, days_back))
        except requests.RequestException as e:
            st.warning(f"BanglaJOL {code_} failed: {e}")
    relevant = filter_by_search_strings(all_articles, search_strings, safety_terms)
    out = []
    for a in relevant:
        d = asdict(a)
        d["pmid"] = a.record_id
        d["query_term"] = a.matched_drug
        d["source"] = "banglajol"
        out.append(d)
    return out

# --- PubMed ingestion ------------------------------------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def build_pubmed_query(drugs, safety_terms, days_back):
    drug_clause = " OR ".join(f'"{d}"[Title/Abstract]' for d in drugs)
    safety_clause = " OR ".join(f'"{t}"[Title/Abstract]' for t in safety_terms)
    since = (date.today() - timedelta(days=days_back)).strftime("%Y/%m/%d")
    today = date.today().strftime("%Y/%m/%d")
    return (f'(({drug_clause}) AND ({safety_clause})) AND '
            f'("{since}"[Date - Publication] : "{today}"[Date - Publication])')

@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def run_pubmed_ingestion(search_strings, safety_terms, days_back=30):
    """Cache each PubMed query for 15 minutes to avoid repeated downloads."""
    seen, out = set(), []
    for category, drugs in search_strings.items():
        query = build_pubmed_query(drugs, safety_terms, days_back)
        try:
            resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi",
                                 params={"db": "pubmed", "term": query, "retmax": 200, "retmode": "json"}, timeout=30)
            resp.raise_for_status()
            pmids = resp.json()["esearchresult"].get("idlist", [])
            time.sleep(0.34)
            if not pmids:
                continue
            resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi",
                                 params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            time.sleep(0.34)
        except requests.RequestException as e:
            st.warning(f"PubMed failed for {category}: {e}")
            continue
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID", default="")
            if not pmid or pmid in seen:
                continue
            seen.add(pmid)
            title = art.findtext(".//ArticleTitle", default="")
            abstract = " ".join(el.text or "" for el in art.findall(".//AbstractText"))
            haystack = f"{title} {abstract}".lower()
            hit_drug = next((d for d in drugs if d.lower() in haystack), category)
            out.append({
                "pmid": pmid, "title": title, "abstract": abstract,
                "journal": art.findtext(".//Journal/Title", default=""), "journal_code": "PubMed",
                "pub_date": art.findtext(".//PubDate/Year", default=""),
                "authors": [f'{a.findtext("LastName","")} {a.findtext("Initials","")}'.strip()
                            for a in art.findall(".//Author") if a.find("LastName") is not None],
                "matched_category": category, "matched_drug": hit_drug,
                "query_term": hit_drug, "source": "pubmed",
            })
    return out

# --- Screening (lightweight local model) ---------------------------------
CANDIDATE_LABELS = ["reports a drug safety issue, adverse event, or manufacturing defect",
                     "does not relate to drug safety"]
@st.cache_resource(show_spinner=False)
def _get_classifier():
    """Load the lightweight classifier once and reuse it across reruns/users."""
    from transformers import pipeline
    import torch
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        device=0 if torch.cuda.is_available() else -1,
    )

MAX_CHARS_FOR_CLASSIFICATION = 1000  # truncate long abstracts for faster classification

def _build_text(article):
    text = f"{article.get('title','')}. {article.get('abstract') or ''}"
    return text[:MAX_CHARS_FOR_CLASSIFICATION]

def screen_batch(articles, confidence_threshold=0.6, batch_size=16, progress_callback=None):
    """Classifies in batches instead of one at a time - meaningfully faster
    than a sequential loop, which is what was slow before."""
    buckets = {"relevant": [], "not_relevant": [], "needs_review": []}
    classifier = _get_classifier()

    texts, indices = [], []
    for i, article in enumerate(articles):
        text = _build_text(article)
        if text.strip(". "):
            texts.append(text)
            indices.append(i)
        else:
            a = dict(article)
            a["screening"] = {"relevant": True, "confidence": 0.0, "reason": "no text to classify"}
            buckets["needs_review"].append(a)

    results_by_index = {}
    total = len(texts)
    done = 0
    for start in range(0, total, batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_results = classifier(batch_texts, CANDIDATE_LABELS, multi_label=False)
        if isinstance(batch_results, dict):
            batch_results = [batch_results]
        for offset, result in enumerate(batch_results):
            orig_idx = indices[start + offset]
            top_label, top_score = result["labels"][0], result["scores"][0]
            results_by_index[orig_idx] = {
                "relevant": top_label == CANDIDATE_LABELS[0],
                "confidence": round(top_score, 3),
                "reason": f"zero-shot top label: '{top_label}' (score={top_score:.2f})",
            }
        done = min(start + batch_size, total)
        if progress_callback:
            progress_callback(done, max(total, 1))

    for i, article in enumerate(articles):
        if i not in results_by_index:
            continue
        screening = results_by_index[i]
        a = dict(article)
        a["screening"] = screening
        if screening["confidence"] < confidence_threshold:
            buckets["needs_review"].append(a)
        elif screening["relevant"]:
            buckets["relevant"].append(a)
        else:
            buckets["not_relevant"].append(a)

    return buckets

# --- Extraction (rule-based) ----------------------------------------------
SEVERITY_KEYWORDS = {"fatal": ["fatal", "death", "died", "deceased", "mortality"],
                      "severe": ["severe", "serious", "life-threatening", "hospitalization", "hospitalisation"],
                      "moderate": ["moderate"], "mild": ["mild", "minor", "self-limiting"]}
CAUSALITY_KEYWORDS = ["definite", "probable", "possible", "unlikely", "unclassifiable",
                       "temporally associated", "causally related", "not established"]
MANUFACTURING_KEYWORDS = ["contamination", "recall", "batch defect", "lot defect", "impurity",
                           "formulation defect", "packaging defect", "counterfeit", "substandard",
                           "contaminated batch", "faulty batch", "manufacturing defect", "manufacturing error"]
DEMOGRAPHIC_PATTERN = re.compile(
    r"(\d{1,3}[-\s]?year[-\s]?old\s+\w+|\baged?\s+\d{1,3}(?:[-\s]?(?:to|-)\s?\d{1,3})?\s*(?:years?)?)", re.IGNORECASE)
CASE_COUNT_PATTERN = re.compile(r"(?:n\s*=\s*(\d+))|(\d+)\s+(?:cases?|patients?|subjects?)", re.IGNORECASE)

def _keyword_present(text_lower, keyword, window=20):
    for m in re.finditer(re.escape(keyword), text_lower):
        preceding = text_lower[max(0, m.start() - window):m.start()]
        if not any(neg in preceding for neg in NEGATION_WORDS):
            return True
    return False

def detect_severity(text):
    text_lower = text.lower()
    for level in ["fatal", "severe", "moderate", "mild"]:
        if any(_keyword_present(text_lower, kw) for kw in SEVERITY_KEYWORDS[level]):
            return level
    return "unknown"

def detect_causality(text):
    text_lower = text.lower()
    found = [kw for kw in CAUSALITY_KEYWORDS if _keyword_present(text_lower, kw)]
    return "; ".join(found) if found else "not stated"

def detect_manufacturing(text):
    text_lower = text.lower()
    found = [kw for kw in MANUFACTURING_KEYWORDS if _keyword_present(text_lower, kw)]
    return (len(found) > 0), ("; ".join(found) if found else "")

def extract_case(article):
    text = f"{article.get('title','')}. {article.get('abstract') or ''}"
    manufacturing_related, manufacturing_notes = detect_manufacturing(text)
    demo_match = DEMOGRAPHIC_PATTERN.search(text)
    count_match = CASE_COUNT_PATTERN.search(text)
    return {
        "pmid": article.get("pmid", ""), "drug_name": article.get("matched_drug", ""),
        "category": article.get("matched_category", ""),
        "adverse_event_term": "not auto-extracted - see title/abstract",
        "severity": detect_severity(text),
        "patient_demographics": demo_match.group(0) if demo_match else "not reported",
        "causality_language": detect_causality(text),
        "manufacturing_related": manufacturing_related, "manufacturing_notes": manufacturing_notes,
        "case_count": int(count_match.group(1) or count_match.group(2)) if count_match else None,
    }

def extract_batch(articles):
    return [extract_case(a) for a in articles]

# --- Prioritization ------------------------------------------------------
SEVERITY_WEIGHTS = {"fatal": 100, "severe": 60, "moderate": 30, "mild": 10, "unknown": 15}
SPECIAL_POPULATION_TERMS = ["pregnan", "paediatric", "pediatric", "neonat", "infant", "elderly",
                             "geriatric", "renal impairment", "hepatic impairment", "breastfeeding"]
MEDICATION_ERROR_TERMS = ["medication error", "dosing error", "wrong dose", "overdose",
                           "look-alike", "sound-alike", "mix-up", "administration error"]
UNEXPECTED_REACTION_TERMS = ["unexpected", "novel", "first report", "previously unreported", "rare"]

def score_case(c):
    score = 0.0
    reasons = []
    severity = (c.get("severity") or "unknown").lower()
    score += SEVERITY_WEIGHTS.get(severity, 15)
    reasons.append(f"severity={severity}")
    text_blob = " ".join([c.get("adverse_event_term", ""), c.get("patient_demographics", ""),
                           c.get("manufacturing_notes", "")]).lower()
    if any(t in text_blob for t in SPECIAL_POPULATION_TERMS):
        score += 25; reasons.append("special population")
    if any(t in text_blob for t in MEDICATION_ERROR_TERMS):
        score += 20; reasons.append("medication error")
    if any(t in text_blob for t in UNEXPECTED_REACTION_TERMS):
        score += 20; reasons.append("unexpected reaction")
    if c.get("manufacturing_related"):
        score += 30; reasons.append("manufacturing-related")
    case_count = c.get("case_count") or 1
    if case_count > 1:
        score += min(case_count * 2, 20); reasons.append(f"n={case_count}")
    return round(score, 1), reasons

def rank_review_queue(cases):
    ranked = []
    for c in cases:
        score, reasons = score_case(c)
        e = dict(c); e["priority_score"] = score; e["priority_reasons"] = "; ".join(reasons)
        ranked.append(e)
    ranked.sort(key=lambda c: c["priority_score"], reverse=True)
    return ranked

# --- Signal detection (PRR/ROR) ------------------------------------------
@dataclass
class ContingencyTable:
    a: int; b: int; c: int; d: int

def compute_prr(t):
    if t.a == 0:
        return 0.0, 0.0, 0.0
    a, b, c, d = t.a, t.b, t.c, t.d
    if b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    prr = (a / (a + b)) / (c / (c + d))
    se = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    lower_ci = math.exp(math.log(prr) - 1.96 * se)
    n = a + b + c + d
    expected_a = (a + b) * (a + c) / n if n else 0
    chi2 = ((a - expected_a) ** 2) / expected_a if expected_a else 0
    return prr, lower_ci, chi2

def compute_ror(t):
    if t.a == 0:
        return 0.0, 0.0
    a, b, c, d = t.a, t.b, t.c, t.d
    if b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    ror = (a * d) / (b * c)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    return ror, math.exp(math.log(ror) - 1.96 * se)

def evaluate_signal(t):
    prr, prr_lo, chi2 = compute_prr(t)
    ror, ror_lo = compute_ror(t)
    return {"prr": round(prr,2), "prr_lower_ci": round(prr_lo,2), "chi_squared": round(chi2,2),
            "ror": round(ror,2), "ror_lower_ci": round(ror_lo,2),
            "signal_prr": (prr >= 2 and chi2 >= 4 and t.a >= 3), "signal_ror": (ror_lo >= 1.0 and t.a >= 3)}

def proxy_event_category(c):
    if c.get("manufacturing_related"):
        return "manufacturing-related issue"
    sev = c.get("severity", "unknown")
    return "unspecified adverse event" if sev == "unknown" else f"{sev} adverse event"

def rank_signals(cases):
    counts = {}
    for c in cases:
        key = (c["drug_name"], proxy_event_category(c))
        counts[key] = counts.get(key, 0) + (c.get("case_count") or 1)
    drugs = {k[0] for k in counts}
    events = {k[1] for k in counts}
    total = sum(counts.values())
    results = []
    for drug in drugs:
        drug_total = sum(v for (d,e),v in counts.items() if d == drug)
        for event in events:
            a = counts.get((drug, event), 0)
            if a == 0:
                continue
            b = drug_total - a
            event_total = sum(v for (d,e),v in counts.items() if e == event)
            c_ = event_total - a
            d_ = total - drug_total - event_total + a
            stats = evaluate_signal(ContingencyTable(a, b, c_, d_))
            results.append({"drug": drug, "event": event, "num_reports": a, **stats})
    results.sort(key=lambda r: r["prr"], reverse=True)
    return results

# =========================================================================
# Streamlit UI
# =========================================================================
for key, default in [("filtered_articles", []), ("screening_buckets", None),
                      ("extracted_cases", []), ("ranked_cases", []), ("ranked_signals", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

st.sidebar.title("Synovia PV Literature Screening")
st.sidebar.caption("Search strings from Literature_Search_String_002 - no API key required")

selected_categories = st.sidebar.multiselect(
    "Therapeutic categories to search",
    options=list(SYNOVIA_SEARCH_STRINGS.keys()),
    default=list(SYNOVIA_SEARCH_STRINGS.keys()),
)
active_search_strings = {k: v for k, v in SYNOVIA_SEARCH_STRINGS.items() if k in selected_categories}

days_back = st.sidebar.slider("Days back to harvest", 7, 730, 365, 7)
selected_journals = st.sidebar.multiselect("BanglaJOL journals", list(JOURNAL_WATCHLIST.keys()),
                                            default=list(JOURNAL_WATCHLIST.keys()))
use_pubmed = st.sidebar.checkbox("Also harvest PubMed", value=True)
confidence_threshold = st.sidebar.slider("Screening confidence threshold", 0.0, 1.0, 0.6, 0.05)

with st.sidebar.expander("View active search strings"):
    for cat, drugs in active_search_strings.items():
        st.caption(f"**{cat}**: {' OR '.join(drugs)}")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Search & Run", "1. Ingestion", "2. Screening", "3. Extraction",
     "4. Prioritization", "5. Signal Detection"])

with tab0:
    st.header("Search & Run — full pipeline in one go")
    st.caption("Runs every stage automatically using the selected categories in the sidebar: "
               "harvest -> screen -> extract -> prioritize -> detect signals.")

    run_clicked = st.button("Search & run full pipeline", type="primary", use_container_width=True)

    if run_clicked:
        if not active_search_strings:
            st.error("Select at least one therapeutic category in the sidebar.")
        else:
            status = st.status("Running pipeline...", expanded=True)

            status.write("**Step 1/5 - Harvesting BanglaJOL + PubMed...**")
            journals = {k: JOURNAL_WATCHLIST[k] for k in selected_journals}
            banglajol = run_banglajol_ingestion(active_search_strings, SAFETY_TERMS, days_back, journals)
            pubmed = run_pubmed_ingestion(active_search_strings, SAFETY_TERMS, min(days_back, 90)) if use_pubmed else []
            seen, merged = set(), []
            for a in banglajol + pubmed:
                key = (a["title"].strip().lower(), a["pmid"])
                if key not in seen:
                    seen.add(key)
                    merged.append(a)
            st.session_state["filtered_articles"] = merged
            status.write(f"  BanglaJOL: {len(banglajol)} | PubMed: {len(pubmed)} | total after dedup: {len(merged)}")

            if not merged:
                status.update(label="No articles found - try more categories or a wider date range.",
                               state="error", expanded=True)
            else:
                status.write("**Step 2/5 - Screening for safety relevance (loading lightweight model)...**")
                buckets = screen_batch(merged, confidence_threshold)
                st.session_state["screening_buckets"] = buckets
                status.write(f"  Relevant: {len(buckets['relevant'])} | "
                              f"Needs review: {len(buckets['needs_review'])} | "
                              f"Not relevant: {len(buckets['not_relevant'])}")

                if not buckets["relevant"]:
                    status.update(label="No articles screened as relevant.", state="complete", expanded=True)
                else:
                    status.write("**Step 3/5 - Extracting structured fields...**")
                    cases = extract_batch(buckets["relevant"])
                    st.session_state["extracted_cases"] = cases
                    status.write(f"  Extracted {len(cases)} cases.")

                    status.write("**Step 4/5 - Ranking by priority...**")
                    ranked_cases = rank_review_queue(cases)
                    st.session_state["ranked_cases"] = ranked_cases
                    status.write(f"  Top priority: {ranked_cases[0]['drug_name']} "
                                  f"(score {ranked_cases[0]['priority_score']})" if ranked_cases else "  Nothing to rank.")

                    status.write("**Step 5/5 - Detecting cross-report signals (PRR/ROR)...**")
                    ranked_signals = rank_signals(cases)
                    st.session_state["ranked_signals"] = ranked_signals
                    status.write(f"  Ranked {len(ranked_signals)} drug-event pairs.")

                    status.update(label="Pipeline complete.", state="complete", expanded=False)

    if st.session_state["ranked_cases"]:
        st.subheader("Prioritized review queue")
        df = pd.DataFrame(st.session_state["ranked_cases"])
        st.dataframe(df, use_container_width=True)
        st.download_button("Download reviewer CSV", df.to_csv(index=False),
                            "pv_review_queue.csv", "text/csv", key="search_tab_download")

    if st.session_state["ranked_signals"]:
        st.subheader("Signals (PRR/ROR)")
        st.caption("Small samples: treat flagged pairs as worth attention, not a confirmed signal.")
        signals_df = pd.DataFrame(st.session_state["ranked_signals"])
        st.dataframe(signals_df, use_container_width=True)

with tab1:
    st.header("Step 1 - Harvest BanglaJOL + PubMed")
    if st.button("Run harvest", type="primary"):
        with st.spinner("Harvesting..."):
            journals = {k: JOURNAL_WATCHLIST[k] for k in selected_journals}
            banglajol = run_banglajol_ingestion(active_search_strings, SAFETY_TERMS, days_back, journals)
            pubmed = run_pubmed_ingestion(active_search_strings, SAFETY_TERMS, min(days_back, 90)) if use_pubmed else []
            seen, merged = set(), []
            for a in banglajol + pubmed:
                key = (a["title"].strip().lower(), a["pmid"])
                if key not in seen:
                    seen.add(key)
                    merged.append(a)
            st.session_state["filtered_articles"] = merged
        st.success(f"BanglaJOL: {len(banglajol)}, PubMed: {len(pubmed)}, total after dedup: {len(merged)}")
    if st.session_state["filtered_articles"]:
        st.dataframe(pd.DataFrame(st.session_state["filtered_articles"])[
            ["pmid", "title", "source", "matched_category", "matched_drug", "pub_date"]], use_container_width=True)

with tab2:
    st.header("Step 2 - AI relevance screening (lightweight model)")
    st.caption("MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 (142MB) - first run downloads the model.")
    if not st.session_state["filtered_articles"]:
        st.info("Run ingestion first.")
    elif st.button("Run screening", type="primary"):
        with st.spinner("Loading model and screening..."):
            progress = st.progress(0)
            buckets = screen_batch(
                st.session_state["filtered_articles"],
                confidence_threshold,
                progress_callback=lambda i, t: progress.progress(i / t),
            )
            st.session_state["screening_buckets"] = buckets
        st.success(f"Relevant: {len(buckets['relevant'])}, needs review: {len(buckets['needs_review'])}, "
                   f"not relevant: {len(buckets['not_relevant'])}")
    if st.session_state["screening_buckets"]:
        b = st.session_state["screening_buckets"]
        all_screened = b["relevant"] + b["needs_review"] + b["not_relevant"]
        if all_screened:
            st.dataframe(pd.DataFrame([{"pmid": a["pmid"], "title": a["title"], "relevant": a["screening"]["relevant"],
                                         "confidence": a["screening"]["confidence"], "reason": a["screening"]["reason"]}
                                        for a in all_screened]).sort_values("confidence", ascending=False),
                         use_container_width=True)

with tab3:
    st.header("Step 3 - Structured extraction (free, rule-based)")
    relevant = st.session_state["screening_buckets"]["relevant"] if st.session_state["screening_buckets"] else []
    if not relevant:
        st.info("Run screening first and confirm at least one relevant article.")
    elif st.button("Run extraction", type="primary"):
        cases = extract_batch(relevant)
        st.session_state["extracted_cases"] = cases
        st.success(f"Extracted {len(cases)} cases.")
    if st.session_state["extracted_cases"]:
        st.dataframe(pd.DataFrame(st.session_state["extracted_cases"]), use_container_width=True)

with tab4:
    st.header("Step 4 - Prioritization")
    if not st.session_state["extracted_cases"]:
        st.info("Run extraction first.")
    elif st.button("Rank review queue", type="primary"):
        ranked = rank_review_queue(st.session_state["extracted_cases"])
        st.session_state["ranked_cases"] = ranked
        st.success(f"Ranked {len(ranked)} cases.")
    if st.session_state["ranked_cases"]:
        df = pd.DataFrame(st.session_state["ranked_cases"])
        st.dataframe(df, use_container_width=True)
        st.download_button("Download reviewer CSV", df.to_csv(index=False), "pv_review_queue.csv", "text/csv")

with tab5:
    st.header("Step 5 - Signal detection (PRR/ROR)")
    st.caption("Small samples: treat flagged pairs as worth attention, not confirmed signals.")
    if not st.session_state["extracted_cases"]:
        st.info("Run extraction first.")
    elif st.button("Run signal detection", type="primary"):
        st.session_state["ranked_signals"] = rank_signals(st.session_state["extracted_cases"])
        st.success(f"Ranked {len(st.session_state['ranked_signals'])} drug-event pairs.")
    if st.session_state["ranked_signals"]:
        st.dataframe(pd.DataFrame(st.session_state["ranked_signals"]), use_container_width=True)
