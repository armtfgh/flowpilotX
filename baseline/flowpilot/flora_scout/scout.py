"""FLORA-Scout — Intelligent literature mining.

Cross-session memory:
  - Reads DOIs from paper_miner_history.json (legacy miner)
  - Reads DOIs from extraction_results/*.json (PRISM extracted papers)
  - Reads DOIs from flora_translate/data/records/*.json (indexed papers)
  All known DOIs are loaded at startup so Scout never re-suggests
  a paper you already have.


Architecture:
  1. QueryPlanner (Claude Sonnet)
     Converts free-text research description into:
     - Multiple search queries (different angles of the same topic)
     - Required terms (at least one must appear in title/abstract)
     - Exclusion terms (discard if present)
     - A relevance description used for per-paper scoring

  2. OpenAlexSearcher
     Executes queries, reconstructs abstracts from inverted index,
     deduplicates by DOI across all queries and across sessions.

  3. RelevanceFilter (Claude Haiku — cheap, batch)
     For each batch of papers, scores relevance 0-10 against the
     user's research description. Papers below threshold are discarded
     with a recorded reason.

  4. IterativeScout (orchestrator)
     Keeps searching until n_target relevant papers are found OR
     max_rounds exhausted. Each round generates fresh queries based
     on what has been found so far (avoiding already-seen DOIs).
"""

import json
import logging
import re
import time
from typing import Callable

import anthropic

logger = logging.getLogger("flora.scout")

# ── Models ──────────────────────────────────────────────────────────────────
MODEL_PLANNER  = "claude-sonnet-4-20250514"   # query planning
MODEL_FILTER   = "claude-haiku-4-5-20251001"  # relevance scoring (cheap)

# ── Relevance threshold (0-10 from Haiku) ───────────────────────────────────
RELEVANCE_THRESHOLD = 6   # papers below this score are discarded
FILTER_BATCH_SIZE   = 8   # papers per Haiku call
MAX_SEARCH_ROUNDS   = 6   # max iterations before giving up


def _get_client():
    return anthropic.Anthropic()


def _parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for sc, ec in [("{", "}"), ("[", "]")]:
            s, e = text.find(sc), text.rfind(ec)
            if s != -1 and e != -1:
                return json.loads(text[s:e+1])
        raise


def _load_known_dois() -> set[str]:
    """Load all DOIs already in the project so Scout never re-suggests them.

    Sources checked (in priority order):
    1. paper_miner_history.json  — legacy miner history
    2. extraction_results/*.json — PRISM-extracted papers
    3. flora_translate/data/records/*.json — indexed records
    """
    from pathlib import Path

    known: set[str] = set()

    def _normalise(doi: str) -> str:
        return (doi or "").lower().strip().replace("https://doi.org/", "")

    # 1. Legacy miner history
    history_path = Path("paper_miner_history.json")
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text())
            for doi in data.get("seen_dois", []):
                known.add(_normalise(doi))
        except Exception:
            pass

    # 2. Extraction results
    for p in Path("extraction_results").glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text())
            doi = rec.get("doi") or rec.get("source_pdf", "")
            if doi:
                known.add(_normalise(doi))
        except Exception:
            pass

    # 3. Indexed records
    for p in Path("flora_translate/data/records").glob("*.json"):
        try:
            rec = json.loads(p.read_text())
            doi = rec.get("doi", "")
            if doi:
                known.add(_normalise(doi))
        except Exception:
            pass

    if known:
        logger.info(f"Scout: loaded {len(known)} known DOIs from project history")
    return known


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    words = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words.keys()))


# ── Query Planner ────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are a scientific literature search expert specializing in chemistry research.
Your task: convert a research description into a precise, multi-angle search strategy
that will find ONLY papers matching the described research.

Return JSON ONLY. No prose."""

PLANNER_USER = """\
Research description:
"{description}"

Generate a search strategy. Return JSON:
{{
  "primary_queries": [
    "query 1 — most direct representation of the topic",
    "query 2 — alternative angle or terminology",
    "query 3 — more specific sub-aspect",
    "query 4 — related methodology",
    "query 5 — related application"
  ],
  "required_terms": [
    "term that MUST appear in relevant papers (at least one from this list)",
    "another required term"
  ],
  "exclusion_terms": [
    "term that disqualifies a paper if present",
    "another exclusion term"
  ],
  "relevance_description": "A 2-3 sentence description of what a RELEVANT paper \
looks like and what a NOT-RELEVANT paper looks like. Be specific about what \
distinguishes truly relevant papers from superficially similar ones.",
  "domain_constraints": "One sentence on the specific domain/field (e.g. \
'Only papers describing flow chemistry experiments in organic synthesis, \
not reviews, not biological applications')."
}}

Rules for required_terms: pick terms that are SPECIFIC to the described research.
Rules for exclusion_terms: pick terms that indicate off-topic areas.
Example: if topic is 'flow chemistry for organic synthesis', exclusion_terms might
include ['biological', 'microfluidic diagnostics', 'blood flow', 'battery', 'fuel cell'].
"""

FOLLOWUP_PLANNER_USER = """\
Research description:
"{description}"

So far we have found {n_found}/{n_target} relevant papers.
Papers found so far (titles):
{found_titles}

Papers searched but rejected (sample):
{rejected_sample}

Generate 4-5 NEW search queries (different from previous ones) to find more papers.
Return JSON:
{{
  "new_queries": [
    "new query 1",
    "new query 2",
    "new query 3",
    "new query 4"
  ],
  "reasoning": "Why these new queries might find different papers"
}}
"""


class QueryPlanner:
    """Generate search queries from a free-text research description."""

    def plan(self, description: str) -> dict:
        """Returns a search plan dict."""
        resp = _get_client().messages.create(
            model=MODEL_PLANNER,
            max_tokens=1024,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content":
                       PLANNER_USER.format(description=description)}],
        )
        plan = _parse_json(resp.content[0].text)
        logger.info(f"  Query plan: {len(plan.get('primary_queries',[]))} queries, "
                    f"{len(plan.get('required_terms',[]))} required terms, "
                    f"{len(plan.get('exclusion_terms',[]))} exclusions")
        return plan

    def followup(self, description: str, n_found: int, n_target: int,
                 found_papers: list[dict], rejected_papers: list[dict]) -> list[str]:
        """Generate follow-up queries when initial searches didn't reach target."""
        found_titles = "\n".join(f"- {p['title'][:80]}" for p in found_papers[:10])
        rejected_sample = "\n".join(
            f"- {p['title'][:60]} (rejected: {p.get('reject_reason','')})"
            for p in rejected_papers[:5]
        )
        prompt = FOLLOWUP_PLANNER_USER.format(
            description=description, n_found=n_found, n_target=n_target,
            found_titles=found_titles, rejected_sample=rejected_sample,
        )
        resp = _get_client().messages.create(
            model=MODEL_PLANNER, max_tokens=512,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _parse_json(resp.content[0].text)
        new_q = result.get("new_queries", [])
        logger.info(f"  Follow-up queries: {new_q}")
        return new_q


# ── Relevance Filter ─────────────────────────────────────────────────────────

FILTER_SYSTEM = """\
You are a scientific paper relevance evaluator. You will receive a research
description and a batch of papers (title + abstract). For each paper, output
a relevance score 0-10 and a one-line reason.

Return ONLY a JSON array, one object per paper, in the same order:
[
  {"index": 0, "score": 8, "reason": "Directly describes photoredox flow synthesis"},
  {"index": 1, "score": 2, "reason": "About microfluidic diagnostics, not synthesis"}
]

Scoring guide:
9-10: Directly reports the described research (exact match)
7-8:  Closely related, same methodology or application area
5-6:  Loosely related, shares some keywords but different focus
3-4:  Superficially similar topic but different domain
0-2:  Unrelated despite keyword overlap"""

FILTER_USER = """\
Research description:
{relevance_description}

Papers to evaluate:
{papers_text}

Score each paper 0-10 for relevance to the research description.
Return JSON array."""


class RelevanceFilter:
    """Score paper relevance using Haiku (cheap, batch processing)."""

    def filter_batch(self, papers: list[dict], relevance_description: str,
                     threshold: int = RELEVANCE_THRESHOLD) -> tuple[list[dict], list[dict]]:
        """Filter a batch of papers. Returns (accepted, rejected)."""
        if not papers:
            return [], []

        papers_text = "\n\n".join(
            f"[{i}] Title: {p.get('title','')}\nAbstract: {p.get('abstract','')[:400]}"
            for i, p in enumerate(papers)
        )

        try:
            resp = _get_client().messages.create(
                model=MODEL_FILTER,
                max_tokens=1024,
                system=FILTER_SYSTEM,
                messages=[{"role": "user", "content":
                           FILTER_USER.format(
                               relevance_description=relevance_description,
                               papers_text=papers_text,
                           )}],
            )
            scores = _parse_json(resp.content[0].text)
            if not isinstance(scores, list):
                scores = []
        except Exception as e:
            logger.warning(f"  Relevance filter failed: {e} — accepting all")
            return papers, []

        accepted, rejected = [], []
        score_map = {s.get("index", i): s for i, s in enumerate(scores)}

        for i, paper in enumerate(papers):
            info = score_map.get(i, {})
            score = int(info.get("score", 5))
            reason = info.get("reason", "")
            paper["relevance_score"] = score
            paper["relevance_reason"] = reason
            if score >= threshold:
                accepted.append(paper)
            else:
                paper["reject_reason"] = reason
                rejected.append(paper)

        return accepted, rejected


# ── OpenAlex Searcher ────────────────────────────────────────────────────────

class OpenAlexSearcher:
    """Execute searches on OpenAlex with cross-session deduplication."""

    def __init__(self, year_from: int = 2015):
        self.year_from = year_from
        # Pre-populate with all DOIs already in the project
        self.seen_dois: set[str] = _load_known_dois()

    def search(self, query: str, per_page: int = 50,
               required_terms: list[str] | None = None,
               exclusion_terms: list[str] | None = None) -> list[dict]:
        """Search OpenAlex for a query. Returns new (unseen) papers."""
        import pyalex

        try:
            works = (
                pyalex.Works()
                .search(query)
                .filter(has_abstract=True,
                        from_publication_date=f"{self.year_from}-01-01")
                .sort(cited_by_count="desc")
                .get(per_page=min(per_page, 200))
            )
        except Exception as e:
            logger.warning(f"  OpenAlex query failed for '{query}': {e}")
            return []

        results = []
        for w in works:
            doi = w.get("doi", "") or ""
            if doi in self.seen_dois:
                continue

            title = (w.get("title") or "").lower()
            abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
            combined = title + " " + abstract.lower()

            # Hard exclusion filter
            if exclusion_terms:
                if any(t.lower() in combined for t in exclusion_terms):
                    continue

            # Soft required filter — at least one required term should appear
            if required_terms:
                if not any(t.lower() in combined for t in required_terms):
                    continue

            self.seen_dois.add(doi)
            results.append({
                "doi": doi,
                "title": w.get("title", ""),
                "year": w.get("publication_year", 0),
                "n_citations": w.get("cited_by_count", 0),
                "abstract": abstract[:600],
                "source": "openalex",
            })

        return results


# ── Scout (main class) ───────────────────────────────────────────────────────

class Scout:
    """Intelligent literature mining with LLM query planning and relevance filtering.

    Input: free-text research description (not just keywords)
    Output: papers that genuinely match the described research

    Pipeline:
      1. QueryPlanner → search queries + exclusions + relevance criteria
      2. OpenAlexSearcher → raw candidate papers (deduplicated)
      3. RelevanceFilter → discard off-topic papers
      4. Repeat until n_target reached or max_rounds exhausted
    """

    def __init__(
        self,
        description: str = "",
        n_target: int = 50,
        year_from: int = 2015,
        snowball: bool = False,
        progress_callback: Callable | None = None,
        # backward-compat alias
        topic: str = "",
    ):
        # Accept both 'description' and legacy 'topic' kwarg
        self.description = description or topic
        self.n_target    = n_target
        self.year_from   = year_from
        self.snowball    = snowball
        self._cb         = progress_callback or (lambda p, m: None)

    def run(self) -> list[dict]:
        """Run the full mining pipeline. Returns accepted papers."""
        self._cb(0.02, "Planning search strategy with LLM...")

        # Step 1: Plan
        planner  = QueryPlanner()
        plan     = planner.plan(self.description)
        queries  = plan.get("primary_queries", [self.description])
        required = plan.get("required_terms", [])
        excluded = plan.get("exclusion_terms", [])
        rel_desc = plan.get("relevance_description", self.description)

        logger.info(f"Scout: target={self.n_target}, queries={len(queries)}")
        logger.info(f"  Required: {required}")
        logger.info(f"  Excluded: {excluded}")

        searcher = OpenAlexSearcher(year_from=self.year_from)
        filt     = RelevanceFilter()

        accepted: list[dict] = []
        rejected: list[dict] = []
        all_queries_used: list[str] = []

        # Step 2: Iterative search rounds
        for round_num in range(MAX_SEARCH_ROUNDS):
            if len(accepted) >= self.n_target:
                break

            # Which queries to use this round
            if round_num == 0:
                round_queries = queries
            else:
                self._cb(
                    0.1 + 0.7 * len(accepted) / max(self.n_target, 1),
                    f"Round {round_num+1}: generating follow-up queries "
                    f"({len(accepted)}/{self.n_target} found so far)..."
                )
                round_queries = planner.followup(
                    self.description, len(accepted), self.n_target,
                    accepted, rejected
                )

            if not round_queries:
                break

            all_queries_used.extend(round_queries)

            # Execute each query in this round
            for qi, query in enumerate(round_queries):
                if len(accepted) >= self.n_target:
                    break

                progress = 0.1 + 0.7 * (len(accepted) / max(self.n_target, 1))
                self._cb(min(progress, 0.85),
                         f"Searching: \"{query[:60]}...\"")
                logger.info(f"  Query [{round_num+1}/{qi+1}]: {query}")

                # Search OpenAlex — request more than needed to account for filtering
                fetch_n = min(
                    max(self.n_target * 3, 100),   # fetch 3x target for filtering margin
                    200,
                )
                candidates = searcher.search(
                    query, per_page=fetch_n,
                    required_terms=required,
                    exclusion_terms=excluded,
                )
                logger.info(f"    Candidates after hard filter: {len(candidates)}")

                if not candidates:
                    continue

                # Batch relevance filter
                for batch_start in range(0, len(candidates), FILTER_BATCH_SIZE):
                    batch = candidates[batch_start:batch_start + FILTER_BATCH_SIZE]
                    acc_batch, rej_batch = filt.filter_batch(batch, rel_desc)
                    accepted.extend(acc_batch)
                    rejected.extend(rej_batch)
                    logger.info(f"    Batch: +{len(acc_batch)} accepted, "
                                f"{len(rej_batch)} rejected")

                    if len(accepted) >= self.n_target:
                        break

            logger.info(f"  End of round {round_num+1}: "
                        f"{len(accepted)}/{self.n_target} accepted")

        # Step 3: Citation snowballing (top papers → fetch their references)
        if self.snowball and accepted:
            self._cb(0.88, "Citation snowballing...")
            accepted = self._snowball(accepted, searcher, filt, rel_desc,
                                      required, excluded)

        # Trim to target, sort by relevance score then citations
        accepted.sort(key=lambda p: (
            -p.get("relevance_score", 5),
            -p.get("n_citations", 0),
        ))
        accepted = accepted[:self.n_target]

        # Add final score for display (0.0-1.0)
        for i, p in enumerate(accepted):
            p["score"] = round(p.get("relevance_score", 5) / 10.0, 2)

        n_rejected_total = len(rejected)
        n_searched = len(searcher.seen_dois)
        self._cb(1.0,
                 f"Done — {len(accepted)} relevant papers "
                 f"(searched {n_searched}, rejected {n_rejected_total})")

        logger.info(f"Scout complete: {len(accepted)} accepted / "
                    f"{n_searched} searched / {n_rejected_total} rejected")
        return accepted

    def _snowball(self, accepted: list[dict], searcher: OpenAlexSearcher,
                  filt: RelevanceFilter, rel_desc: str,
                  required: list[str], excluded: list[str]) -> list[dict]:
        """Fetch references of top papers and filter them."""
        import pyalex

        top_dois = [p["doi"] for p in accepted[:5] if p.get("doi")]
        new_candidates = []

        for doi in top_dois:
            try:
                # Get this paper's referenced works
                clean_doi = doi.replace("https://doi.org/", "")
                works = (
                    pyalex.Works()
                    .filter(cites=clean_doi, has_abstract=True)
                    .get(per_page=20)
                )
                for w in works:
                    w_doi = w.get("doi", "") or ""
                    if w_doi in searcher.seen_dois:
                        continue
                    abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
                    combined = (w.get("title","") + " " + abstract).lower()
                    if excluded and any(t.lower() in combined for t in excluded):
                        continue
                    searcher.seen_dois.add(w_doi)
                    new_candidates.append({
                        "doi": w_doi,
                        "title": w.get("title",""),
                        "year": w.get("publication_year", 0),
                        "n_citations": w.get("cited_by_count", 0),
                        "abstract": abstract[:600],
                        "source": "snowball",
                    })
            except Exception as e:
                logger.debug(f"Snowball failed for {doi}: {e}")

        if not new_candidates:
            return accepted

        logger.info(f"  Snowball: {len(new_candidates)} new candidates")
        for batch_start in range(0, len(new_candidates), FILTER_BATCH_SIZE):
            batch = new_candidates[batch_start:batch_start + FILTER_BATCH_SIZE]
            acc_batch, _ = filt.filter_batch(batch, rel_desc)
            accepted.extend(acc_batch)

        return accepted
