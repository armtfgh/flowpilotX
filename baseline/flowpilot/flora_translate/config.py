"""FLORA-Translate — Configuration and constants."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RECORDS_DIR = DATA_DIR / "records"
CHROMA_DIR = DATA_DIR / "chroma_db"
PROMPTS_DIR = BASE_DIR / "prompts"
LAB_INVENTORY_PATH = DATA_DIR / "lab_inventory.json"

# ---------------------------------------------------------------------------
# Base model aliases  (change these to remap multiple components at once)
# ---------------------------------------------------------------------------

_SONNET  = "claude-sonnet-4-6"
_OPUS    = "claude-opus-4-6"

# kept for backward compatibility — other modules still import these
TRANSLATION_MODEL    = _SONNET
SUMMARY_MODEL        = _SONNET
CHEMISTRY_MODEL      = _OPUS
CHEMISTRY_MAX_TOKENS = 16384         # complete frontier-model chemistry schema without a duplicate retry
EMBEDDING_MODEL      = "text-embedding-3-small"
EMBEDDING_DIM        = 1536

# ---------------------------------------------------------------------------
# Per-component model selection  ← edit here to change individual components
# ---------------------------------------------------------------------------
# Each line controls exactly one part of the pipeline.
# All are Anthropic models by default; swap to any Claude model string.
#
# Available Anthropic models:
#   claude-opus-4-6            (most capable, slowest, most expensive)
#   claude-sonnet-4-6          (balanced - default for most components)
#   claude-haiku-4-5-20251001  (fast + cheap — good for parsing / classification)
# ---------------------------------------------------------------------------

MODEL_INPUT_PARSER       = _SONNET   # parses free-text batch protocol into BatchRecord
MODEL_CHEMISTRY_AGENT    = _OPUS     # Layer 1 chemistry reasoning (Opus recommended)
MODEL_TRANSLATION        = _SONNET   # Layer 2 flow proposal generation
MODEL_OUTPUT_FORMATTER   = _SONNET   # generates human-readable explanation
MODEL_REVISION_AGENT     = _SONNET   # targeted patch to an existing design
MODEL_CONVERSATION_AGENT = _SONNET   # chat intent classifier (TRANSLATE/REVISE/ANSWER/ASK)
MODEL_EMBEDDING_SUMMARY  = _SONNET   # LLM summary written before ChromaDB indexing
MODEL_TOPOLOGY_POLISHER  = "claude-haiku-4-5-20251001"  # lightweight topology review

# ---------------------------------------------------------------------------
# Lightweight upstream mode for weak/local models
# ---------------------------------------------------------------------------
# The main upstream path remains the default. A separate lightweight chemistry
# planning path can be enabled for weaker or local models that struggle with the
# full one-shot ChemistryPlan schema.
#
# Modes:
#   "auto"   → use lightweight path only for models matching weak/local markers
#   "always" → force lightweight upstream path
#   "v2"     → force fair lightweight path: compact JSON + evidence-backed
#              deterministic extraction; no kinetic/design anchor overwrite
#   "never"  → force full upstream path
# ---------------------------------------------------------------------------

LIGHTWEIGHT_UPSTREAM_MODE = "auto"
LIGHTWEIGHT_CHEMISTRY_MAX_TOKENS = 3072
LIGHTWEIGHT_LOCAL_INPUT_PARSER_MAX_TOKENS = 1536
LIGHTWEIGHT_UPSTREAM_WEAK_MODEL_MARKERS = (
    "mini",
    "haiku",
    "gemma",
    "llama",
    "mistral",
    "qwen",
    "local",
)

# ---------------------------------------------------------------------------
# Flow translation policy
# ---------------------------------------------------------------------------
# `evidence_first` treats predicted intensification as a screening hypothesis,
# not a hard feasibility requirement. It preserves conservative candidates and
# lets measured campaign evidence set residence-time floors. `intensify` keeps
# the former hard residence-time and process-value gates for explicitly
# requested intensification studies. `neutral` applies neither preference.
# ---------------------------------------------------------------------------

FLOW_TRANSLATION_POLICY = "evidence_first"  # "evidence_first" | "intensify" | "neutral"
FLOW_MAX_TAU_TO_BATCH_RATIO = 1.0

# ---------------------------------------------------------------------------
# Flow-value / process-intensification policy
# ---------------------------------------------------------------------------
# These parameters make "better than batch" an explicit design objective rather
# than only a hard feasibility ceiling.
# ---------------------------------------------------------------------------

BATCH_PROXIMITY_THRESHOLD = 0.85
POOL_REJECTION_THRESHOLD = 0.50
PVS_THRESHOLD = 0.35
MAX_WEAK_POOL_CYCLES = 2

PVS_WEIGHT_SELECTIVITY = 0.20
PVS_WEIGHT_TAU = 0.35
PVS_WEIGHT_SAFETY = 0.25
PVS_WEIGHT_HEAT = 0.20

CHIEF_SCORE_WEIGHT_DOMAIN = 0.45
CHIEF_SCORE_WEIGHT_PVS = 0.40
CHIEF_SCORE_WEIGHT_GEOMETRY = 0.15

# ---------------------------------------------------------------------------
# ENGINE Council — provider and rounds
# ---------------------------------------------------------------------------
# Change ENGINE_PROVIDER to route specialist agents + Chief Engineer through
# a different LLM backend.
#
#   "anthropic" → Claude Sonnet  (ENGINE_MODEL_ANTHROPIC)
#   "openai"    → GPT-4o         (ENGINE_MODEL_OPENAI)
#   "ollama"    → local model    (ENGINE_MODEL_OLLAMA at OLLAMA_BASE_URL)
#
# How to switch:
#   To OpenAI  : set ENGINE_PROVIDER = "openai"   + OPENAI_API_KEY in env
#   To Ollama  : set ENGINE_PROVIDER = "ollama"   (no API key needed)
#   To Anthropic: set ENGINE_PROVIDER = "anthropic"
# ---------------------------------------------------------------------------

ENGINE_PROVIDER        = "openai"                # ← flip this to switch

ENGINE_MODEL_ANTHROPIC = TRANSLATION_MODEL       # claude-sonnet-4-6
ENGINE_MODEL_OPENAI    = "gpt-4o"                # gpt-4o | gpt-4o-mini | o1-mini
ENGINE_MODEL_OLLAMA    = "gemma4-flora"           # gemma4:31b + num_ctx 8192 (see Modelfile)

# Ollama server address (the machine running `ollama serve`)
OLLAMA_BASE_URL        = os.getenv("FLOWPILOT_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")

# ---------------------------------------------------------------------------
# Council rounds — how many deliberation rounds per provider
# ---------------------------------------------------------------------------
# Cloud models (Anthropic, OpenAI) converge fast — 2 rounds is sufficient.
# Local models benefit from more rounds to compensate for weaker per-call
# reasoning. Increase OLLAMA_COUNCIL_ROUNDS for harder problems.
# ---------------------------------------------------------------------------

ENGINE_MAX_ROUNDS = {
    "anthropic": 2,
    "openai":    2,
    "ollama":    3,    # ← increase this for more deliberation on local models
}

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

COLLECTION_NAME = "flora_records"
PAIRS_COLLECTION_NAME = "flora_pairs"
TOP_K_RETRIEVAL = 20
TOP_K_ANALOGIES = 3

# Reranking weights
W_SEMANTIC = 0.6
W_FIELD = 0.4

# Field similarity weights
W_PHOTOCATALYST = 0.30
W_SOLVENT = 0.20
W_WAVELENGTH = 0.20
W_TEMPERATURE = 0.15
W_CONCENTRATION = 0.15

# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

MAX_COUNCIL_ROUNDS = 3

# Kinetics thresholds
INTENSIFICATION_LOW_BOUND = 0.3   # factor below median → warning
INTENSIFICATION_HIGH_BOUND = 3.0  # factor above median → warning
DEFAULT_INTENSIFICATION = 50.0    # median for photocatalytic reactions

# Fluidics thresholds
PRESSURE_WARNING_FRACTION = 0.8   # warn at 80% of pump max
RE_TURBULENT = 2300
RE_VERY_LOW = 1

# ---------------------------------------------------------------------------
# Photocatalyst class matching
# ---------------------------------------------------------------------------

PHOTOCATALYST_CLASSES: dict[str, str] = {
    "Ir(ppy)3": "Ir_cyclometalated",
    "fac-Ir(ppy)3": "Ir_cyclometalated",
    "[Ir(ppy)2(dtbbpy)]PF6": "Ir_cyclometalated",
    "[Ir(dF(CF3)ppy)2(dtbbpy)]PF6": "Ir_cyclometalated",
    "Ir(dF(CF3)ppy)2(dtbbpy)PF6": "Ir_cyclometalated",
    "Ru(bpy)3Cl2": "Ru_polypyridyl",
    "[Ru(bpy)3]Cl2": "Ru_polypyridyl",
    "[Ru(bpy)3]2+": "Ru_polypyridyl",
    "Ru(bpy)3(PF6)2": "Ru_polypyridyl",
    "4CzIPN": "organic_donor_acceptor",
    "3DPA2FBN": "organic_donor_acceptor",
    "TPT": "organic_donor_acceptor",
    "Eosin Y": "organic_xanthene",
    "Rose Bengal": "organic_xanthene",
    "Fluorescein": "organic_xanthene",
    "Methylene Blue": "organic_thiazine",
    "Acridinium": "organic_acridinium",
}

# Map classes to metal families for partial matching
METAL_FAMILIES: dict[str, str] = {
    "Ir_cyclometalated": "Ir",
    "Ru_polypyridyl": "Ru",
    "organic_donor_acceptor": "organic",
    "organic_xanthene": "organic",
    "organic_thiazine": "organic",
    "organic_acridinium": "organic",
}

# ---------------------------------------------------------------------------
# Solvent properties
# ---------------------------------------------------------------------------

SOLVENT_VISCOSITY_cP: dict[str, float] = {
    "MeCN": 0.37, "acetonitrile": 0.37, "CH3CN": 0.37,
    "DMF": 0.92, "N,N-dimethylformamide": 0.92,
    "DMSO": 2.00, "dimethyl sulfoxide": 2.00,
    "THF": 0.46, "tetrahydrofuran": 0.46,
    "EtOAc": 0.45, "ethyl acetate": 0.45,
    "DCM": 0.44, "dichloromethane": 0.44, "CH2Cl2": 0.44,
    "MeOH": 0.54, "methanol": 0.54,
    "EtOH": 1.08, "ethanol": 1.08,
    "iPrOH": 2.04, "isopropanol": 2.04,
    "toluene": 0.56,
    "water": 1.00, "H2O": 1.00,
    "DMA": 0.92, "dimethylacetamide": 0.92,
    "NMP": 1.67, "N-methylpyrrolidone": 1.67,
    "dioxane": 1.18, "1,4-dioxane": 1.18,
    "benzene": 0.60,
    "hexane": 0.31,
    "acetone": 0.31,
}

SOLVENT_DENSITY_g_mL: dict[str, float] = {
    "MeCN": 0.786, "acetonitrile": 0.786,
    "DMF": 0.944, "DMSO": 1.100,
    "THF": 0.889, "EtOAc": 0.902,
    "DCM": 1.327, "dichloromethane": 1.327,
    "MeOH": 0.791, "EtOH": 0.789,
    "toluene": 0.867, "water": 1.000,
    "DMA": 0.937, "NMP": 1.028,
    "dioxane": 1.034, "acetone": 0.784,
    "hexane": 0.659, "benzene": 0.879,
}

SOLVENT_BOILING_POINT_C: dict[str, float] = {
    "MeCN": 82, "acetonitrile": 82,
    "DMF": 153, "DMSO": 189,
    "THF": 66, "EtOAc": 77,
    "DCM": 40, "dichloromethane": 40,
    "MeOH": 65, "EtOH": 78,
    "toluene": 111, "water": 100,
    "DMA": 165, "NMP": 202,
    "dioxane": 101, "acetone": 56,
    "hexane": 69, "benzene": 80,
}

# Material incompatibility table: (material, solvent) → concern
INCOMPATIBLE_COMBOS: dict[tuple[str, str], str] = {
    ("FEP", "THF"): "FEP swells in THF above 60°C",
    ("PTFE", "DCM"): "PTFE not recommended for DCM at elevated temperature",
    ("SS", "HCl"): "SS corrodes in HCl",
    ("FEP", "toluene"): "FEP may swell in toluene above 80°C",
}
