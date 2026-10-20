#%%
import fitz          # PyMuPDF
import pdfplumber
import json, os, re
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")


def extract_text_pymupdf(pdf_path: str) -> str:
    """Fast full-text extraction — good for body text."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def extract_tables_pdfplumber(pdf_path: str) -> list[str]:
    """Extract tables separately — conditions are often in Table 1/2."""
    tables_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                if table:
                    rows = [" | ".join(str(c) for c in row if c) for row in table]
                    tables_text.append(f"[Table on page {i+1}]\n" + "\n".join(rows))
    return tables_text

    
def parse_pdf(pdf_path: str) -> dict:
    """Combine body text + tables into one document object."""
    body   = extract_text_pymupdf(pdf_path)
    tables = extract_tables_pdfplumber(pdf_path)
    return {
        "path"  : pdf_path,
        "body"  : body,
        "tables": tables,
        "full"  : body + "\n\n" + "\n\n".join(tables),
    }


# Section headers commonly used in chemistry journals
EXPERIMENTAL_MARKERS = [
    r"experimental\s+section",
    r"experimental\s+procedure",
    r"general\s+procedure",
    r"materials\s+and\s+methods",
    r"synthesis\s+procedure",
    r"typical\s+procedure",
    r"general\s+method",
    r"optimized\s+conditions",
    r"reaction\s+conditions",
]

def locate_experimental(text: str, window_chars: int = 6000) -> str:
    """
    Find the experimental section by regex and return a text window.
    If not found, fall back to the last 40% of the paper 
    (experimental is almost always in the second half).
    """
    text_lower = text.lower()
    for marker in EXPERIMENTAL_MARKERS:
        match = re.search(marker, text_lower)
        if match:
            start = match.start()
            return text[start : start + window_chars]

    # Fallback: last 40% of paper
    fallback_start = int(len(text) * 0.60)
    return text[fallback_start : fallback_start + window_chars]


CONDITION_SCHEMA = {
    # ── Reaction identity ──────────────────────────────────────
    "reaction_type"         : "",       # e.g. "Suzuki-Miyaura coupling"
    "reactor_type"          : "",       # "round-bottom flask" | "microreactor" | "packed-bed" | ...
    "process_mode"          : "",       # "batch" | "flow"

    # ── Substrates ─────────────────────────────────────────────
    "electrophile"          : "",       # e.g. "4-bromoanisole"
    "nucleophile"           : "",       # e.g. "phenylboronic acid"
    "electrophile_eq"       : None,     # equivalents
    "nucleophile_eq"        : None,

    # ── Catalyst system ────────────────────────────────────────
    "catalyst"              : "",       # e.g. "Pd(OAc)2"
    "catalyst_mol_percent"  : None,
    "ligand"                : "",       # e.g. "SPhos"
    "catalyst_phase"        : "",       # "homogeneous" | "heterogeneous"

    # ── Reaction medium ────────────────────────────────────────
    "base"                  : "",
    "base_eq"               : None,
    "solvent"               : "",
    "solvent_ratio"         : "",       # e.g. "DMF/H2O 4:1"
    "concentration_M"       : None,

    # ── Conditions ─────────────────────────────────────────────
    "temperature_C"         : None,
    "reaction_time_min"     : None,     # batch-specific
    "residence_time_min"    : None,     # flow-specific
    "flow_rate_mL_min"      : None,     # flow-specific
    "pressure_bar"          : None,     # flow-specific (back pressure)
    "atmosphere"            : "",       # "N2" | "Ar" | "air" | "H2" | ...

    # ── Outcome ────────────────────────────────────────────────
    "scale_mmol"            : None,
    "yield_percent"         : None,
    "conversion_percent"    : None,
    "selectivity_percent"   : None,

    # ── Metadata ───────────────────────────────────────────────
    "has_optimization_table": False,    # True if paper shows condition screening
    "explicitly_batch"      : False,    # paper uses the word "batch"
    "explicitly_flow"       : False,    # paper uses the word "flow"/"continuous"
    "confidence"            : 0,        # 1-3: how clearly conditions are stated
    "notes"                 : ""
}


def extract_conditions(text: str, pdf_path: str) -> dict:
    prompt = f"""
You are a chemistry data extraction expert. Extract ALL reaction conditions 
from the experimental text below into the JSON schema provided.

STRICT RULES:
- Set a field to null if it is not explicitly stated. Do NOT guess or infer.
- Convert all times to minutes (e.g. "12 h" → 720).
- Convert all temperatures to Celsius.
- If multiple procedures are described, extract the OPTIMIZED or BEST condition set.
- Set "confidence" to 3 if conditions are clearly tabulated, 2 if clearly written 
  in prose, 1 if scattered or ambiguous.

Schema to fill:
{json.dumps(CONDITION_SCHEMA, indent=2)}

Experimental text:
{text}

Return only the filled JSON. No explanation outside the JSON.
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    result["source_pdf"] = os.path.basename(pdf_path)
    return result




CONDITION_SCHEMA = {
    # ── Reaction identity ──────────────────────────────────────
    "reaction_type"         : "",       # e.g. "Suzuki-Miyaura coupling"
    "reactor_type"          : "",       # "round-bottom flask" | "microreactor" | "packed-bed" | ...
    "process_mode"          : "",       # "batch" | "flow"

    # ── Substrates ─────────────────────────────────────────────
    "electrophile"          : "",       # e.g. "4-bromoanisole"
    "nucleophile"           : "",       # e.g. "phenylboronic acid"
    "electrophile_eq"       : None,     # equivalents
    "nucleophile_eq"        : None,

    # ── Catalyst system ────────────────────────────────────────
    "catalyst"              : "",       # e.g. "Pd(OAc)2"
    "catalyst_mol_percent"  : None,
    "ligand"                : "",       # e.g. "SPhos"
    "catalyst_phase"        : "",       # "homogeneous" | "heterogeneous"

    # ── Reaction medium ────────────────────────────────────────
    "base"                  : "",
    "base_eq"               : None,
    "solvent"               : "",
    "solvent_ratio"         : "",       # e.g. "DMF/H2O 4:1"
    "concentration_M"       : None,

    # ── Conditions ─────────────────────────────────────────────
    "temperature_C"         : None,
    "reaction_time_min"     : None,     # batch-specific
    "residence_time_min"    : None,     # flow-specific
    "flow_rate_mL_min"      : None,     # flow-specific
    "pressure_bar"          : None,     # flow-specific (back pressure)
    "atmosphere"            : "",       # "N2" | "Ar" | "air" | "H2" | ...

    # ── Outcome ────────────────────────────────────────────────
    "scale_mmol"            : None,
    "yield_percent"         : None,
    "conversion_percent"    : None,
    "selectivity_percent"   : None,

    # ── Metadata ───────────────────────────────────────────────
    "has_optimization_table": False,    # True if paper shows condition screening
    "explicitly_batch"      : False,    # paper uses the word "batch"
    "explicitly_flow"       : False,    # paper uses the word "flow"/"continuous"
    "confidence"            : 0,        # 1-3: how clearly conditions are stated
    "notes"                 : ""
}


def extract_conditions(text: str, pdf_path: str) -> dict:
    prompt = f"""
You are a chemistry data extraction expert. Extract ALL reaction conditions 
from the experimental text below into the JSON schema provided.

STRICT RULES:
- Set a field to null if it is not explicitly stated. Do NOT guess or infer.
- Convert all times to minutes (e.g. "12 h" → 720).
- Convert all temperatures to Celsius.
- If multiple procedures are described, extract the OPTIMIZED or BEST condition set.
- Set "confidence" to 3 if conditions are clearly tabulated, 2 if clearly written 
  in prose, 1 if scattered or ambiguous.

Schema to fill:
{json.dumps(CONDITION_SCHEMA, indent=2)}

Experimental text:
{text}

Return only the filled JSON. No explanation outside the JSON.
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    result["source_pdf"] = os.path.basename(pdf_path)
    return result


#%%

import glob

def process_folder(folder_path: str, label: str) -> list[dict]:
    """
    label: "batch" or "flow" — you already know which folder is which.
    """
    pdfs    = glob.glob(os.path.join(folder_path, "*.pdf"))
    records = []

    for pdf_path in pdfs:
        print(f"  Processing: {os.path.basename(pdf_path)}")
        try:
            doc        = parse_pdf(pdf_path)
            exp_text   = locate_experimental(doc["full"])
            conditions = extract_conditions(exp_text, pdf_path)
            conditions["expected_process_mode"] = label  # ground truth label
            records.append(conditions)
        except Exception as e:
            print(f"    [warn] Failed: {e}")

    return records

#%%
# ── Run ────────────────────────────────────────────────────────────────────
batch_records = process_folder("pdfs/batch/", label="batch")
flow_records  = process_folder("pdfs/flow/",  label="flow")
all_records   = batch_records + flow_records

# ── Save ───────────────────────────────────────────────────────────────────
with open("suzuki_conditions.json", "w") as f:
    json.dump(all_records, f, indent=2)

print(f"\nExtracted {len(batch_records)} batch + {len(flow_records)} flow records.")
print(f"Saved to suzuki_conditions.json")
# %%


FULL_PROCESS_SCHEMA = {

    # ════════════════════════════════════════════════
    # LAYER 1 — Reaction conditions (already have this)
    # ════════════════════════════════════════════════
    "conditions": {
        "catalyst"          : "",
        "ligand"            : "",
        "base"              : "",
        "solvent"           : "",
        "temperature_C"     : None,
        "reaction_time_min" : None,     # batch
        "residence_time_min": None,     # flow
        "flow_rate_mL_min"  : None,     # flow
        "concentration_M"   : None,
        "pressure_bar"      : None,
        "yield_percent"     : None,
        "scale_mmol"        : None,
        "atmosphere"        : "",
    },

    # ════════════════════════════════════════════════
    # LAYER 2 — Process design
    # ════════════════════════════════════════════════
    "process_design": {

        # Reactor hardware
        "reactor_type"       : "",  # "coil" | "chip" | "packed-bed" | "CSTR" |
                                    # "tubular" | "round-bottom flask" | "autoclave"
        "reactor_material"   : "",  # "PTFE" | "stainless steel" | "glass" | "PFA"
        "reactor_volume_mL"  : None,
        "reactor_length_m"   : None,
        "tubing_diameter_mm" : None,

        # Pumping / flow control
        "pump_type"          : "",  # "syringe" | "HPLC" | "peristaltic" | "none"
        "number_of_inlets"   : None,
        "mixer_type"         : "",  # "T-mixer" | "Y-mixer" | "static mixer" |
                                    # "microchip" | "magnetic stirring"

        # Temperature control
        "heating_method"     : "",  # "oil bath" | "microwave" | "heat exchanger" |
                                    # "Peltier" | "sand bath" | "column oven"
        "cooling_required"   : False,  # True if reaction needs active cooling

        # Pressure
        "back_pressure_regulator": False,
        "back_pressure_bar"  : None,

        # Phase regime
        "phase_regime"       : "",  # "single-phase" | "liquid-liquid" |
                                    # "gas-liquid" | "liquid-solid" | "triphasic"
        "gas_involved"       : "",  # e.g. "H2", "CO", "O2", "none"

        # Multi-step connectivity
        "is_multistep"       : False,
        "number_of_steps"    : None,
        "steps_sequence"     : [],  # e.g. ["lithiation", "borylation", "Suzuki"]
        "inline_quench"      : False,
        "inline_workup"      : False,

        # Inline analytics
        "inline_analytics"   : [],  # e.g. ["IR", "UV-Vis", "HPLC", "NMR"]

        # Solids handling
        "solids_present"     : False,
        "solids_strategy"    : "",  # "slurry pump" | "ultrasound" | "dissolution"
                                    # | "packed-bed" | "none"
    },

    # ════════════════════════════════════════════════
    # LAYER 3 — Engineering logic (the "why")
    # ════════════════════════════════════════════════
    "engineering_logic": {

        # Why flow / why batch
        "batch_limitation"   : "",  # what problem batch had
                                    # e.g. "poor heat dissipation", "exotherm",
                                    # "short-lived intermediate", "scale-up issues"
        "flow_advantage"     : "",  # what flow solved
                                    # e.g. "better mixing", "safe handling of HCN",
                                    # "precise residence time control"

        # Key translation decisions
        "residence_time_reasoning": "",    # why this residence time was chosen
        "temperature_reasoning"   : "",    # why this temperature vs batch
        "concentration_change"    : "",    # was concentration changed? why?
        "solvent_change"          : "",    # was solvent changed for flow? why?
                                           # (e.g. viscosity, solubility of catalyst)

        # Kinetics discussed?
        "kinetics_discussed"      : False,
        "rate_limiting_step"      : "",    # e.g. "transmetalation", "oxidative addition"
        "reaction_type_kinetics"  : "",    # "Type A: fast" | "Type B: moderate" |
                                           # "Type C: slow" per Roberge classification

        # Safety
        "safety_concern"          : "",    # e.g. "exotherm", "toxic gas", "explosive"
        "safety_resolution"       : "",    # how flow addressed it

        # Scale-up
        "scale_up_discussed"      : False,
        "scale_up_strategy"       : "",    # "numbering-up" | "longer coil" |
                                           # "larger diameter tubing"

        # Batch-flow performance comparison
        "batch_yield_percent"     : None,  # if paper explicitly compares
        "flow_yield_percent"      : None,
        "time_reduction_factor"   : None,  # e.g. 12h batch → 10min flow = 72x
    },

    # ════════════════════════════════════════════════
    # LAYER 4 — Process figures (extracted separately)
    # ════════════════════════════════════════════════
    "process_figures": {
        "has_reactor_scheme"      : False,  # True if paper has a reactor diagram
        "has_flow_diagram"        : False,  # True if paper has a process flow chart
        "has_optimization_table"  : False,
        "figure_descriptions"     : [],     # LLM descriptions of each figure
    },

    # ════════════════════════════════════════════════
    # Metadata
    # ════════════════════════════════════════════════
    "source_pdf"     : "",
    "process_mode"   : "",   # "batch" | "flow"
    "confidence"     : 0,    # 1-3
    "notes"          : ""
}


import fitz   # PyMuPDF
import base64

def extract_figures(pdf_path: str, output_dir: str = "figures") -> list[dict]:
    """Extract all images from PDF and save them."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    figures = []

    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref  = img[0]
            image = doc.extract_image(xref)
            ext   = image["ext"]
            data  = image["image"]

            # Skip tiny images (logos, icons — not process figures)
            if len(data) < 10_000:
                continue

            fname = f"{output_dir}/p{page_num+1}_img{img_index}.{ext}"
            with open(fname, "wb") as f:
                f.write(data)
            figures.append({"path": fname, "page": page_num + 1, "data": data})

    doc.close()
    return figures


def describe_figure(image_data: bytes) -> dict:
    """Ask GPT-4o to describe a figure and classify it."""
    b64 = base64.b64encode(image_data).decode("utf-8")
    prompt = """
You are a flow chemistry expert. Describe this figure extracted from a chemistry paper.

Return JSON:
{
  "figure_type": "<reactor_scheme|flow_diagram|optimization_table|
                   reaction_scheme|graph|structure|other>",
  "description": "<2-3 sentence description of what is shown>",
  "components_identified": ["pump", "mixer", "coil reactor", ...],
  "process_mode": "<batch|flow|both|unclear>",
  "key_information": "<any specific conditions, dimensions, or parameters visible>"
}
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",       "text": prompt},
                {"type": "image_url",  "image_url": {
                    "url": f"data:image/png;base64,{b64}"
                }}
            ]
        }],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)



def extract_full_process(pdf_path: str, label: str) -> dict:
    print(f"  [{label}] {os.path.basename(pdf_path)}")

    # Text + tables
    doc      = parse_pdf(pdf_path)
    exp_text = locate_experimental(doc["full"])

    # Layer 1, 2, 3 — text extraction
    prompt = f"""
You are a process chemistry expert. Extract the FULL process design from 
the experimental text below. Fill every field carefully.

RULES:
- null for anything not explicitly stated
- Convert times to minutes, temperatures to Celsius
- For "batch_limitation" and "flow_advantage": quote or closely paraphrase 
  the paper's own reasoning — do not invent explanations
- For "steps_sequence": list reaction steps in order if multistep

Schema:
{json.dumps(FULL_PROCESS_SCHEMA, indent=2)}

Experimental text:
{exp_text}

Return only valid JSON.
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    record = json.loads(response.choices[0].message.content)

    # Layer 4 — figure extraction
    figures      = extract_figures(pdf_path)
    descriptions = [describe_figure(f["data"]) for f in figures]

    # Keep only meaningful figures (reactor schemes and flow diagrams)
    relevant = [
        d for d in descriptions
        if d.get("figure_type") in ("reactor_scheme", "flow_diagram", "optimization_table")
    ]
    record["process_figures"]["figure_descriptions"] = relevant
    record["process_figures"]["has_reactor_scheme"]  = any(
        d["figure_type"] == "reactor_scheme" for d in relevant
    )
    record["process_figures"]["has_flow_diagram"]    = any(
        d["figure_type"] == "flow_diagram" for d in relevant
    )

    record["source_pdf"]   = os.path.basename(pdf_path)
    record["process_mode"] = label
    return record

#%%
# ── Run on your folders ────────────────────────────────────────────────────
batch_records = [extract_full_process(p, "batch")
                 for p in glob.glob("pdfs/batch/*.pdf")]
flow_records  = [extract_full_process(p, "flow")
                 for p in glob.glob("pdfs/flow/*.pdf")]

with open("suzuki_full_process.json", "w") as f:
    json.dump(batch_records + flow_records, f, indent=2)

# %%
