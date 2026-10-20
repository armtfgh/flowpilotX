"""Shared feed identity and role normalization, independent of chemical names."""

from __future__ import annotations

import re
from collections.abc import Iterable


_QUANTITY = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mM|M|mmol|mol|equiv|eq\.?|mg|g|mL|uL|wt\s*%)(?![A-Za-z])",
    re.I,
)
_MEDIUM_ROLE = re.compile(r"\b(?:co[ -]?solvent|solvent|carrier|diluent)\b", re.I)
_REACTIVE_ROLE = re.compile(
    r"\b(?:base|acid|substrate|reactant|reagent|catalyst|photocatalyst|oxidant|"
    r"reductant|donor|acceptor|nucleophile|electrophile)\b", re.I,
)

# Exact conventional synonyms only; no substring/fuzzy chemical identity matching.
SOLVENT_IDENTITY_ALIASES = (
    ("EtOH", "ethanol"), ("MeOH", "methanol"),
    ("MeCN", "acetonitrile"), ("THF", "tetrahydrofuran"),
    ("2-MeTHF", "2-methyltetrahydrofuran"),
    ("DMSO", "dimethyl sulfoxide"), ("DCM", "dichloromethane"),
)


def has_catalytic_role(value: str) -> bool:
    """Recognize catalyst role variants without depending on a compound name."""
    return bool(re.search(r"\b(?:pre[- ]?)?(?:photo|electro)?catal(?:yst|ytic)\b", str(value), re.I))


def has_stationary_catalytic_role(value: str, *, declared_loading: str = "") -> bool:
    """Shared placement classification, used only in a packed-bed context."""
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))
    stationary = re.search(
        r"\b(?:heterogeneous|immobilized|supported catalyst|pre packed|prepacked|"
        r"packed|cartridge|fixed bed|stationary|not in solution|not in feed|"
        r"dissolved loading not applicable)\b", normalized,
    )
    # Use the component's declared flow loading, not a bed mentioned elsewhere
    # or only in a historical batch procedure.
    flow_loading = re.split(r"\bflow\s*:", declared_loading, flags=re.I)[-1]
    fixed_loading = re.search(
        r"\b(?:pre[ -]?)?packed[ -](?:bed|reactor|tube|column)\b|\bcartridge\b", flow_loading, re.I,
    )
    # Physical placement is authoritative even when the model calls a bed
    # component a material, source, reagent or adsorbent instead of a catalyst.
    supported_material = re.search(r"\b(?:heterogeneous|immobilized|supported)\b", normalized)
    pumped_solid = re.search(
        r"\b(?:slurry|suspension|suspended|pumped feed)\b",
        f"{normalized} {flow_loading}", re.I,
    )
    explicit_stationary_phase = re.search(r"\b(?:stationary|fixed) phase\b", normalized)
    installed_bed = re.search(r"\b(?:installed|supplied|existing)\s+(?:[^.;]+?\s+)?bed\b", flow_loading, re.I)
    return bool(not pumped_solid and (
        (has_catalytic_role(normalized) and stationary)
        or (supported_material and (fixed_loading or explicit_stationary_phase or installed_bed))
    ))


def has_output_only_role(value: str) -> bool:
    """An output identity does not authorize adding that species to a feed.

    Explicit dual input uses, such as product seeding or an analytical standard,
    remain inputs. The caller preserves deliberately declared feed membership.
    """
    role = str(value).casefold()
    output = re.search(r"\b(?:product|by[ -]?product|intermediate)\b", role)
    input_use = re.search(r"\b(?:substrate|reactant|reagent|additive|seed|seeding|standard|starting material|feed)\b", role)
    return bool(output and not input_use)


def component_name(value: str) -> str:
    text = str(value).strip()
    # Strip a dose/role annotation, never parentheses within a chemical name.
    tail = re.search(r"\s+\(([^()]*)\)\s*$", text)
    if tail and (
        _QUANTITY.search(tail[1]) or _MEDIUM_ROLE.search(tail[1])
        or re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?", tail[1])
    ):
        return text[:tail.start()].strip() or text
    return text


def component_key(value: str) -> str:
    return re.sub(r"\s+", "", component_name(value)).casefold()


def unique_components(values: Iterable[str]) -> list[str]:
    """Discard bare duplicates, but retain conflicting annotated entries for validation."""
    groups: dict[str, list[str]] = {}
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        group = groups.setdefault(component_key(text), [])
        if text.casefold() not in {item.casefold() for item in group}:
            group.append(text)
    output = []
    for group in groups.values():
        annotated = [item for item in group if component_name(item) != item]
        output.extend(annotated or group[:1])
    return output


def is_solvent_component(source: str, declared_solvent: str, planned_role: str) -> bool:
    """A medium-only annotation is not a stoichiometric reagent requirement.

    A dual role (for example solvent/base) remains reactive. An explicit inline
    medium-only annotation overrides a mistaken upstream role, as before.
    """
    generic_buffer = _generic_buffer(component_name(source))
    # A pH-only buffer is an unidentified medium, even if a model also calls
    # it a base. Its salt identity/strength cannot be inferred from substrate C.
    if generic_buffer:
        return True
    tail = re.search(r"\s+\(([^()]*)\)\s*$", source)
    if tail and _MEDIUM_ROLE.search(tail[1]):
        return not bool(_REACTIVE_ROLE.search(tail[1]))
    if _MEDIUM_ROLE.search(planned_role):
        return not bool(_REACTIVE_ROLE.search(planned_role))
    if _REACTIVE_ROLE.search(planned_role):
        return False
    return bool(declared_solvent) and declared_solvent_member(component_name(source), declared_solvent)


def _generic_buffer(value):
    # A phase descriptor is not a chemical species or a buffer stock strength.
    return re.fullmatch(r"pH\s*(\d+(?:\.\d+)?)\s+(?:aqueous\s+)?buffer(?:\s*\((?:aqueous|aq\.?)\))?", value.strip(), re.I)


def declared_solvent_member(name: str, solvent: str) -> bool:
    """Match only explicitly named solvent-mixture members, without chemical guessing."""
    mixture = re.sub(r",\s*(?:degassed|deoxygenated|sparged|purged)\b.*$", "", solvent, flags=re.I)
    mixture = re.sub(r"\(\s*\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\s*,?\s*(?:v/v)?\s*\)", "", mixture, flags=re.I)
    mixture = re.sub(r"\s*\([^()]*\d[^()]*\)\s*$", "", mixture).strip()
    mixture = re.sub(r"\s+\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\s*(?:v/v)?\s*$", "", mixture, flags=re.I)
    members = re.split(r"\s*[:/]\s*", mixture)
    def keys(value):
        variants = [value]
        # pH describes the aqueous medium, not a molar amount of an unnamed salt.
        buffer = _generic_buffer(value)
        if buffer:
            variants += [f'pH {buffer[1]} buffer', f'pH {buffer[1]} aqueous buffer']
        for group in SOLVENT_IDENTITY_ALIASES:
            if component_key(value) in {component_key(x) for x in group}:
                variants.extend(group)
        return {component_key(x) for x in variants}
    return bool(keys(name).intersection(set().union(*(keys(item) for item in [solvent, mixture, *members] if item))))


def protocol_component_quantity(name: str, protocol: str) -> str:
    """Read a dose immediately following an exact chemical name in the protocol."""
    doses = []
    for match in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(component_name(name)) + r"\s*\(([^()]*)\)", protocol, re.I):
        dose = re.search(r"\d+(?:\.\d+)?\s*(?:mol\s*%|equiv(?:alents)?|eq\.?)(?![A-Za-z])", match[1], re.I)
        if dose:
            doses.append(dose[0])
    # Multiple different doses require stage-specific interpretation, not a guess.
    unique = list(dict.fromkeys(doses))
    return unique[0] if len(unique) == 1 else ""
