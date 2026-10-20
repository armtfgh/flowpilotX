#!/usr/bin/env python3
"""Offline, source-preserving OOXML slide composition for the KHU review decks.

Requires lxml (available in the workspace's flent environment); does not use
python-pptx, a renderer, a model, or network access.  The plan is JSON:
  {"sources": {"new": "additions.pptx", "qwen": "qwen.pptx", ...},
   "root_source": "new",
   "slides": [{"source": "qwen", "slide": 9, "section": "Qwen 3.6-27B"}]}
Paths are resolved relative to the current working directory, then the plan
directory as a fallback.  Source slide numbers refer to presentation order.

Native slide XML is retained except the two existing bottom footer shapes.
Relationships are relocated while their IDs remain unchanged in copied parts.
Layouts, masters, themes, notes, media, charts and embedded objects are copied
recursively.  Differing single-instance notes masters, conflicting table-style
definitions, source-level embedded fonts and duplicate selected source slides
are deliberately rejected instead of silently changing document semantics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import posixpath
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree as E

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
}
R = "{" + NS["r"] + "}"
P = "{" + NS["p"] + "}"
PR = "{" + NS["pr"] + "}"
CT = "{" + NS["ct"] + "}"
PARSER = E.XMLParser(resolve_entities=False, no_network=True, remove_blank_text=False)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse(data: bytes):
    return E.fromstring(data, parser=PARSER)


def xml(root) -> bytes:
    return E.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)


def rel_path(part: str) -> str:
    if not part:
        return "_rels/.rels"
    return posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")


def resolve(owner: str, target: str) -> tuple[str, str]:
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        raise ValueError(f"Internal relationship contains an external URL: {target}")
    path = unquote(parsed.path)
    resolved = posixpath.normpath(path.lstrip("/")) if path.startswith("/") else posixpath.normpath(posixpath.join(posixpath.dirname(owner), path))
    if resolved == ".." or resolved.startswith("../"):
        raise ValueError(f"Relationship escapes package root: {owner}: {target}")
    return resolved, ("#" + parsed.fragment if parsed.fragment else "")


def relocated(owner: str, target: str, fragment: str = "") -> str:
    return posixpath.relpath(target, posixpath.dirname(owner) or ".") + fragment


def structural_hash(root) -> str:
    """Namespace-prefix-independent hash, including all attributes and text."""
    def node(el):
        return [str(el.tag), sorted(el.attrib.items()), el.text or "", el.tail or "", [node(c) for c in el]]
    return sha(json.dumps(node(root), ensure_ascii=False, separators=(",", ":")).encode())


class Source:
    def __init__(self, key: str, path: Path):
        self.key, self.path = key, path
        self.hash_before = sha(path.read_bytes())
        with ZipFile(path) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate ZIP member names in {path}")
            self.parts = {n: z.read(n) for n in names if not n.endswith("/")}
        root_rels = self.rels("")
        office = [r for r in root_rels if r.get("Type", "").endswith("/officeDocument")]
        if len(office) != 1:
            raise ValueError(f"Expected one officeDocument relationship: {path}")
        self.main, _ = resolve("", office[0].get("Target"))
        self.presentation = parse(self.parts[self.main])
        self.presentation_rels = self.rels(self.main)
        self.rel_by_id = {r.get("Id"): r for r in self.presentation_rels}
        self.slide_parts = []
        for entry in self.presentation.findall("p:sldIdLst/p:sldId", NS):
            rel = self.rel_by_id[entry.get(R + "id")]
            self.slide_parts.append(resolve(self.main, rel.get("Target"))[0])
        ct = parse(self.parts["[Content_Types].xml"])
        self.overrides = {x.get("PartName").lstrip("/"): x.get("ContentType") for x in ct if x.tag == CT + "Override"}
        self.defaults = {x.get("Extension").lower(): x.get("ContentType") for x in ct if x.tag == CT + "Default"}

    def rels(self, part: str):
        raw = self.parts.get(rel_path(part))
        return list(parse(raw)) if raw else []

    def content_type(self, part: str) -> str:
        result = self.overrides.get(part) or self.defaults.get(part.rsplit(".", 1)[-1].lower())
        if not result:
            raise ValueError(f"Missing content type: {self.key}: {part}")
        return result

    def direct_rel_part(self, suffix: str):
        matching = [r for r in self.presentation_rels if r.get("Type", "").endswith("/" + suffix)]
        if len(matching) > 1:
            raise ValueError(f"Multiple {suffix} parts unsupported: {self.key}")
        return resolve(self.main, matching[0].get("Target"))[0] if matching else None

    def equivalent_tree_hash(self, part: str, seen=()) -> str:
        """For acyclic notes-master graphs; normalize relationship IDs/targets."""
        if part in seen:
            raise ValueError(f"Cycle in single-master equivalence check: {part}")
        rels = self.rels(part)
        signatures = {}
        for rel in rels:
            if rel.get("TargetMode") == "External":
                dest = rel.get("Target")
            else:
                dest = self.equivalent_tree_hash(resolve(part, rel.get("Target"))[0], seen + (part,))
            signatures[rel.get("Id")] = (rel.get("Type"), rel.get("TargetMode", "Internal"), dest)
        raw = self.parts[part]
        if part.endswith(".xml"):
            root = parse(raw)
            for el in root.iter():
                for key, value in list(el.attrib.items()):
                    if key.startswith(R) and value in signatures:
                        el.set(key, json.dumps(signatures[value]))
            raw = structural_hash(root).encode()
        return sha(raw + json.dumps(sorted(signatures.values()), separators=(",", ":")).encode())


class Merger:
    def __init__(self, sources: dict[str, Source], root_source: str, plan: dict):
        self.sources, self.base, self.plan = sources, sources[root_source], plan
        self.parts: dict[str, bytes] = {}
        self.types: dict[str, str] = {}
        self.mapping: dict[tuple[str, str], str] = {}
        self.claims: dict[str, tuple[str, str]] = {}
        self.copied: set[tuple[str, str]] = set()
        self.part_audit = []
        self.slide_audit = []
        self.warnings = []
        self.single_master_aliases = []
        self.master_id_remapping = []
        self.master_metadata_audit = []
        self.main = "ppt/presentation.xml"
        self.mapping[(self.base.key, self.base.main)] = self.main
        self.claims[self.main] = (self.base.key, self.base.main)
        self.copied.add((self.base.key, self.base.main))
        self.selections = []
        seen = set()
        size = self.base.presentation.find("p:sldSz", NS)
        self.slide_size = dict(size.attrib)
        self.slide_height = int(size.get("cy"))
        for number, entry in enumerate(plan["slides"], 1):
            source = sources[entry["source"]]
            original = int(entry["slide"])
            if not 1 <= original <= len(source.slide_parts):
                raise ValueError(f"Invalid source slide {source.key}:{original}")
            srcpart = source.slide_parts[original - 1]
            key = (source.key, srcpart)
            if key in seen:
                raise ValueError(f"Duplicate selected source slide: {key}")
            seen.add(key)
            if dict(source.presentation.find("p:sldSz", NS).attrib) != self.slide_size:
                raise ValueError(f"Slide sizes differ: {source.key}")
            if source.presentation.find("p:embeddedFontLst", NS) is not None:
                raise ValueError(f"Source-level embedded fonts need explicit merge support: {source.key}")
            dest = f"ppt/slides/slide{number}.xml"
            self.mapping[key] = dest
            self.claims[dest] = key
            self.selections.append((number, source, original, srcpart, entry.get("section", "Shared")))
        self.prepare_notes_aliases()

    def prepare_notes_aliases(self):
        """A presentation has one notes master; alias only proven equivalents."""
        basenotes = self.base.direct_rel_part("notesMaster")
        for source in self.sources.values():
            if source.key == self.base.key:
                continue
            notes = source.direct_rel_part("notesMaster")
            if not notes:
                continue
            if not basenotes:
                raise ValueError("Root deck lacks the notes master required by an imported source")
            original_hash = source.equivalent_tree_hash(notes)
            base_hash = self.base.equivalent_tree_hash(basenotes)
            if original_hash != base_hash:
                raise ValueError(f"Differing notes-master appearance is not safely mergeable: {source.key}")
            target = self.destination(self.base, basenotes)
            self.mapping[(source.key, notes)] = target
            self.copied.add((source.key, notes))
            self.single_master_aliases.append({"source": source.key, "source_part": notes, "output_part": target, "equivalent_dependency_tree_sha256": original_hash})

    def destination(self, source: Source, part: str) -> str:
        key = (source.key, part)
        if key in self.mapping:
            return self.mapping[key]
        if part in source.slide_parts:
            raise ValueError(f"Relationship points to omitted slide: {source.key}:{part}")
        if part == source.main:
            raise ValueError(f"Unexpected imported presentation dependency: {source.key}")
        if source.key == self.base.key:
            target = part
        else:
            target = posixpath.join(posixpath.dirname(part), source.key + "__" + posixpath.basename(part))
        if target in self.claims and self.claims[target] != key:
            raise ValueError(f"Output part collision: {target}")
        self.mapping[key] = target
        self.claims[target] = key
        return target

    def copy_part(self, source: Source, part: str) -> str:
        target = self.destination(source, part)
        key = (source.key, part)
        if key in self.copied:
            return target
        if part not in source.parts:
            raise ValueError(f"Missing source dependency: {source.key}:{part}")
        self.copied.add(key)  # Break layout/master and slide/notes cycles.
        raw = source.parts[part]
        self.parts[target] = raw
        self.types[target] = source.content_type(part)
        rels = source.rels(part)
        if rels:
            out = E.Element(PR + "Relationships", nsmap={None: NS["pr"]})
            for original in rels:
                rel = copy.deepcopy(original)
                if rel.get("TargetMode") != "External":
                    child, fragment = resolve(part, rel.get("Target"))
                    rel.set("Target", relocated(target, self.copy_part(source, child), fragment))
                out.append(rel)
            self.parts[rel_path(target)] = xml(out)
        self.part_audit.append({"source": source.key, "source_part": part, "output_part": target, "source_sha256": sha(raw), "content_type": self.types[target]})
        return target

    def update_footer(self, raw: bytes, section: str, number: int):
        root = parse(raw)
        footer = []
        pages = []
        for shape in root.findall("p:cSld/p:spTree/p:sp", NS):
            off = shape.find("p:spPr/a:xfrm/a:off", NS)
            if off is None or int(off.get("y", "0")) < self.slide_height * 0.92:
                continue
            text = "".join(shape.xpath(".//a:t/text()", namespaces=NS)).strip()
            if text.startswith("FlowPilotX"):
                footer.append(shape)
            elif text.isdigit():
                pages.append(shape)
        if len(footer) != 1 or len(pages) != 1:
            raise ValueError(f"Slide {number}: expected one existing FlowPilotX footer and numeric page shape; found {len(footer)}/{len(pages)}")
        footer_y = footer[0].find("p:spPr/a:xfrm/a:off", NS).get("y")
        page_y = pages[0].find("p:spPr/a:xfrm/a:off", NS).get("y")
        if footer_y != page_y:
            raise ValueError(f"Slide {number}: numeric page and footer are at different y positions")
        shape_ids = [s.find("p:nvSpPr/p:cNvPr", NS).get("id") for s in footer + pages]
        def without_footers(r):
            r = copy.deepcopy(r)
            for sp in r.findall("p:cSld/p:spTree/p:sp", NS):
                nv = sp.find("p:nvSpPr/p:cNvPr", NS)
                if nv is not None and nv.get("id") in shape_ids:
                    sp.getparent().remove(sp)
            return structural_hash(r)
        before = without_footers(root)
        old_text = []
        for shape, replacement in [(footer[0], f"FlowPilotX   {section}   KHU conditional proposals   15 October 2026"), (pages[0], str(number))]:
            texts = shape.findall(".//a:t", NS)
            old_text.append("".join(t.text or "" for t in texts))
            if not texts:
                raise ValueError("Footer has no text run")
            texts[0].text = replacement
            for node in texts[1:]:
                node.text = ""
            shape.find("p:nvSpPr/p:cNvPr", NS).set("name", replacement)
        after = without_footers(root)
        if before != after:
            raise AssertionError(f"Non-footer slide content changed: {number}")
        return xml(root), {"footer_shape_ids": shape_ids, "previous_footer_and_page": old_text, "footer_y_emu_unchanged": int(footer_y), "non_footer_source_structural_sha256": before, "non_footer_output_structural_sha256": after, "non_footer_identical": True}

    def compose(self):
        pres = copy.deepcopy(self.base.presentation)
        if pres.find("p:custShowLst", NS) is not None:
            raise ValueError("Root custom shows require explicit remapping")
        for section in pres.xpath(".//*[local-name()='sectionLst']"):
            section.getparent().remove(section)
            self.warnings.append("Removed root section metadata; ordered plan sections are displayed in slide footers/dividers.")
        sld_list = pres.find("p:sldIdLst", NS)
        if sld_list is None:
            raise ValueError("Root presentation has no slide ID list")
        for child in list(sld_list):
            sld_list.remove(child)
        main_rels = E.Element(PR + "Relationships", nsmap={None: NS["pr"]})
        for original in self.base.presentation_rels:
            if original.get("Type", "").endswith("/slide"):
                continue
            rel = copy.deepcopy(original)
            if rel.get("TargetMode") != "External":
                child, fragment = resolve(self.base.main, rel.get("Target"))
                rel.set("Target", relocated(self.main, self.copy_part(self.base, child), fragment))
            main_rels.append(rel)
        used_ids = {r.get("Id") for r in main_rels}
        def add_main_rel(kind, target):
            n = 1
            while f"rIdMerge{n}" in used_ids:
                n += 1
            rid = f"rIdMerge{n}"
            used_ids.add(rid)
            E.SubElement(main_rels, PR + "Relationship", Id=rid, Type=NS["r"] + "/" + kind, Target=relocated(self.main, target))
            return rid
        for number, source, original, srcpart, section in self.selections:
            target = self.copy_part(source, srcpart)
            result, record = self.update_footer(self.parts[target], section, number)
            self.parts[target] = result
            rid = add_main_rel("slide", target)
            E.SubElement(sld_list, P + "sldId", id=str(255 + number), attrib={R + "id": rid})
            self.slide_audit.append({"output_slide": number, "source": source.key, "source_slide": original, "source_part": srcpart, "output_part": target, "section": section, **record})
        master_list = pres.find("p:sldMasterIdLst", NS)
        if master_list is None:
            master_list = E.Element(P + "sldMasterIdLst")
            pres.insert(0, master_list)
        master_targets = {resolve(self.main, r.get("Target"))[0] for r in main_rels if r.get("Type", "").endswith("/slideMaster")}
        next_master = max([int(e.get("id")) for e in master_list] + [2147483647]) + 1
        for source in self.sources.values():
            for entry in source.presentation.findall("p:sldMasterIdLst/p:sldMasterId", NS):
                rel = source.rel_by_id[entry.get(R + "id")]
                srcpart = resolve(source.main, rel.get("Target"))[0]
                target = self.mapping.get((source.key, srcpart))
                if target is None or target in master_targets:
                    continue
                rid = add_main_rel("slideMaster", target)
                E.SubElement(master_list, P + "sldMasterId", id=str(next_master), attrib={R + "id": rid})
                next_master += 1
                master_targets.add(target)
        self.assign_global_master_and_layout_ids(pres, main_rels)
        self.merge_table_styles()
        self.parts[self.main] = xml(pres)
        self.types[self.main] = self.base.content_type(self.base.main)
        self.parts[rel_path(self.main)] = xml(main_rels)
        root_rels = E.Element(PR + "Relationships", nsmap={None: NS["pr"]})
        for original in self.base.rels(""):
            rel = copy.deepcopy(original)
            if rel.get("TargetMode") != "External":
                child, fragment = resolve("", rel.get("Target"))
                rel.set("Target", self.copy_part(self.base, child) + fragment)
            root_rels.append(rel)
        self.parts["_rels/.rels"] = xml(root_rels)
        self.update_app_properties()
        content_types = E.Element(CT + "Types", nsmap={None: NS["ct"]})
        E.SubElement(content_types, CT + "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
        for part, ctype in sorted(self.types.items()):
            E.SubElement(content_types, CT + "Override", PartName="/" + part, ContentType=ctype)
        self.parts["[Content_Types].xml"] = xml(content_types)
        self.verify_package()

    def assign_global_master_and_layout_ids(self, pres, main_rels):
        """Avoid global ID collisions when multiple source masters are combined.

        Layout relationships and all native graphical content stay unchanged.
        Record both source and final registry IDs and independently compare each
        master's XML after masking only the layout registry id attributes.
        """
        main_rel_map = {r.get("Id"): r for r in main_rels}
        master_entries = pres.findall("p:sldMasterIdLst/p:sldMasterId", NS)
        next_id = 2147483648
        master_parts = []
        for entry in master_entries:
            rel = main_rel_map[entry.get(R + "id")]
            target = resolve(self.main, rel.get("Target"))[0]
            source_key, source_part = self.claims[target]
            source = self.sources[source_key]
            registrations = []
            for source_entry in source.presentation.findall("p:sldMasterIdLst/p:sldMasterId", NS):
                source_rel = source.rel_by_id[source_entry.get(R + "id")]
                if resolve(source.main, source_rel.get("Target"))[0] == source_part:
                    registrations.append({"source": source_key, "source_master_part": source_part, "source_master_id": int(source_entry.get("id"))})
            entry.set("id", str(next_id))
            self.master_id_remapping.append({"output_part": target, "output_master_id": next_id, "source_registrations": registrations})
            master_parts.append(target)
            next_id += 1
        for part in master_parts:
            root = parse(self.parts[part])
            def masked_hash(tree):
                tree = copy.deepcopy(tree)
                for item in tree.findall("p:sldLayoutIdLst/p:sldLayoutId", NS):
                    item.set("id", "LAYOUT_REGISTRY_ID")
                return structural_hash(tree)
            before = masked_hash(root)
            changes = []
            for entry in root.findall("p:sldLayoutIdLst/p:sldLayoutId", NS):
                original = int(entry.get("id"))
                entry.set("id", str(next_id))
                changes.append({"source_layout_id": original, "output_layout_id": next_id, "relationship_id_unchanged": entry.get(R + "id")})
                next_id += 1
            after = masked_hash(root)
            if before != after:
                raise AssertionError(f"Non-registry master content changed: {part}")
            if any(c["source_layout_id"] != c["output_layout_id"] for c in changes):
                self.parts[part] = xml(root)
            self.master_metadata_audit.append({"output_part": part, "layout_id_remapping": changes, "non_registry_source_structural_sha256": before, "non_registry_output_structural_sha256": after, "non_registry_content_identical": True})
        assigned = [x["output_master_id"] for x in self.master_id_remapping]
        assigned += [c["output_layout_id"] for x in self.master_metadata_audit for c in x["layout_id_remapping"]]
        if len(assigned) != len(set(assigned)):
            raise AssertionError("Master/layout registry IDs are not globally unique")

    def merge_table_styles(self):
        base_part = self.base.direct_rel_part("tableStyles")
        if not base_part:
            for source in self.sources.values():
                p = source.direct_rel_part("tableStyles")
                if p and len(parse(source.parts[p])):
                    raise ValueError("Root has no table-style part but source defines native styles")
            return
        target = self.mapping[(self.base.key, base_part)]
        root = parse(self.parts[target])
        known = {c.get("styleId"): structural_hash(c) for c in root}
        for source in self.sources.values():
            p = source.direct_rel_part("tableStyles")
            if not p:
                continue
            for child in parse(source.parts[p]):
                key, digest = child.get("styleId"), structural_hash(child)
                if key in known and known[key] != digest:
                    raise ValueError(f"Conflicting native table style {key} in {source.key}")
                if key not in known:
                    root.append(copy.deepcopy(child))
                    known[key] = digest
        if structural_hash(root) != structural_hash(parse(self.parts[target])):
            self.parts[target] = xml(root)

    def update_app_properties(self):
        for part, content_type in self.types.items():
            if not content_type.endswith("extended-properties+xml"):
                continue
            root = parse(self.parts[part])
            for tag in ("Slides", "Notes"):
                child = root.find("ep:" + tag, NS)
                if child is not None:
                    count = len(self.selections) if tag == "Slides" else sum("/notesSlides/" in p and p.endswith(".xml") for p in self.types)
                    child.text = str(count)
            # Remove optional stale title/category summaries from the four-slide root.
            for tag in ("HeadingPairs", "TitlesOfParts"):
                child = root.find("ep:" + tag, NS)
                if child is not None:
                    root.remove(child)
            self.parts[part] = xml(root)

    def verify_package(self):
        for part, data in self.parts.items():
            if part.endswith(".xml") or part.endswith(".rels"):
                parse(data)
            if not part.endswith(".rels"):
                continue
            if part == "_rels/.rels":
                owner = ""
            else:
                folder, filename = posixpath.split(part)
                owner = posixpath.join(posixpath.dirname(folder), filename[:-5])
            rel_ids = set()
            for rel in parse(data):
                if rel.get("Id") in rel_ids:
                    raise AssertionError(f"Duplicate relationship ID: {part}")
                rel_ids.add(rel.get("Id"))
                if rel.get("TargetMode") != "External":
                    target, _ = resolve(owner, rel.get("Target"))
                    if target not in self.parts:
                        raise AssertionError(f"Dangling relationship {part} -> {target}")
        expected = {f"ppt/slides/slide{i}.xml" for i in range(1, len(self.selections) + 1)}
        actual = {p for p, t in self.types.items() if t.endswith("presentationml.slide+xml")}
        if expected != actual:
            raise AssertionError(f"Non-canonical or extra slides: {actual ^ expected}")
        for item in self.part_audit:
            part = item["output_part"]
            item["output_sha256"] = sha(self.parts[part])
            item["bytes_identical"] = item["source_sha256"] == item["output_sha256"]
            bookkeeping_parts = {x["output_part"] for x in self.master_metadata_audit}
            if not item["bytes_identical"] and part not in expected and part not in bookkeeping_parts and part not in {self.mapping.get((self.base.key, self.base.direct_rel_part("tableStyles")))} and not item["content_type"].endswith("extended-properties+xml"):
                raise AssertionError(f"Unexpected non-slide dependency mutation: {part}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    for target in (args.output, args.report):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
    plan = json.loads(args.plan.read_text())
    sources = {}
    for key, value in plan["sources"].items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", key):
            raise ValueError(f"Unsafe source name: {key}")
        path = Path(value["path"] if isinstance(value, dict) else value)
        if not path.exists() and not path.is_absolute():
            path = args.plan.parent / path
        sources[key] = Source(key, path.resolve(strict=True))
    root_source = plan.get("root_source", "new" if "new" in sources else next(iter(sources)))
    merger = Merger(sources, root_source, plan)
    merger.compose()
    source_audit = []
    for source in sources.values():
        after = sha(source.path.read_bytes())
        if after != source.hash_before:
            raise AssertionError(f"Source file changed during merge: {source.path}")
        source_audit.append({"source": source.key, "path": str(source.path), "sha256_before": source.hash_before, "sha256_after": after, "unchanged": True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        with ZipFile(stream, "w", ZIP_DEFLATED, compresslevel=6) as z:
            for part, data in sorted(merger.parts.items()):
                z.writestr(part, data)
    report = {
        "schema_version": "flowpilotx-ooxml-slide-merge-v1",
        "plan": str(args.plan.resolve()), "plan_sha256": sha(args.plan.read_bytes()),
        "output": str(args.output.resolve()), "output_sha256": sha(args.output.read_bytes()),
        "source_root": root_source, "sources": source_audit,
        "slide_count": len(merger.selections), "package_part_count": len(merger.parts),
        "native_shapes_preserved": True, "rasterization_performed": False,
        "only_slide_changes": "Existing bottom footer text/name and page text/name; original positions and all non-footer XML structure preserved.",
        "non_footer_content_identical": all(s["non_footer_identical"] for s in merger.slide_audit),
        "all_internal_relationships_resolve": True,
        "equivalent_notes_master_aliases": merger.single_master_aliases,
        "master_id_remapping": merger.master_id_remapping,
        "master_metadata_audit": merger.master_metadata_audit,
        "master_and_layout_ids_globally_unique": True,
        "slides": merger.slide_audit, "copied_parts": merger.part_audit,
        "warnings": merger.warnings,
        "limitations": ["No PowerPoint UI inspection or visual rendering is performed by this helper.", "Duplicate source slides and links to omitted slides are rejected.", "Differing notes masters, conflicting table-style definitions and source-level embedded fonts are rejected.", "The root deck's presentation-level settings and metadata are used; imported slides retain their native layout/master/theme dependencies."],
    }
    with args.report.open("x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "slides": len(merger.selections), "source_files_unchanged": True, "non_footer_content_identical": True}))


if __name__ == "__main__":
    main()
