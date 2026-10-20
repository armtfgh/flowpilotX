"""Share byte-identical PPTX image parts while preserving original artwork.

This changes package relationships only. It does not rasterize, edit or replace
image contents, slide XML, table values, notes or native connector geometry.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import posixpath
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED


def compact(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with ZipFile(source) as archive:
        assert archive.testzip() is None
        entries = {n: archive.read(n) for n in archive.namelist()}
    groups = defaultdict(list)
    for name, data in entries.items():
        if name.startswith('ppt/media/'):
            groups[(PurePosixPath(name).suffix, sha256(data).hexdigest())].append(name)
    replacements = {}
    for names in groups.values():
        first, *duplicates = sorted(names)
        replacements.update({name: first for name in duplicates})
    changed = {}
    relation_count = 0
    for name, data in entries.items():
        if not name.endswith('.rels'):
            continue
        xml = ET.fromstring(data)
        base = str(PurePosixPath(name).parent.parent)
        edits = 0
        for relation in xml:
            if relation.get('TargetMode') == 'External':
                continue
            target = relation.get('Target', '')
            absolute = posixpath.normpath(target.lstrip('/') if target.startswith('/') else posixpath.join(base, target))
            if absolute in replacements:
                replacement = replacements[absolute]
                relation.set('Target', '/' + replacement if target.startswith('/') else posixpath.relpath(replacement, base))
                edits += 1
        if edits:
            changed[name] = ET.tostring(xml, encoding='utf-8', xml_declaration=True)
            relation_count += edits
    types = ET.fromstring(entries['[Content_Types].xml'])
    removed_types = 0
    for item in list(types):
        if item.get('PartName', '').lstrip('/') in replacements:
            types.remove(item)
            removed_types += 1
    if removed_types:
        changed['[Content_Types].xml'] = ET.tostring(types, encoding='utf-8', xml_declaration=True)
    with ZipFile(destination, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in entries.items():
            if name not in replacements:
                archive.writestr(name, changed.get(name, data))
    unchanged = 0
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        remaining = set(archive.namelist())
        for name, data in entries.items():
            if name not in replacements and name not in changed:
                assert archive.read(name) == data
                unchanged += 1
        for name in remaining:
            if name.endswith('.rels'):
                base = str(PurePosixPath(name).parent.parent)
                for rel in ET.fromstring(archive.read(name)):
                    if rel.get('TargetMode') == 'External':
                        continue
                    target = rel.get('Target', '')
                    absolute = posixpath.normpath(target.lstrip('/') if target.startswith('/') else posixpath.join(base, target))
                    assert absolute in remaining, (name, absolute)
    return {'source':str(source),'output':str(destination),'source_sha256':sha256(source.read_bytes()).hexdigest(),
            'output_sha256':sha256(destination.read_bytes()).hexdigest(),'original_media_parts':sum(map(len,groups.values())),
            'unique_media_parts':len(groups),'duplicate_parts_removed':len(replacements),'relationships_updated':relation_count,
            'other_parts_byte_identical':unchanged,'source_bytes':source.stat().st_size,'output_bytes':destination.stat().st_size,
            'retained_media_sha256':{min(names):key[1] for key,names in groups.items()},'image_bytes_changed':False,
            'slide_content_changed':False,'internal_relationship_targets_verified':True}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('output');parser.add_argument('--report',required=True)
    args=parser.parse_args()
    result=compact(args.source,args.output)
    with Path(args.report).open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
