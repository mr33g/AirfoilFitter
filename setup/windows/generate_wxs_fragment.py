import os
import uuid
import hashlib
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom

BASE_EXCLUDED_DIRS = {
    '__pycache__',
    '.git',
    '.vscode',
    '.idea',
    'build',
    'dist',
    '.claude',
    'res',
    'out',
    'doc',
}

EXCLUDED_FILES = {
    '.DS_Store',
    'Thumbs.db',
    '*cwd'
}

def make_id(prefix, path):
    raw = path.replace(os.sep, "_")
    safe_chars = []
    for ch in raw:
        if ch.isalnum() or ch in "._":
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe_path = "".join(safe_chars)
    if not safe_path or not (safe_path[0].isalpha() or safe_path[0] == "_"):
        safe_path = f"_{safe_path}"
    full_id = f"{prefix}_{safe_path}"
    if len(full_id) <= 72:
        return full_id
    hash_str = hashlib.md5(path.encode()).hexdigest()[:16]
    return f"{prefix}_{hash_str}"


def should_exclude_dir(dirpath, source_dir, excluded_dirs):
    rel_path = os.path.relpath(dirpath, source_dir)
    parts = rel_path.split(os.sep)
    return any(part in excluded_dirs for part in parts)


def generate_wxs_fragment(source_dir, output_file, component_group_id, directory_id_root):
    setup_dir = os.path.dirname(os.path.abspath(output_file))
    source_dir_abs = os.path.abspath(source_dir)
    try:
        source_dir_rel = os.path.relpath(source_dir_abs, setup_dir).replace(os.sep, '/')
        if not source_dir_rel.startswith('..') and not source_dir_rel.startswith('.'):
            source_dir_rel = f"./{source_dir_rel}"
    except ValueError:
        source_dir_rel = source_dir_abs.replace(os.sep, '/')

    excluded_dirs = BASE_EXCLUDED_DIRS.copy()

    root = ET.Element("Wix", xmlns="http://wixtoolset.org/schemas/v4/wxs")
    fragment = ET.SubElement(root, "Fragment")

    dir_map = {".": directory_id_root, "": directory_id_root}
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(source_dir):
        if should_exclude_dir(dirpath, source_dir, excluded_dirs):
            continue
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        rel_path = os.path.relpath(dirpath, source_dir)
        if rel_path in [".", ""]:
            continue
        dir_id = make_id("DIR", rel_path)
        dir_map[rel_path] = dir_id

    comp_group = ET.SubElement(fragment, "ComponentGroup", Id=component_group_id)

    for dirpath, dirnames, filenames in os.walk(source_dir):
        if should_exclude_dir(dirpath, source_dir, excluded_dirs):
            continue
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        rel_path = os.path.relpath(dirpath, source_dir)
        dir_id = dir_map.get(rel_path, directory_id_root)

        for filename in filenames:
            if filename in EXCLUDED_FILES:
                continue
            file_path = os.path.join(dirpath, filename)
            rel_file_path = os.path.relpath(file_path, source_dir)

            comp_id = make_id("CMP", rel_file_path)
            file_id = make_id("FILE", rel_file_path)

            comp = ET.SubElement(comp_group, "Component",
                                 Id=comp_id,
                                 Guid=str(uuid.uuid4()).upper(),
                                 Directory=dir_id)

            ET.SubElement(comp, "RegistryValue",
                          Root="HKCU",
                          Key=f"Software\\AirfoilFitter\\Files\\{file_id}",
                          Name="installed",
                          Type="integer",
                          Value="1",
                          Action="write")

            file_source_path = f"{source_dir_rel}/{rel_file_path.replace(os.sep, '/')}"
            file_source_path = file_source_path.replace('\\', '/').replace('//', '/')
            ET.SubElement(comp, "File",
                          Id=file_id,
                          Source=file_source_path,
                          KeyPath="yes")
            file_count += 1

    dir_fragment = ET.SubElement(root, "Fragment")

    sorted_dir_items = sorted(dir_map.items(), key=lambda x: x[0].count(os.sep) if os.sep in x[0] else x[0].count("/"))
    for rel_path, dir_id in sorted_dir_items:
        if rel_path in [".", ""]:
            continue
        parent_rel = os.path.dirname(rel_path)
        parent_id = dir_map.get(parent_rel, directory_id_root)
        p_ref = ET.SubElement(dir_fragment, "DirectoryRef", Id=parent_id)
        ET.SubElement(p_ref, "Directory", Id=dir_id, Name=os.path.basename(rel_path))

    xml_str = ET.tostring(root, encoding='unicode')
    xml_dom = minidom.parseString(xml_str)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_dom.toprettyxml(indent="    "))

    print(f"Generated {output_file} with {file_count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate WiX fragment file for installer")
    parser.add_argument("--output", default="Files.wxs", help="Output file name (default: Files.wxs)")
    parser.add_argument("--source-dir", default="..", help="Source directory to scan for files (default: ..)")
    args = parser.parse_args()

    generate_wxs_fragment(
        source_dir=args.source_dir,
        output_file=args.output,
        component_group_id="HarvestedAppFiles",
        directory_id_root="APPLICATIONFOLDER",
    )
