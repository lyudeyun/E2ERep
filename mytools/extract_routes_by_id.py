#!/usr/bin/env python3
"""Copy selected ``<route>`` elements from one routes XML into a smaller XML.

Route id is the number in names like ``RouteScenario_<id>_rep0_...``.
Pass ``--ids`` for the subset (defaults are two example ids only).
"""
import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

# Placeholder defaults only; pass --ids for a real slice.
DEFAULT_ROUTE_IDS = [2086, 2129]

_REPO_ROOT = Path(__file__).resolve().parents[1]

def _indent_compat(elem, level=0, space="   "):
    """Indent XML for pretty printing.

    Python 3.9+ provides ``xml.etree.ElementTree.indent``; older versions do not.
    """
    i = "\n" + level * space
    if len(elem):
        if not (elem.text and elem.text.strip()):
            elem.text = i + space
        for child in elem:
            _indent_compat(child, level + 1, space=space)
        if not (elem.tail and elem.tail.strip()):
            elem.tail = i
    else:
        if level and not (elem.tail and elem.tail.strip()):
            elem.tail = i


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-s",
        "--source",
        default=str(_REPO_ROOT / "leaderboard/data/bench2drive220.xml"),
        help="Source routes XML",
    )
    p.add_argument(
        "-o",
        "--output",
        default=str(_REPO_ROOT / "leaderboard/data/bench2drive220_route_subset.xml"),
        help="Output routes XML",
    )
    p.add_argument(
        "--ids",
        type=int,
        nargs="*",
        default=None,
        help="Route ids to extract (default: two example ids only)",
    )
    args = p.parse_args()
    ids = args.ids if args.ids is not None else DEFAULT_ROUTE_IDS
    want = {str(i) for i in ids}

    tree = ET.parse(args.source)
    root = tree.getroot()
    by_id = {}
    for route in root.findall("route"):
        rid = route.get("id")
        if rid in want:
            by_id[rid] = route

    missing = sorted(want - set(by_id), key=int)
    if missing:
        raise SystemExit(f"Missing route ids in source: {missing}")

    out = ET.Element("routes")
    for i in ids:
        rid = str(i)
        out.append(copy.deepcopy(by_id[rid]))

    new_tree = ET.ElementTree(out)
    # Pretty print: ET.indent exists only in Python 3.9+
    if hasattr(ET, "indent"):
        ET.indent(new_tree, space="   ")
    else:
        _indent_compat(out, space="   ")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<!-- Route subset by id. -->\n")
        new_tree.write(f, encoding="unicode", xml_declaration=False)


if __name__ == "__main__":
    main()
