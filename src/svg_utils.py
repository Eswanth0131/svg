import re
from lxml import etree

SVGNS = "http://www.w3.org/2000/svg"
EMPTY_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"></svg>'

SYSTEM_PROMPT = """You are an expert SVG illustrator. Generate a single, complete, valid SVG for a 256x256 canvas.

RULES:
- Output ONLY the SVG XML. No markdown, no explanations, no ```svg fences.
- Always start with: <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
- Always end with: </svg>
- Use simple, clean shapes. Prefer <rect>, <circle>, <ellipse>, <polygon> over complex <path> where possible.
- Fill shapes with appropriate, visible colors matching the prompt. Never use fill="" (empty fill).
- Center the composition in the 256x256 canvas with reasonable margins.
- Match the prompt semantically and visually — a coffee cup must look like a coffee cup.
- Aim for 200-350 tokens of SVG content.
"""

def _extract_svg_text(raw: str) -> str:
    """Pull the first <svg>...</svg> block from raw model output."""
    raw = raw.strip()
    raw = re.sub(r"```[a-zA-Z]*\n?", "", raw)
    start = raw.find("<svg")
    if start == -1:
        return ""
    end = raw.rfind("</svg>")
    if end != -1:
        return raw[start:end + 6]
    sc = raw.find("/>", start)
    if sc != -1:
        return raw[start:sc + 2]
    return raw[start:]

def _local_tag(el):
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    return tag.split("}")[-1] if "}" in tag else tag

def _fix_attributes(root):
    """Remove/fix bad attributes and useless elements."""
    to_remove = []
    for el in root.iter():
        el.attrib.pop("filling", None)

        if el.attrib.get("fill") == "":
            el.attrib.pop("fill")
        if el.attrib.get("fill-opacity") == "":
            el.attrib.pop("fill-opacity")

        tag = _local_tag(el)
        if tag == "path" and not el.attrib.get("d", "").strip():
            parent = el.getparent()
            if parent is not None:
                to_remove.append(el)

    for el in to_remove:
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

def _ensure_viewbox(root):
    """Ensure the SVG root has a viewBox and sensible dimensions."""
    if "viewBox" not in root.attrib:
        w = root.attrib.get("width", "256").replace("px", "")
        h = root.attrib.get("height", "256").replace("px", "")
        try:
            root.attrib["viewBox"] = f"0 0 {float(w):.0f} {float(h):.0f}"
        except ValueError:
            root.attrib["viewBox"] = "0 0 256 256"

    if "width" not in root.attrib:
        root.attrib["width"] = "256"
    if "height" not in root.attrib:
        root.attrib["height"] = "256"

def _has_visible_content(root) -> bool:
    """Return True if the SVG contains at least one shape with real geometry."""
    for el in root.iter():
        tag = _local_tag(el)

        if tag == "path" and el.attrib.get("d", "").strip():
            return True
        if tag in ("rect", "circle", "ellipse", "line"):
            return True
        if tag == "text" and (el.text or "").strip():
            return True
        if tag in ("polygon", "polyline") and el.attrib.get("points", "").strip():
            return True
    return False

def repair_svg(raw: str) -> str:
    """Extract, clean, and validate SVG from raw model output.
    Returns EMPTY_SVG if nothing usable can be recovered.
    """
    text = _extract_svg_text(raw)
    if not text:
        return EMPTY_SVG

    root = None
    for recover in (False, True):
        try:
            parser = etree.XMLParser(recover=recover)
            root = etree.fromstring(text.encode("utf-8"), parser=parser)
            break
        except Exception:
            continue

    if root is None:
        return EMPTY_SVG

    _fix_attributes(root)
    _ensure_viewbox(root)

    if not _has_visible_content(root):
        return EMPTY_SVG

    return etree.tostring(root, encoding="unicode")

def extract_basic_features(svg: str) -> dict:
    return {
        "n_paths": len(re.findall(r"<path\b", svg)),
        "n_rects": len(re.findall(r"<rect\b", svg)),
        "n_circles": len(re.findall(r"<circle\b", svg)),
        "is_empty": svg == EMPTY_SVG,
        "char_len": len(svg),
    }

def build_user_prompt(prompt: str, retrieved: list) -> str:
    parts = []
    if retrieved:
        parts.append("Here are similar SVG examples for reference:")
        for i, r in enumerate(retrieved[:2], 1):
            snippet = r["svg"][:500] + ("..." if len(r["svg"]) > 500 else "")
            parts.append(f"Example {i} prompt: {r['prompt']}")
            parts.append(f"Example {i} SVG:\n{snippet}")
            parts.append("")
    parts.append(f"Now generate an SVG for this prompt: {prompt}")
    parts.append("Output only the SVG XML, starting with <svg and ending with </svg>.")
    return "\n".join(parts)
