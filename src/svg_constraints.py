
from lxml import etree

ALLOWED_TAGS = {
    "svg","g","path","rect","circle","ellipse","line",
    "polyline","polygon","defs","use","symbol","clipPath",
    "mask","linearGradient","radialGradient","stop","text",
    "tspan","title","desc","style","pattern","marker","filter"
}

def _strip_namespace(tag):
    if tag is None or not isinstance(tag, str):
        return ""
    return tag.split("}")[-1]

def prune_disallowed(root):
    for el in list(root.iter()):
        tag = _strip_namespace(el.tag)

        if tag == "":
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
            continue

        if tag not in ALLOWED_TAGS:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    return root

def repair_svg(svg_string):
    try:
        if svg_string is None:
            return '<svg width="256" height="256" viewBox="0 0 256 256"></svg>'
        svg_string = str(svg_string)

        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(svg_string.encode("utf-8"), parser=parser)

        if root is None:
            return '<svg width="256" height="256" viewBox="0 0 256 256"></svg>'

        root = prune_disallowed(root)

        # Ensure root is svg
        if _strip_namespace(root.tag) != "svg":
            new_root = etree.Element("svg")
            new_root.append(root)
            root = new_root

        # Normalize canvas attrs
        if "width" not in root.attrib:
            root.set("width", "256")
        if "height" not in root.attrib:
            root.set("height", "256")
        if "viewBox" not in root.attrib:
            root.set("viewBox", "0 0 256 256")

        return etree.tostring(root, encoding="unicode")
    except Exception:
        return '<svg width="256" height="256" viewBox="0 0 256 256"></svg>'

def extract_basic_features(svg_string):
    """
    Returns exact feature names expected:
    svg_len, num_paths, num_circles, num_rects, num_groups
    """
    try:
        if svg_string is None:
            raise ValueError("svg_string is None")

        svg_string = str(svg_string)
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(svg_string.encode("utf-8"), parser=parser)

        if root is None:
            raise ValueError("parsed root is None")

        tags = [_strip_namespace(el.tag) for el in root.iter()]
        tags = [t for t in tags if t]

        return {
            "svg_len": len(svg_string),
            "num_paths": tags.count("path"),
            "num_circles": tags.count("circle"),
            "num_rects": tags.count("rect"),
            "num_groups": tags.count("g"),
        }

    except Exception:
        return {
            "svg_len": 0,
            "num_paths": 0,
            "num_circles": 0,
            "num_rects": 0,
            "num_groups": 0,
        }
