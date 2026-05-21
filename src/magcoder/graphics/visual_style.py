"""Visual style constants for standardized MAG DOT rendering."""

GRAPH_ATTRIBUTES: dict[str, str] = {
    "rankdir": "LR",
    "labelloc": "t",
    "fontname": "Helvetica",
    "bgcolor": "white",
}

NODE_ATTRIBUTES: dict[str, str] = {
    "shape": "box",
    "style": "rounded,filled",
    "fillcolor": "#eef6ff",
    "color": "#2f5f8f",
    "fontname": "Helvetica",
}

EDGE_ATTRIBUTES: dict[str, str] = {
    "color": "#44546a",
    "fontname": "Helvetica",
    "fontsize": "10",
}
