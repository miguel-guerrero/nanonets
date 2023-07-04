from graphviz import Digraph
import re


# see https://github.com/karpathy/micrograd/blob/master/trace_graph.ipynb


def _trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.inputs:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def clean_arr(arr):
    return f"{arr.shape}"


def colorByLayerName(name, defColor="black", defStyle="default"):
    x = re.match(r"\[.*\] L(\d+)_", name)
    if x:
        try:
            n = int(x.group(1))
            colors = ["red", "aquamarine", "green", "cyan"]
            return colors[n % len(colors)], "filled"
        except ValueError:
            pass
    return defColor, defStyle


def draw_dot(root, filename="out.gv", format="svg", rankdir="TB"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = _trace(root)
    dot = Digraph(
        format=format, graph_attr={"rankdir": rankdir}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        # TODO for debug
        # grad = "" if n._pass_grad is None else f"| grad:\\n{clean_arr(n._pass_grad)} "
        grad = "" if n._grad is None else f"| grad: {clean_arr(n.grad)} "
        if n.name == "":
            label = "{ data: %s %s}" % (clean_arr(n.data), grad)
        else:
            label = "{ %s | data: %s %s}" % (n.name, clean_arr(n.data), grad)
        dot.node(
            name=str(id(n)),
            label=label,
            shape="record",
            color="grey80" if n.needs_grad() else "grey50",
            style="filled",
        )
        if n._op:
            color, style = colorByLayerName(n._op, "yellow", "filled")
            dot.node(
                name=str(id(n)) + n._op,
                label=n._op,
                color=color,
                style=style,
            )
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    with open(filename, "w") as fw:
        fw.write(dot.source)

    return dot
