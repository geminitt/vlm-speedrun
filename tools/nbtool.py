"""Maintain the study notebooks as plain text.

Notebook source format: cells start with a line that is exactly "### MD" or "### CODE".

    python tools/nbtool.py extract NOTEBOOK.ipynb SRC.txt   # notebook -> text
    python tools/nbtool.py build SRC.txt NOTEBOOK.ipynb     # text -> executed notebook
    python tools/nbtool.py sync SRC.txt NOTEBOOK.ipynb      # copy prose only, keep outputs
    python tools/nbtool.py dump NOTEBOOK.ipynb              # print text outputs of code cells

Run from this folder with the project's environment: pixi run --manifest-path ../pixi.toml ...
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # the notes worktree: notebooks run from here


def parse(text):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        if line.strip() in ("### MD", "### CODE"):
            if kind:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = ("md" if "MD" in line else "code"), []
        else:
            buf.append(line)
    if kind:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def source(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def extract(nb_path, src_path):
    nb = json.loads(Path(nb_path).read_text())
    parts = [("### MD" if c["cell_type"] == "markdown" else "### CODE") + "\n" + source(c) + "\n"
             for c in nb["cells"]]
    Path(src_path).write_text("\n".join(parts))


def build(src_path, nb_path, execute=True):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, body in parse(Path(src_path).read_text()):
        nb.cells.append(nbformat.v4.new_markdown_cell(body) if kind == "md"
                        else nbformat.v4.new_code_cell(body))
    if execute:
        t = time.time()
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(HERE)}}).execute()
        print(f"executed in {time.time() - t:.0f}s")
    nbformat.write(nb, nb_path)


def sync(src_path, nb_path):
    cells = parse(Path(src_path).read_text())
    nb = json.loads(Path(nb_path).read_text())
    assert len(cells) == len(nb["cells"]), f"cell count {len(cells)} vs {len(nb['cells'])}"
    changed = 0
    for (kind, body), c in zip(cells, nb["cells"]):
        if kind == "code":
            assert c["cell_type"] == "code" and source(c) == body, "a code cell differs: rebuild instead"
        elif source(c) != body:
            c["source"] = body
            changed += 1
    Path(nb_path).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(f"updated {changed} markdown cells")


def dump(nb_path):
    nb = json.loads(Path(nb_path).read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        print(f"\n--- [{i}] {source(c).splitlines()[0][:70] if source(c) else ''}")
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                print("".join(o["text"]).rstrip()[:3000])
            elif o.get("output_type") == "error":
                print("ERROR", o["ename"], o["evalue"])
            elif "data" in o:
                print("[image]" if "image/png" in o["data"] else "".join(o["data"].get("text/plain", ""))[:500])


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"extract": extract, "build": build, "sync": sync, "dump": dump}[cmd](*args)
