import json

with open("12_OLID_WL_Subtree_Engine.ipynb", "r", encoding="utf-8") as f:
    nb12 = json.load(f)

for cell in nb12["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "sig_A_check =" in source:
            # We need to change sig_A_check to use toy_G.nodes[n]['label'] instead of toy_labels
            # Or just use current_labels since it was defined earlier
            # sig_A_check = (current_labels['area'], tuple(sorted([current_labels[n] for n in toy_G.neighbors('area')])))
            source = source.replace(
                "sig_A_check = (toy_labels['area'], tuple(sorted([toy_labels[n] for n in toy_G.neighbors('area')])))",
                "sig_A_check = (current_labels['area'], tuple(sorted([current_labels[n] for n in toy_G.neighbors('area')])))"
            )
            
        lines = [line + "\n" for line in source.split("\n")]
        if not source.endswith("\n") and lines:
            lines[-1] = lines[-1][:-1]
        if not lines[-1] and len(lines) > 1:
            lines.pop()
        cell["source"] = lines

with open("12_OLID_WL_Subtree_Engine.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb12, f, indent=1)

print("Determinism check patched.")
