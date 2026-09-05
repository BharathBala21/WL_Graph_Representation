import json
import re

# ------------- Fix Notebook 12 -------------
with open("12_OLID_WL_Subtree_Engine.ipynb", "r", encoding="utf-8") as f:
    nb12 = json.load(f)

for cell in nb12["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Fix node label in transform_single
        if "G.nodes[node]['label'] = label_row[node]" in source:
            source = source.replace("G.nodes[node]['label'] = label_row[node]", 
                                    "G.nodes[node]['label'] = f\"{node}::{label_row[node]}\"")
            
        # Fix manual example nodes and labels
        if "toy_G = nx.path_graph(['A', 'B', 'C', 'D'])" in source:
            source = source.replace(
                "toy_G = nx.path_graph(['A', 'B', 'C', 'D'])",
                "toy_G = nx.path_graph(['area', 'perimeter', 'green_ratio', 'glcm_contrast'])"
            )
            source = source.replace(
                "toy_labels = {'A': 'Low', 'B': 'High', 'C': 'High', 'D': 'Medium'}",
                "toy_labels = {'area': 'Low', 'perimeter': 'High', 'green_ratio': 'High', 'glcm_contrast': 'Medium'}"
            )
            source = source.replace(
                "toy_G.nodes[n]['label'] = toy_labels[n]",
                "toy_G.nodes[n]['label'] = f\"{n}::{toy_labels[n]}\""
            )
            source = source.replace("'A'", "'area'").replace("'B'", "'perimeter'").replace("'C'", "'green_ratio'").replace("'D'", "'glcm_contrast'")
            source = source.replace("A and B", "area and perimeter")
            source = source.replace("B and C", "perimeter and green_ratio")
            source = source.replace("C and D", "green_ratio and glcm_contrast")
            source = source.replace("A and D", "area and glcm_contrast")
            source = source.replace("A != B", "area != perimeter")
            source = source.replace("B != C", "perimeter != green_ratio")
            source = source.replace("C != D", "green_ratio != glcm_contrast")
            source = source.replace("A != D", "area != glcm_contrast")
            source = source.replace("A re-queried", "area re-queried")
            
        # Fix Sanity Checks to include area::High != green_ratio::High
        if "assert area_label in ['Low', 'Medium', 'High']" in source:
            source = source.replace(
                "assert area_label in ['Low', 'Medium', 'High'],         f\"Invalid label for 'area' in graph {i}: {area_label}\"",
                "assert area_label in ['area::Low', 'area::Medium', 'area::High'], f\"Invalid label for 'area' in graph {i}: {area_label}\""
            )
            source = source.replace(
                "assert set(G.edges()) == edges_ref, f\"Edge mismatch in graph {i}!\"",
                "assert set(G.edges()) == edges_ref, f\"Edge mismatch in graph {i}!\"\n    if 'green_ratio' in G.nodes():\n        assert G.nodes['area']['label'] != G.nodes['green_ratio']['label'], \"Labels for different features must be distinct\""
            )
            
        # Update source back to cell
        # json expects a list of strings where each string is a line except it can also just be a string. nbformat handles string lists.
        # But we can just write it as a single string. Wait, nbformat expects list of lines with \n.
        # Let's split by newline and keep the \n.
        lines = [line + "\n" for line in source.split("\n")]
        # Remove trailing newline from the last element if the original string didn't end with one.
        if not source.endswith("\n"):
            lines[-1] = lines[-1][:-1]
        if not lines[-1]:
            lines.pop()
        cell["source"] = lines

with open("12_OLID_WL_Subtree_Engine.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb12, f, indent=1)


# ------------- Fix Notebook 15 -------------
with open("15_OLID_WL_Representation_Diagnosis.ipynb", "r", encoding="utf-8") as f:
    nb15 = json.load(f)

for cell in nb15["cells"]:
    # Check both code and markdown
    source = "".join(cell["source"])
    
    # 1. strategy='uniform' -> strategy='quantile'
    source = source.replace("strategy='uniform'", "strategy='quantile'")
    
    # 2. Correct interpretation of degree correlation
    source = source.replace("Spearman correlation between graph degree and discriminative score is weak", 
                            "There is a moderate positive association between correlation-graph degree and the univariate F-statistic")
    source = source.replace("weak relationship with univariate discriminative power", 
                            "moderate positive association with univariate discriminative power")
    source = source.replace("correlates weakly with univariate discriminative power", 
                            "has a moderate positive association with univariate discriminative power")
    source = source.replace("Topologies may simply encode feature correlation (redundancy) rather than synergistic class-discriminative interactions", 
                            "The moderate positive association suggests graph degree captures discriminative information.")
    
    # Update H2 from Supported to Inconclusive
    if "H2:" in source and "'Status': 'Supported'" in source:
        source = source.replace("'Status': 'Supported'", "'Status': 'Inconclusive'")
        
    lines = [line + "\n" for line in source.split("\n")]
    if not source.endswith("\n") and lines:
        lines[-1] = lines[-1][:-1]
    if not lines[-1] and len(lines) > 1:
        lines.pop()
    cell["source"] = lines

with open("15_OLID_WL_Representation_Diagnosis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb15, f, indent=1)

print("Notebooks patched successfully.")
