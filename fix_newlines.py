with open('full_pipeline/decomposition/21_source.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace any print("\n... with print("\n...
import re
# If a line ends with print(" or print(f" and next line starts with "
text = text.replace('print("\n"\n', 'print("\\n"\n')
text = text.replace('print(f"\n"\n', 'print(f"\\n"\n')
text = text.replace('print("\n"', 'print("\\n"')
text = text.replace('print(f"\n"', 'print(f"\\n"')
text = text.replace('print("\n', 'print("\\n')
text = text.replace('print(f"\n', 'print(f"\\n')

with open('full_pipeline/decomposition/21_source.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Replacement complete.")

