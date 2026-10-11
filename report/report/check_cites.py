import re, pathlib

bdir = pathlib.Path(".")
tex_files = list(bdir.glob("chapters/*.tex"))

cite_re = re.compile(r"\\cite\{([^}]*)\}")
bib_entry_re = re.compile(r"@\w+\{\s*([^,\s]+)")
ref_re = re.compile(r"\\ref\{([^}]*)\}")
label_re = re.compile(r"\\label\{([^}]*)\}")

cite_keys = set()
cite_locations = {}
for f in tex_files:
    txt = f.read_text(encoding='utf-8', errors='replace')
    for m in cite_re.finditer(txt):
        for part in m.group(1).split(','):
            part = part.strip()
            if part:
                cite_keys.add(part)
                cite_locations.setdefault(part, []).append(f.name)

bib = bdir.joinpath('references.bib').read_text(encoding='utf-8', errors='replace')
bib_entries = set(bib_entry_re.findall(bib))

print("All cite keys used:", sorted(cite_keys))
print()
print("Keys NOT in references.bib:")
for m in sorted(k for k in cite_keys if k not in bib_entries):
    print(" -", m, "used in:", cite_locations.get(m))

print()
print("Chapter refs/labels:")
alltex = bdir.joinpath("main.tex").read_text(encoding='utf-8', errors='replace')
for f in tex_files:
    alltex += f.read_text(encoding='utf-8', errors='replace')
refs = set(ref_re.findall(alltex))
labels = set(label_re.findall(alltex))
broken_refs = sorted(r for r in refs if r not in labels)
print("Labels defined:", sorted(labels))
print("Refs used:", sorted(refs))
print("Broken refs (ref with no matching label):", broken_refs)
