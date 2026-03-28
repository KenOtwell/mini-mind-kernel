"""Quick peek at an NPC template entry from ChIM's SQL dump."""
import re
import sys

npc_name = sys.argv[1] if len(sys.argv) > 1 else "belethor"
sql_path = r"C:\Users\Ken\Projects\many-mind-kernel\docs\Dwemer Distro\var\www\html\HerikaServer\data\npc_templates_20250302001.sql"

with open(sql_path, "r", encoding="utf-8") as f:
    text = f.read()

# Match: INSERT INTO public.npc_templates VALUES ('name', 'bio', 'tags', '', '', 'voice', '');
pattern = rf"INSERT INTO public\.npc_templates VALUES \('{re.escape(npc_name)}',\s*'((?:[^']|'')*)',\s*'((?:[^']|'')*)',\s*'((?:[^']|'')*)',\s*'((?:[^']|'')*)',\s*'((?:[^']|'')*)',\s*'((?:[^']|'')*)',\s*'((?:[^']|'')*)'\)"
m = re.search(pattern, text, re.DOTALL)

if not m:
    print(f"NPC '{npc_name}' not found.")
    sys.exit(1)

bio = m.group(1).replace("''", "'")
tags = m.group(2).replace("''", "'")
voice = m.group(6).replace("''", "'")

print(f"=== NPC: {npc_name} ===")
print(f"Tags: {tags}")
print(f"Voice: {voice}")
print(f"\n--- Bio ---")
print(bio)
