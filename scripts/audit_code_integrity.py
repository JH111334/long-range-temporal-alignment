import re
from pathlib import Path

target_scripts = [
    Path('scripts/run_stage03_targeted_arms4_5_modal.py'),
    Path('scripts/run_stage03_targeted_arms4_5_5epochs_modal.py'),
]

suspicious = [
    'is_correct = 1 if',
    'pred = gold',
    'if cid in',
    'if example_id in',
    '32.03',
    '27.00',
    '8.20',
]

for p in target_scripts:
    print(f"\n=== FORENSIC CODE AUDIT: {p} ===")
    code = p.read_text(encoding='utf-8')
    for s in suspicious:
        matches = [m.start() for m in re.finditer(re.escape(s), code)]
        print(f'Pattern "{s}": {len(matches)} occurrences')
        assert len(matches) == 0, f"Suspicious pattern '{s}' found in {p}!"

print("\nALL TARGETED MODAL SCRIPTS AUDITED: 0 SUSPICIOUS PATTERNS DETECTED.")
