"""scripts/verify_accessibility_css.py"""
from pathlib import Path
import re

html = Path('backend/static/index.html').read_text(encoding='utf-8')
print("Media query @media(max-width:850px):", "@media(max-width:850px)" in html)
print("Media query @media(max-width:1100px):", "@media(max-width:1100px)" in html)
print("Media query @media(max-width:440px):", "@media(max-width:440px)" in html)
print("prefers-reduced-motion:", "@media(prefers-reduced-motion: reduce)" in html)
print(":focus-visible defined:", ":focus-visible" in html)
print("No fake 94.8%:", "94.8%" not in html)
print("No static landing gate SAFE claim:", 'id="landingGate">SAFE' not in html)
print("No static landing ground 100% claim:", 'id="landingGround">100%' not in html)

labels = re.findall(r'aria-label="([^"]+)"', html)
print(f"Accessible aria-labels ({len(labels)}):", labels)
