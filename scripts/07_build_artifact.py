"""
Step 7: render REPORT.md into a self-contained HTML page with embedded figures.

Everything is inlined (images as data URIs, no external CSS/JS/fonts) so the page
renders under a strict content-security policy.
"""
import base64
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import markdown

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
OUT = ROOT / "report_artifact.html"

# ---------------------------------------------------------------------------
# LaTeX -> HTML (the small subset this report actually uses)
# ---------------------------------------------------------------------------
GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\sigma": "σ", r"\rho": "ρ",
    r"\mu": "μ", r"\lambda": "λ", r"\varepsilon": "ε", r"\epsilon": "ε",
    r"\Phi": "Φ", r"\Delta": "Δ", r"\pi": "π", r"\tau": "τ",
}
OPS = {
    r"\approx": "≈", r"\le": "≤", r"\ge": "≥", r"\ne": "≠", r"\propto": "∝",
    r"\times": "×", r"\cdot": "·", r"\in": "∈", r"\pm": "±", r"\to": "→",
    r"\lfloor": "⌊", r"\rfloor": "⌋", r"\ldots": "…", r"\infty": "∞",
}


def _take_braced(s: str, i: int) -> tuple[str, int]:
    """Return the contents of the {...} group starting at s[i] == '{'."""
    assert s[i] == "{"
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def tex(s: str) -> str:
    """Convert a LaTeX fragment to HTML. Handles the constructs used in this report."""
    out, i = [], 0
    while i < len(s):
        ch = s[i]

        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m:
                cmd, ln = "\\" + m.group(1), m.end()
                nxt = i + ln

                if cmd in (r"\frac", r"\tfrac", r"\dfrac"):
                    while nxt < len(s) and s[nxt] == " ":
                        nxt += 1
                    num, nxt = _take_braced(s, nxt)
                    while nxt < len(s) and s[nxt] == " ":
                        nxt += 1
                    den, nxt = _take_braced(s, nxt)
                    out.append(f'<span class="frac"><span class="fnum">{tex(num)}</span>'
                               f'<span class="fden">{tex(den)}</span></span>')
                    i = nxt
                    continue

                if cmd == r"\sqrt":
                    while nxt < len(s) and s[nxt] == " ":
                        nxt += 1
                    if nxt < len(s) and s[nxt] == "{":
                        arg, nxt = _take_braced(s, nxt)
                    else:
                        arg, nxt = s[nxt], nxt + 1
                    out.append(f'<span class="sqrt">{tex(arg)}</span>')
                    i = nxt
                    continue

                if cmd in (r"\text", r"\operatorname", r"\mathrm", r"\textbf"):
                    while nxt < len(s) and s[nxt] == " ":
                        nxt += 1
                    arg, nxt = _take_braced(s, nxt)
                    out.append(f'<span class="up">{arg}</span>')
                    i = nxt
                    continue

                if cmd == r"\widehat":
                    while nxt < len(s) and s[nxt] == " ":
                        nxt += 1
                    arg, nxt = _take_braced(s, nxt)
                    out.append(f'<span class="hat">{tex(arg)}</span>')
                    i = nxt
                    continue

                if cmd in (r"\left", r"\right", r"\big", r"\bigg", r"\Big", r"\Bigg"):
                    i = nxt
                    continue
                if cmd in (r"\quad", r"\qquad"):
                    out.append('<span class="gap"></span>')
                    i = nxt
                    continue
                if cmd in (r"\,", r"\;", r"\!", r"\:"):
                    i = nxt
                    continue
                if cmd in (r"\min", r"\max", r"\log", r"\exp", r"\sum", r"\prod"):
                    sym = {r"\sum": "∑", r"\prod": "∏"}.get(cmd)
                    out.append(sym if sym else f'<span class="up">{cmd[1:]}</span>')
                    i = nxt
                    continue
                if cmd in GREEK:
                    out.append(GREEK[cmd])
                    i = nxt
                    continue
                if cmd in OPS:
                    out.append(OPS[cmd])
                    i = nxt
                    continue
                out.append(m.group(1))
                i = nxt
                continue
            # escaped punctuation
            out.append(s[i + 1] if i + 1 < len(s) else "")
            i += 2
            continue

        if ch in "_^":
            tag = "sub" if ch == "_" else "sup"
            j = i + 1
            while j < len(s) and s[j] == " ":
                j += 1
            if j < len(s) and s[j] == "{":
                arg, j = _take_braced(s, j)
            elif j < len(s):
                arg, j = s[j], j + 1
            else:
                arg = ""
            out.append(f"<{tag}>{tex(arg)}</{tag}>")
            i = j
            continue

        if ch in "{}":
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Figure embedding
# ---------------------------------------------------------------------------
def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# Captions are italic paragraphs that may wrap over several lines, so the pattern
# spans newlines and terminates at the closing '*' before a blank line.
FIGURE_LINE = re.compile(
    r"\*Figure\s+(\d+)\s*—\s*`?([^`\n]*?\.png)`?\.?((?:[^*]|\n)*?)\*(?=\s*(?:\n\s*\n|\n#|\Z))",
    re.MULTILINE,
)


def embed_figures(md: str) -> str:
    def repl(m):
        num, ref, rest = m.group(1), m.group(2).strip(), m.group(3).strip()
        name = Path(ref).name
        p = FIGS / name
        if not p.exists():
            return m.group(0)
        cap = " ".join(rest.lstrip(".").split())
        cap_html = f'<span class="cap-body">{cap}</span>' if cap else ""
        return (f'<figure class="fig">\n<img src="{data_uri(p)}" alt="Figure {num}">\n'
                f'<figcaption><span class="cap-num">Figure {num}</span>{cap_html}'
                f'<code class="cap-file">{name}</code></figcaption>\n</figure>')
    return FIGURE_LINE.sub(repl, md)


def append_unreferenced(md: str) -> str:
    """Any figure not cited inline is appended as a gallery so nothing is orphaned."""
    used = {Path(r).name for r in re.findall(r"([0-9]{2}_[a-z_]+\.png)", md)}
    rest = sorted(p for p in FIGS.glob("*.png") if p.name not in used)
    if not rest:
        return md
    blocks = ["\n\n---\n\n## Appendix C — Additional figures\n"]
    for p in rest:
        title = p.stem.split("_", 1)[1].replace("_", " ").capitalize()
        blocks.append(
            f'<figure class="fig">\n<img src="{data_uri(p)}" alt="{title}">\n'
            f'<figcaption><span class="cap-num">{title}</span>'
            f'<code class="cap-file">{p.name}</code></figcaption>\n</figure>\n')
    return md + "\n".join(blocks)


# ---------------------------------------------------------------------------
# Math extraction
# ---------------------------------------------------------------------------
def convert_math(md: str) -> tuple[str, dict]:
    store, n = {}, 0

    md = md.replace(r"\$", "\x00ESCDOLLAR\x00")

    def block(m):
        nonlocal n
        n += 1
        key = f"MATHBLOCK{n}ZZ"
        store[key] = f'<div class="math-block">{tex(m.group(1).strip())}</div>'
        return f"\n\n{key}\n\n"

    md = re.sub(r"\$\$(.+?)\$\$", block, md, flags=re.DOTALL)

    def inline(m):
        nonlocal n
        n += 1
        key = f"MATHINLINE{n}ZZ"
        store[key] = f'<span class="math">{tex(m.group(1))}</span>'
        return key

    md = re.sub(r"\$([^$\n]+?)\$", inline, md)
    md = md.replace("\x00ESCDOLLAR\x00", "$")
    return md, store


# ---------------------------------------------------------------------------
# Verdict chips
# ---------------------------------------------------------------------------
def verdict_chips(html: str) -> str:
    html = html.replace("<td>✗</td>", '<td><span class="chip fail">Rejected</span></td>')
    html = html.replace("<td>✓</td>", '<td><span class="chip pass">Held</span></td>')
    return html


def build_toc(html: str) -> str:
    items = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', html)
    if not items:
        return ""
    lis = "\n".join(
        f'<li><a href="#{i}">{re.sub("<.*?>", "", t)}</a></li>' for i, t in items)
    return f'<nav class="toc" aria-label="Contents"><h2>Contents</h2><ol>{lis}</ol></nav>'


CSS = """
:root{
  color-scheme: light;
  --ground:#fbfbfc; --panel:#f4f5f8; --ink:#12151c; --ink-2:#3d4453; --muted:#5a6272;
  --rule:#e2e5ea; --rule-strong:#c9cfd9;
  --accent:#1f5fa8; --accent-soft:#e8f0fa;
  --fail:#b23636; --fail-soft:#fbeeee;
  --pass:#1c7a52; --pass-soft:#e9f5ef;
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --ground:#0e1116; --panel:#161b23; --ink:#e9ecf2; --ink-2:#c3c9d6; --muted:#98a1b3;
    --rule:#242a35; --rule-strong:#39414f;
    --accent:#6ba4e8; --accent-soft:#16283d;
    --fail:#e57373; --fail-soft:#2c1a1a;
    --pass:#5cc79a; --pass-soft:#13291f;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#0e1116; --panel:#161b23; --ink:#e9ecf2; --ink-2:#c3c9d6; --muted:#98a1b3;
  --rule:#242a35; --rule-strong:#39414f;
  --accent:#6ba4e8; --accent-soft:#16283d;
  --fail:#e57373; --fail-soft:#2c1a1a;
  --pass:#5cc79a; --pass-soft:#13291f;
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font-family:var(--serif); font-size:17px; line-height:1.68;
  margin:0; padding:0 1.5rem 6rem;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:70ch; margin:0 auto}
.wide{max-width:104ch; margin-inline:auto}

/* masthead */
.masthead{
  max-width:104ch; margin:0 auto; padding:4.5rem 0 2rem;
  border-bottom:2px solid var(--rule-strong);
}
.eyebrow{
  font-family:var(--mono); font-size:.7rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--muted); margin:0 0 1.1rem;
}
h1{
  font-size:clamp(2rem,4.6vw,3.1rem); line-height:1.12; font-weight:600;
  letter-spacing:-.02em; margin:0 0 1rem; text-wrap:balance; max-width:22ch;
}
.standfirst{font-size:1.12rem; color:var(--ink-2); max-width:60ch; margin:0 0 2rem}
.meta{
  display:flex; flex-wrap:wrap; gap:.6rem 2.2rem;
  font-family:var(--mono); font-size:.76rem; color:var(--muted);
}
.meta b{color:var(--ink-2); font-weight:600}

/* verdict banner - the page's one bold move */
.verdict{
  max-width:104ch; margin:2.5rem auto 0; display:flex; align-items:stretch;
  border:1px solid var(--fail); border-left-width:5px; background:var(--fail-soft);
  border-radius:3px; overflow:hidden;
}
.verdict-stamp{
  font-family:var(--mono); font-size:.72rem; letter-spacing:.18em; text-transform:uppercase;
  color:var(--fail); font-weight:700; padding:1.2rem 1.4rem;
  border-right:1px solid color-mix(in srgb, var(--fail) 30%, transparent);
  display:flex; align-items:center; white-space:nowrap;
}
.verdict-body{padding:1.1rem 1.4rem; font-size:.97rem; color:var(--ink-2)}
.verdict-body strong{color:var(--ink)}

h2{
  font-size:1.62rem; font-weight:600; letter-spacing:-.01em; line-height:1.25;
  margin:3.6rem 0 1rem; padding-top:1.5rem; border-top:1px solid var(--rule);
  text-wrap:balance;
}
h3{font-size:1.18rem; font-weight:600; margin:2.4rem 0 .7rem; text-wrap:balance}
h4{font-size:1rem; font-weight:600; margin:1.8rem 0 .5rem; color:var(--ink-2)}
p{margin:0 0 1.05rem}
strong{font-weight:600}
a{color:var(--accent); text-decoration:none; border-bottom:1px solid color-mix(in srgb,var(--accent) 35%,transparent)}
a:hover{border-bottom-color:var(--accent)}
a:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:2px}
hr{border:0; border-top:1px solid var(--rule); margin:3rem 0}
ul,ol{margin:0 0 1.1rem; padding-left:1.4rem}
li{margin:.35rem 0}
li::marker{color:var(--muted)}
blockquote{
  margin:1.6rem 0; padding:.2rem 0 .2rem 1.3rem;
  border-left:3px solid var(--accent); color:var(--ink-2);
}
blockquote p:last-child{margin-bottom:0}

/* table of contents */
.toc{
  max-width:104ch; margin:3rem auto 0; padding:1.5rem 1.8rem;
  background:var(--panel); border:1px solid var(--rule); border-radius:3px;
}
.toc h2{
  font-family:var(--mono); font-size:.72rem; letter-spacing:.16em; text-transform:uppercase;
  color:var(--muted); margin:0 0 .9rem; padding:0; border:0; font-weight:600;
}
.toc ol{
  columns:2; column-gap:2.5rem; margin:0; padding:0; list-style:none;
  counter-reset:toc; font-family:var(--mono); font-size:.83rem;
}
.toc li{margin:.3rem 0; break-inside:avoid}
.toc a{border:0; color:var(--ink-2)}
.toc a:hover{color:var(--accent)}
@media(max-width:640px){.toc ol{columns:1}}

/* code */
code{
  font-family:var(--mono); font-size:.86em; background:var(--panel);
  padding:.12em .4em; border-radius:3px; border:1px solid var(--rule);
}
pre{
  font-family:var(--mono); font-size:.82rem; line-height:1.6;
  background:var(--panel); border:1px solid var(--rule); border-radius:3px;
  padding:1.1rem 1.3rem; overflow-x:auto; margin:1.5rem 0;
}
pre code{background:none; border:0; padding:0; font-size:inherit}

/* Wide content breaks out of the prose measure without leaving the article flow. */
.tablewrap,.fig,.math-block{
  width:min(104ch, calc(100vw - 3rem));
  margin-left:50%; transform:translateX(-50%);
}

/* tables */
.tablewrap{overflow-x:auto; margin-top:1.8rem; margin-bottom:2.2rem}
table{border-collapse:collapse; width:100%; font-family:var(--mono); font-size:.79rem;
      font-variant-numeric:tabular-nums; line-height:1.45}
th,td{padding:.5rem .7rem; text-align:right; border-bottom:1px solid var(--rule);
      white-space:nowrap}
th:first-child,td:first-child{text-align:left; white-space:normal; min-width:12ch}
thead th{
  border-bottom:1.5px solid var(--rule-strong); color:var(--muted);
  font-weight:600; font-size:.72rem; letter-spacing:.04em; text-transform:uppercase;
  vertical-align:bottom;
}
tbody tr:hover{background:var(--panel)}
td strong,th strong{font-weight:700; color:var(--ink)}

/* verdict chips */
.chip{
  display:inline-block; font-family:var(--mono); font-size:.68rem; font-weight:700;
  letter-spacing:.08em; text-transform:uppercase; padding:.16em .55em;
  border-radius:2px; border:1px solid currentColor;
}
.chip.fail{color:var(--fail); background:var(--fail-soft)}
.chip.pass{color:var(--pass); background:var(--pass-soft)}

/* figures */
.fig{margin-top:2.4rem; margin-bottom:2.6rem}
.fig img{
  width:100%; height:auto; display:block; border:1px solid var(--rule);
  border-radius:3px; background:#fcfcfb;
}
figcaption{
  margin-top:.7rem; font-family:var(--mono); font-size:.74rem; line-height:1.55;
  color:var(--muted); display:flex; flex-wrap:wrap; gap:.5rem .9rem; align-items:baseline;
}
.cap-num{color:var(--ink-2); font-weight:700; letter-spacing:.04em; text-transform:uppercase}
.cap-body{flex:1 1 24ch; min-width:20ch}
.cap-file{background:none; border:0; padding:0; color:var(--muted); font-size:.95em}

/* math */
.math-block{
  margin-top:1.7rem; margin-bottom:1.7rem; padding:1.1rem 1.3rem; overflow-x:auto;
  background:var(--panel); border:1px solid var(--rule); border-left:3px solid var(--accent);
  border-radius:3px; font-family:var(--serif); font-style:italic;
  font-size:1.02rem; text-align:center;
}
.math{font-style:italic; white-space:nowrap}
.math-block .up,.math .up{font-style:normal; font-family:var(--mono); font-size:.86em}
sub,sup{font-size:.68em; font-style:normal; line-height:0}
.frac{display:inline-flex; flex-direction:column; vertical-align:middle;
      text-align:center; margin:0 .25em}
.fnum{border-bottom:1px solid currentColor; padding:0 .35em}
.fden{padding:0 .35em}
.sqrt{border-top:1px solid currentColor; padding:0 .2em; margin-left:.1em}
.sqrt::before{content:"√"; margin-left:-.55em; border:0}
.hat::after{content:"\\0302"}
.gap{display:inline-block; width:1.6em}

@media(max-width:680px){
  body{font-size:16px; padding-inline:1.1rem}
  .masthead{padding-top:2.8rem}
}
@media(prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""

HEAD = """<title>Multifactor Null Result</title>
<style>%s</style>
""" % CSS

MASTHEAD = """
<header class="masthead">
  <p class="eyebrow">Quantitative equity research &middot; Working paper</p>
  <h1>Does a Price-Based Multi-Factor Composite Add Value in US Large Caps?</h1>
  <p class="standfirst">A systematic long-only factor strategy on the S&amp;P 500, built with
  point-in-time index membership and tested to destruction against four pre-registered
  falsification criteria.</p>
  <div class="meta">
    <span><b>Universe</b> S&amp;P 500, point-in-time</span>
    <span><b>Sample</b> 1999–2026 &middot; 27.4 years</span>
    <span><b>Rebalance</b> Monthly, 8 bps/side</span>
    <span><b>Tests</b> 49 passing</span>
  </div>
</header>
<div class="verdict">
  <div class="verdict-stamp">Hypothesis&nbsp;rejected</div>
  <div class="verdict-body">All four pre-registered falsification criteria fired. The composite
  underperformed SPY by <strong>5.3%/yr out of sample</strong>, produced <strong>no alpha</strong>
  against FF5+momentum (&minus;2.5%/yr, t&nbsp;=&nbsp;&minus;1.56), and selected stocks
  <strong>no better than chance</strong> &mdash; landing at the 31st percentile of 500 random
  portfolios drawn from the same universe.</div>
</div>
"""


def main():
    # utf-8-sig: REPORT.md carries a BOM. Reading it as plain utf-8 leaves U+FEFF
    # at the head of the string, which defeats the \A anchor below (and stops
    # markdown recognising the first line as a heading).
    md = (ROOT / "REPORT.md").read_text(encoding="utf-8-sig")

    # strip the h1 + standfirst; the masthead replaces them
    md = re.sub(r"\A#[^\n]*\n+\*\*[^\n]*\*\*\n+---\n", "", md)

    md = embed_figures(md)
    md = append_unreferenced(md)
    md, math_store = convert_math(md)

    html = markdown.markdown(
        md, extensions=["tables", "fenced_code", "toc", "md_in_html", "attr_list"]
    )

    for key, val in math_store.items():
        html = html.replace(f"<p>{key}</p>", val).replace(key, val)

    html = verdict_chips(html)
    html = re.sub(r"<table>", '<div class="tablewrap"><table>', html)
    html = re.sub(r"</table>", "</table></div>", html)

    toc = build_toc(html)

    # Prose sits in a narrow measure; tables, figures and formulas break out wider
    # via a pure-CSS centred breakout (see .tablewrap/.fig/.math-block rules) so the
    # document tree stays valid.
    body = f'{MASTHEAD}{toc}<article class="wrap">{html}</article>'

    # Emit pure ASCII, escaping every non-ASCII character as a numeric entity.
    # The page is published as a body fragment, so a <meta charset> of our own
    # would arrive too late to influence parsing; entities are charset-proof.
    doc = (HEAD + body).encode("ascii", "xmlcharrefreplace").decode("ascii")
    OUT.write_text(doc, encoding="ascii")
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.name}  ({size:.2f} MB)")
    if size > 16:
        print("WARNING: exceeds the 16 MB artifact limit")


if __name__ == "__main__":
    main()
