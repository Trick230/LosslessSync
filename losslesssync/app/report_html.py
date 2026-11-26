"""Self-contained HTML reports: one page per episode and an index for a batch."""
from __future__ import annotations

import html
import os

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#deded8;--card:#fff;
--seg:#3b7dd8;--seg2:#6aa1ea;--gap:#c9c9c2;--drop:#d8453b;--warn:#c98a12;--ok:#2f9e5a}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--fg:#e8e8e4;--mut:#9a9a94;
--line:#34343a;--card:#202024;--seg:#5b9bf0;--seg2:#3e74c0;--gap:#4a4a50;--drop:#f0685e;
--warn:#e0a63a;--ok:#4cc07a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 8px}
.mut{color:var(--mut)}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--drop)}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.kpi b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
.kpi span{color:var(--mut);font-size:12px}
.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:500;font-size:12px}
td.n,th.n{text-align:right}
ul{margin:0;padding-left:20px}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--mut);font-size:11px}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;
vertical-align:-1px}
a{color:var(--seg)}
"""


def _t(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def _m(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _num(v, fmt=".3f", unit="") -> str:
    return "-" if v is None else f"{v:{fmt}}{unit}"


def _e(s) -> str:
    return html.escape(str(s))


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>{body}"
            f"</main></body></html>")


def _ticks(dur: float, x0: float, w: float, y: float) -> str:
    step = next(s for s in (30, 60, 120, 300, 600, 1200, 3600) if dur / s <= 12)
    out = []
    t = 0.0
    while t <= dur + 1e-6:
        x = x0 + t / dur * w
        out.append(f"<line x1={x:.1f} x2={x:.1f} y1={y - 4} y2={y} stroke='var(--line)'/>"
                   f"<text x={x:.1f} y={y + 12} text-anchor=middle>{_m(t)}</text>")
        t += step
    return "".join(out)


def _timeline(r: dict) -> str:
    dur = r.get("reference_duration") or 1.0
    W, x0, w = 1000, 10, 980
    X = lambda t: x0 + max(0.0, min(t, dur)) / dur * w
    parts = [f"<rect x={x0} y=10 width={w} height=26 fill='var(--gap)' rx='3'/>"]
    for i, s in enumerate(r.get("segments", [])):
        col = "var(--seg)" if i % 2 == 0 else "var(--seg2)"
        tip = (f"#{i} {_t(s['ref_start'])} - {_t(s['ref_end'])}, offset "
               f"{s['offset_s'] * 1000:.3f} ms, {s['method']}")
        parts.append(f"<rect x={X(s['ref_start']):.1f} y=10 width={max(0.5, X(s['ref_end']) - X(s['ref_start'])):.1f}"
                     f" height=26 fill='{col}'><title>{_e(tip)}</title></rect>")
    for s in r.get("segments", [])[1:]:
        parts.append(f"<line x1={X(s['ref_start']):.1f} x2={X(s['ref_start']):.1f} y1=6 y2=40 "
                     f"stroke='var(--fg)' stroke-width=1><title>cut {_t(s['ref_start'])}</title></line>")
    for d in r.get("dropouts") or []:
        x = X(d["start"])
        parts.append(f"<rect x={x - 1.5:.1f} y=4 width={max(3.0, X(d['end']) - x + 3):.1f} height=38 "
                     f"fill='var(--drop)'><title>{_e(d['kind'])} {_t(d['start'])} - {_t(d['end'])}"
                     f"</title></rect>")
    parts.append(_ticks(dur, x0, w, 54))
    return (f"<svg viewBox='0 0 {W} 70' role=img aria-label='timeline'>{''.join(parts)}</svg>"
            "<div class='legend mut'><span style='background:var(--seg)'></span>source used"
            "<span style='background:var(--gap)'></span>gap (no source)"
            "<span style='background:var(--drop)'></span>dropout"
            "<span style='background:var(--fg);width:2px'></span>cut</div>")


def _verify_plot(r: dict) -> str:
    pts = (r.get("verify") or {}).get("points") or []
    if not pts:
        return "<p class=mut>No per-window measurements in this report.</p>"
    dur = r.get("reference_duration") or 1.0
    W, H, x0, w, top, h = 1000, 200, 40, 950, 10, 160
    vals = [abs(e) for _, e in pts if e is not None]
    vals.sort()
    lim = max(2.0, min(10.0, (vals[int(0.95 * (len(vals) - 1))] if vals else 1.0) * 2))
    Y = lambda e: top + h / 2 - max(-lim, min(lim, e)) / lim * h / 2
    X = lambda t: x0 + t / dur * w
    g = [f"<rect x={x0} y={top} width={w} height={h} fill='none' stroke='var(--line)'/>"]
    for v in (-1.0, 1.0):
        g.append(f"<line x1={x0} x2={x0 + w} y1={Y(v):.1f} y2={Y(v):.1f} stroke='var(--warn)' "
                 f"stroke-dasharray='4 3'/><text x={x0 - 4} y={Y(v) + 4:.1f} text-anchor=end>{v:+.0f}</text>")
    g.append(f"<line x1={x0} x2={x0 + w} y1={Y(0):.1f} y2={Y(0):.1f} stroke='var(--line)'/>"
             f"<text x={x0 - 4} y={Y(0) + 4:.1f} text-anchor=end>0</text>"
             f"<text x={x0 - 4} y={top + 10} text-anchor=end>{lim:.0f}</text>")
    for t, e in pts:
        if e is None:
            g.append(f"<line x1={X(t):.1f} x2={X(t):.1f} y1={top + h - 6} y2={top + h} "
                     f"stroke='var(--gap)'><title>{_t(t)}: not measured</title></line>")
            continue
        col = "var(--ok)" if abs(e) <= 1 else "var(--warn)" if abs(e) <= 5 else "var(--drop)"
        g.append(f"<circle cx={X(t):.1f} cy={Y(e):.1f} r=3 fill='{col}'>"
                 f"<title>{_t(t)}: {e:+.3f} ms</title></circle>")
    g.append(_ticks(dur, x0, w, top + h + 14))
    return (f"<svg viewBox='0 0 {W} {H}' role=img aria-label='verify error per window'>"
            f"{''.join(g)}</svg><p class=mut>Timing error of the result against the reference, "
            f"per 6 s window (ms, clipped at ±{lim:.0f}). Grey ticks: windows that could not be "
            f"measured (no shared music/effects).</p>")


def review_items(r: dict, p95_ms: float = 1.0) -> list[str]:
    """Things worth a look, most important first."""
    out = []
    for d in r.get("dropouts") or []:
        what = "audio lost" if d["kind"] == "lost" else "muted"
        out.append(f"<b class=bad>dropout</b> {_t(d['start'])} - {_t(d['end'])} "
                   f"({(d['end'] - d['start']) * 1000:.0f} ms, {what}): listen around it")
    v = r.get("verify") or {}
    if v.get("p95_abs_ms") is not None and v["p95_abs_ms"] > p95_ms:
        out.append(f"<b class=warn>timing</b> p95 {v['p95_abs_ms']:.2f} ms is above {p95_ms:g} ms")
    bad = [(t, e) for t, e in v.get("points") or [] if e is not None and abs(e) > 1.0]
    for t, e in sorted(bad, key=lambda x: -abs(x[1]))[:15]:
        out.append(f"<b class={'bad' if abs(e) > 5 else 'warn'}>window</b> around {_t(t)}: "
                   f"{e:+.3f} ms")
    if len(bad) > 15:
        out.append(f"... and {len(bad) - 15} more window(s) above 1 ms")
    if "verify" in r and not v.get("measured"):
        out.append("<b class=warn>not verified</b>: no window could be measured")
    else:
        from ..sync.pipeline import unmeasured_spans
        for a, b in unmeasured_spans(v.get("points") or []):
            if b - a >= 15:
                out.append(f"<b class=warn>not measured</b> {_t(a)} - {_t(b)}: too little "
                           f"sound shared with the reference to check the timing; listen")
    cov = r.get("coverage_pct")
    if cov is not None and cov < 95:
        out.append(f"<b class=warn>coverage</b> only {cov:.1f}% of the reference has source audio")
    for s in r.get("segments", []):
        if s.get("spread_ms") is not None and s["spread_ms"] > 1.0:
            out.append(f"<b class=warn>segment</b> {_t(s['ref_start'])} - {_t(s['ref_end'])}: "
                       f"offset spread {s['spread_ms']:.2f} ms")
    return out


def episode_page(r: dict, p95_ms: float = 1.0) -> str:
    name = os.path.basename(r.get("output") or r.get("source") or "episode")
    v = r.get("verify") or {}
    kp = r.get("keep_pitch")
    mode = r.get("stretch_mode", "-")
    if kp:
        mode += f" ({kp.get('engine')}" + (f" ±{kp['tol_ms']:g} ms" if "tol_ms" in kp else "") + ")"
    kpis = [("coverage", _num(r.get("coverage_pct"), ".1f", "%")),
            ("median error", _num(v.get("median_abs_ms"), ".3f", " ms")),
            ("p95 error", _num(v.get("p95_abs_ms"), ".3f", " ms")),
            ("within 1 ms", _num(v.get("within_1ms_pct"), ".1f", "%")),
            ("windows measured", f"{v.get('measured', '-')}/{v.get('windows', '-')}"),
            ("segments", str(len(r.get("segments", [])))),
            ("gaps", str(len(r.get("gaps", [])))),
            ("dropouts", str(len(r.get("dropouts") or [])))]
    b = [f"<h1>{_e(name)}</h1>",
         f"<div class=mut>reference {_e(os.path.basename(r.get('reference', '')))} "
         f"(a:{r.get('reference_track')}) · source {_e(os.path.basename(r.get('source', '')))} "
         f"(a:{r.get('source_track')}) · speed ×{r.get('speed', 0):.7f} "
         f"({_e(r.get('speed_fraction'))}) · mode {_e(mode)}</div>",
         "<div class=kpis>" + "".join(f"<div class=kpi><b>{val}</b><span>{k}</span></div>"
                                      for k, val in kpis) + "</div>"]
    items = review_items(r, p95_ms)
    b.append("<h2>To review</h2><div class=box>"
             + ("<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>" if items
                else "<span class=ok>Nothing flagged.</span>") + "</div>")
    b.append(f"<h2>Timeline</h2><div class=box>{_timeline(r)}</div>")
    b.append(f"<h2>Timing accuracy</h2><div class=box>{_verify_plot(r)}</div>")
    rows = "".join(
        f"<tr><td>{i}</td><td>{_t(s['ref_start'])}</td><td>{_t(s['ref_end'])}</td>"
        f"<td>{_t(s['src_start'])}</td><td>{_t(s['src_end'])}</td>"
        f"<td class=n>{s['offset_s'] * 1000:.3f}</td><td>{_e(s['method'])}</td>"
        f"<td class=n>{_num(s.get('spread_ms'))}</td><td class=n>{s.get('anchors', '')}</td></tr>"
        for i, s in enumerate(r.get("segments", [])))
    b.append("<h2>Segments</h2><div class=box><table><tr><th>#</th><th>ref start</th>"
             "<th>ref end</th><th>src start</th><th>src end</th><th class=n>offset ms</th>"
             f"<th>method</th><th class=n>spread ms</th><th class=n>anchors</th></tr>{rows}</table></div>")
    drops = r.get("dropouts") or []
    lost = {(round(d["start"], 3), round(d["end"], 3)) for d in drops if d["kind"] == "lost"}
    ev = [(g["start"], g["end"], "gap", "no source") for g in r.get("gaps", [])
          if (round(g["start"], 3), round(g["end"], 3)) not in lost]
    ev += [(d["start"], d["end"], "dropout", "audio lost" if d["kind"] == "lost" else "muted")
           for d in drops]
    if ev:
        rows = "".join(f"<tr><td>{k}</td><td>{_t(a)}</td><td>{_t(e)}</td>"
                       f"<td class=n>{e - a:.3f}</td><td>{_e(n)}</td></tr>"
                       for a, e, k, n in sorted(ev))
        b.append("<h2>Gaps and dropouts</h2><div class=box><table><tr><th>kind</th><th>start</th>"
                 f"<th>end</th><th class=n>length s</th><th></th></tr>{rows}</table></div>")
    if kp:
        b.append("<h2>Pitch-keeping stretch</h2><div class=box><table>" + "".join(
            f"<tr><td>{_e(k)}</td><td>{_e(f'{val:.3f}' if isinstance(val, float) else val)}</td></tr>"
            for k, val in kp.items()) + "</table></div>")
    return _page(name, "".join(b))


def batch_index(summary: dict, out_dir: str) -> str:
    """Index page for a batch; links each episode's HTML report (relative to out_dir)."""
    eps = summary.get("episodes", [])
    rows = []
    for e in eps:
        page = e.get("html")
        link = (f"<a href='{_e(os.path.relpath(page, out_dir).replace(os.sep, '/'))}'>"
                f"{_e(e['episode'])}</a>" if page else _e(e["episode"]))
        st = e["status"]
        cls = "bad" if st == "error" else "mut" if st.startswith("skip") else "ok"
        chk = "; ".join(e.get("check") or [])
        rows.append(f"<tr><td>{link}</td><td class={cls}>{_e(st)}</td>"
                    f"<td class=n>{_num(e.get('coverage_pct'), '.1f', '%')}</td>"
                    f"<td class=n>{_num(e.get('median_ms'))}</td><td class=n>{_num(e.get('p95_ms'))}</td>"
                    f"<td class=n>{_num(e.get('within_1ms_pct'), '.1f', '%')}</td>"
                    f"<td class=n>{e.get('dropouts', 0) or 0}</td><td class=n>{e.get('gaps', '-')}</td>"
                    f"<td class={'warn' if chk else 'mut'}>{_e(chk) or '-'}</td></tr>")
    n_err = sum(e["status"] == "error" for e in eps)
    n_chk = sum(bool(e.get("check")) for e in eps)
    b = ["<h1>Batch report</h1>",
         f"<div class=mut>{len(eps)} episode(s) · {n_err} error(s) · {n_chk} to check · "
         f"{summary.get('seconds', 0):.0f} s</div>"]
    if summary.get("warnings"):
        b.append("<h2>Warnings</h2><div class=box><ul>"
                 + "".join(f"<li>{_e(w)}</li>" for w in summary["warnings"]) + "</ul></div>")
    b.append("<h2>Episodes</h2><div class=box><table><tr><th>episode</th><th>status</th>"
             "<th class=n>coverage</th><th class=n>median ms</th><th class=n>p95 ms</th>"
             "<th class=n>within 1 ms</th><th class=n>dropouts</th><th class=n>gaps</th>"
             f"<th>check</th></tr>{''.join(rows)}</table></div>")
    return _page("Batch report", "".join(b))
