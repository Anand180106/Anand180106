"""
Animated GitHub profile banner generator.
Master Prompt Specification Implementation.
Emits dark.svg / light.svg + still PNG preview images + metrics.
"""
import io, json, math, os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2

rng = np.random.default_rng(7)

# Search locations for photo
PHOTO_PATHS = [
    r"c:\Users\Anandamirtharaj\OneDrive\Desktop\Anandamirtharaj-dev-main\assets\about_photo.png",
    r"c:\Users\Anandamirtharaj\OneDrive\Desktop\Anandamirtharaj-dev-main\assets\user_picture.jpg",
]

SRC = None
for p in PHOTO_PATHS:
    if os.path.exists(p):
        SRC = p
        break

if not SRC:
    raise FileNotFoundError("User photo not found in desktop assets.")

PADW = 1400          # white side-padding so the crop can pull back to head+shoulders
GW, GH = 300, 340

# ══════════════════════════════════════════════════════════ palette
PAL = {
    "dark": dict(portrait="#A78BFA", chrome="#22D3EE", accent="#10B981",
                 bg="#0A101F", panel="#0D1526", line="#1B2942",
                 text="#C8D6EE", dim="#5A6E93", leader="#243450",
                 title="#8FA6C8", pill_fg="#04121A"),
    "light": dict(portrait="#7C3AED", chrome="#0891B2", accent="#059669",
                  bg="#F4F6FB", panel="#FFFFFF", line="#D9E1EE",
                  text="#1E2A3D", dim="#6B7C96", leader="#C6D2E2",
                  title="#33455F", pill_fg="#FFFFFF"),
}

# ══════════════════════════════════════════════════════════ geometry
W, H      = 1180, 610
BAR       = 38
FR_X, FR_Y, FR_W, FR_H = 20, 58, 448, 504      # portrait frame
PAD       = 12
SCALE     = 1.413                               # grid unit -> px
PT_X, PT_Y = FR_X + PAD, FR_Y + PAD             # portrait origin
IP_X, IP_W = 490, 670                           # info panel
INNER_L   = IP_X + 12
INNER_R   = IP_X + IP_W - 12
CW        = 8.4                                 # mono advance @ 14px
SLOTS     = int((INNER_R - INNER_L) / CW)

ROWS = [
    ("Subject",        "Anandamirtharaj D"),
    ("Role",           "Full Stack Developer · SaaS · AI"),
    ("Origin",         "Coimbatore, Tamil Nadu, India"),
    ("Education",      "B.E Computer Science & Engineering"),
    ("Status",         "Code. Build. Break. Learn. Repeat."),
    ("ToolChain",      "VS Code, Git, GitHub, Docker, Antigravity"),
    None,
    ("Core.Lang",      "Java, TypeScript, JavaScript"),
    ("Core.Frontend",  "React, Next.js, Tailwind, Bootstrap"),
    ("Core.Backend",   "Node.js, Express, FastAPI, LangChain"),
    ("Core.Database",  "PostgreSQL, MySQL, Firestore"),
    ("Core.Infra",     "AWS, Docker, Supabase, Vercel, CI/CD"),
    None,
    ("Grid.Mail",      "anand180106@gmail.com"),
    ("Grid.Portfolio", "anandamirtharaj.me"),
    ("Grid.LinkedIn",  "/in/anandamirtharaj-d"),
    ("Grid.GitHub",    "/Anand180106"),
    ("Grid.LeetCode",  "/u/anandamirtharaj"),
]
HANDLE = "@Anand180106"
LOGOS  = ["vercel", "code", "supabase"]

# ══════════════════════════════════════════════════════════ 1. portrait
def load_padded():
    im = Image.open(SRC).convert("RGB")
    w, h = im.size
    pad = Image.new("RGB", (PADW, h), (255, 255, 255))
    pad.paste(im, ((PADW - w) // 2, 0))
    return pad

def subject_mask(rgb, t=35):
    d = np.sqrt(((255 - rgb.astype(np.float32)) ** 2).sum(2))
    m = ndi.binary_fill_holes(ndi.binary_closing(d > t, np.ones((15, 15))))
    lab, n = ndi.label(m)
    if n:
        s = ndi.sum(m, lab, range(1, n + 1))
        m = lab == int(np.argmax(s)) + 1
    return m

def masked_autocontrast(g, m, cutoff=1):
    lo, hi = np.percentile(g[m], [cutoff, 100 - cutoff])
    return np.clip((g.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1)), 0, 255).astype(np.uint8)

def fs_serpentine(g):
    a = g.astype(np.float32).copy(); h, w = a.shape
    out = np.zeros((h, w), bool)
    for y in range(h):
        rows = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        st = 1 if y % 2 == 0 else -1
        for x in rows:
            old = a[y, x]; new = 255.0 if old >= 128 else 0.0
            out[y, x] = new > 0; err = old - new
            nx, px = x + st, x - st
            if 0 <= nx < w: a[y, nx] += err * .4375
            if y + 1 < h:
                if 0 <= px < w: a[y + 1, px] += err * .1875
                a[y + 1, x] += err * .3125
                if 0 <= nx < w: a[y + 1, nx] += err * .0625
    return out

def portrait_bits():
    im = load_padded()
    w, h = im.size
    mfull = subject_mask(np.asarray(im))
    
    CROP_H = int(h * 0.85)
    CROP_TOP = int(h * 0.05)
    CROP_CX = PADW // 2
    CROP_W = int(round(CROP_H * GW / GH))
    CROP = (CROP_CX - CROP_W // 2, CROP_TOP, CROP_CX - CROP_W // 2 + CROP_W, CROP_TOP + CROP_H)

    im_c = im.crop(CROP).resize((GW, GH), Image.LANCZOS)
    mask = np.asarray(Image.fromarray((mfull * 255).astype(np.uint8))
                      .crop(CROP).resize((GW, GH), Image.LANCZOS)) > 127
    core = ndi.binary_erosion(mask, np.ones((3, 3)))   # kills diffusion bleed at the edge

    g = masked_autocontrast(np.asarray(im_c.convert("L")), mask, 1)
    p = ImageEnhance.Contrast(Image.fromarray(g)).enhance(1.3)
    p = p.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    g = np.asarray(p)

    bits = fs_serpentine(np.where(mask, g, 0).astype(np.uint8)) & core
    return bits, bits, core

# ══════════════════════════════════════════════════════════ 2. logo clouds
def draw_logo_image(name):
    im = Image.new("L", (240, 240), 255)
    dr = ImageDraw.Draw(im)
    if name == "vercel":
        dr.polygon([(120, 35), (215, 205), (25, 205)], fill=0)
    elif name == "code":
        # Slash
        dr.polygon([(135, 40), (155, 40), (105, 200), (85, 200)], fill=0)
        # Left bracket <
        dr.polygon([(75, 120), (115, 70), (95, 70), (55, 120), (95, 170), (115, 170)], fill=0)
        # Right bracket >
        dr.polygon([(165, 120), (125, 70), (145, 70), (185, 120), (145, 170), (125, 170)], fill=0)
    elif name == "supabase":
        # Lightning bolt shape
        dr.polygon([(130, 25), (65, 130), (120, 130), (110, 215), (175, 110), (120, 110)], fill=0)
    return im

def logo_points(name, n=900):
    im = draw_logo_image(name)
    a = np.asarray(im)
    fill = a < 128
    
    lo, hi = 1.5, 40.0
    for _ in range(60):
        s = (lo + hi) / 2
        pts = []
        r = 0
        y = 0.0
        while y < 240:
            off = (s / 2) if r % 2 else 0.0
            x = off
            while x < 240:
                iy, ix = int(y), int(x)
                if 0 <= iy < 240 and 0 <= ix < 240 and fill[iy, ix]:
                    pts.append((x, y))
                x += s
            y += s * 0.866; r += 1
        if len(pts) > n: lo = s
        else: hi = s
        if abs(len(pts) - n) <= 12: break
    pts = np.array(pts, float)
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    pts += rng.normal(0, s * 0.13, pts.shape)
    
    c = pts.mean(0); span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    k = (GW * 0.60) / span
    pts = (pts - c) * k + np.array([GW / 2, GH / 2])
    return pts

def ot_match(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    C = ((a[:, None, :] - b[None, :, :]) ** 2).sum(2)
    r, c = linear_sum_assignment(C)
    return a[r], b[c], math.sqrt(C[r, c].mean())

# ══════════════════════════════════════════════════════════ 3. drift bands
def drift_bands(bits, target, nb=94, frac=0.42, sigma=4.0):
    ys, xs = np.where(bits)
    pos = np.stack([xs, ys], 1).astype(float)
    noisy = pos + rng.normal(0, sigma, pos.shape)
    dv = frac * (target - noisy)
    cent, lab = kmeans2(dv, nb, minit="++", seed=11, iter=40)
    band_d = np.zeros((nb, 2))
    for i in range(nb):
        sel = lab == i
        band_d[i] = dv[sel].mean(0) if sel.any() else 0
    return xs, ys, lab, band_d

def label_field(xs, ys, lab, mask):
    seed = -np.ones((GH, GW), int); seed[ys, xs] = lab
    ind = ndi.distance_transform_edt(seed < 0, return_distances=False, return_indices=True)
    return np.where(mask, seed[tuple(ind)], -1)

def gridness(F):
    out = []
    for A in (F, F.T):
        v = (A[:, :-1] >= 0) & (A[:, 1:] >= 0) & (A[:, :-1] != A[:, 1:])
        h = v.sum(0).astype(float)
        if h.sum() == 0: out.append(0.0); continue
        h /= h.sum(); k = max(1, int(0.05 * len(h)))
        out.append(float(np.sort(h)[-k:].sum() - 0.05))
    return max(out)

def evenness(xs, ys, groups, ng, cells=8):
    cy = (ys * cells // GH).clip(0, cells - 1); cx = (xs * cells // GW).clip(0, cells - 1)
    cid = cy * cells + cx
    glob = np.bincount(cid, minlength=cells * cells).astype(float); glob /= glob.sum()
    vals = []
    for g in range(ng):
        s = groups == g
        if s.sum() < 5: continue
        h = np.bincount(cid[s], minlength=cells * cells).astype(float); h /= h.sum()
        vals.append(0.5 * np.abs(h - glob).sum())
    return float(np.mean(vals))

# ══════════════════════════════════════════════════════════ 4. path runs
def runs_path(xs, ys):
    if len(xs) == 0: return ""
    o = np.lexsort((xs, ys)); xs, ys = xs[o], ys[o]
    hr = []; i = 0; n = len(xs)
    while i < n:
        j = i
        while j + 1 < n and ys[j + 1] == ys[i] and xs[j + 1] == xs[j] + 1: j += 1
        hr.append((int(xs[i]), int(ys[i]), int(xs[j] - xs[i] + 1))); i = j + 1
    by = {}
    for x, y, w in hr: by.setdefault((x, w), []).append(y)
    out = []
    for (x, w), yy in by.items():
        yy.sort(); s = p = yy[0]
        for y in yy[1:]:
            if y == p + 1: p = y
            else: out.append(f"M{x} {s}h{w}v{p-s+1}h-{w}z"); s = p = y
        out.append(f"M{x} {s}h{w}v{p-s+1}h-{w}z")
    return "".join(out)

# ══════════════════════════════════════════════════════════ 5. timing
DUR = 14.2
KT  = [0, 3.0, 4.3, 6.3, 7.6, 9.6, 10.9, 12.9, 14.2]
KTS = ";".join(f"{t/DUR:.4f}" for t in KT)
INTRO = 3.2

# ══════════════════════════════════════════════════════════ 6. SVG
def esc(s): return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def build_svg(theme, bits, bands, travellers, metrics):
    P = PAL[theme]
    xs, ys, lab, band_d = bands
    nb = len(band_d)
    S = []
    A = S.append
    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">')
    A(f'<rect width="{W}" height="{H}" rx="14" fill="{P["bg"]}"/>')
    A(f'<rect x=".5" y=".5" width="{W-1}" height="{H-1}" rx="13.5" fill="none" stroke="{P["line"]}"/>')
    # title bar
    A(f'<path d="M0 14a14 14 0 0 1 14-14h{W-28}a14 14 0 0 1 14 14v{BAR-14}H0z" fill="{P["panel"]}"/>')
    A(f'<path d="M0 {BAR}h{W}" stroke="{P["line"]}"/>')
    for i, c in enumerate(["#FF5F57", "#FEBC2E", "#28C840"]):
        A(f'<circle cx="{22+i*18}" cy="{BAR/2}" r="5.5" fill="{c}"/>')
    A(f'<text x="{W/2}" y="{BAR/2+4.5}" text-anchor="middle" font-size="13" '
      f'fill="{P["title"]}" letter-spacing=".5">profile.sh --live</text>')

    # ── portrait frame
    A(f'<rect x="{FR_X}" y="{FR_Y}" width="{FR_W}" height="{FR_H}" rx="10" '
      f'fill="{P["panel"]}" stroke="{P["line"]}"/>')
    A(f'<text x="{FR_X+14}" y="{FR_Y+24}" font-size="11" fill="{P["chrome"]}" '
      f'letter-spacing="1.6">VISUAL.MAP</text>')
    A(f'<clipPath id="pc_{theme}"><rect x="{PT_X}" y="{PT_Y}" width="{GW*SCALE:.1f}" height="{GH*SCALE:.1f}"/></clipPath>')
    A(f'<g clip-path="url(#pc_{theme})"><g transform="translate({PT_X},{PT_Y}) scale({SCALE})" '
      f'shape-rendering="crispEdges" fill="{P["portrait"]}">')

    # intro layer — 60 interleaved random groups
    NG = 60
    grp = metrics["groups"]
    A(f'<g id="intro">')
    for g in range(NG):
        s = grp == g
        A(f'<path opacity="0" d="{runs_path(xs[s], ys[s])}">'
          f'<animate attributeName="opacity" values="0;1" dur="0.42s" '
          f'begin="{g*2.0/NG:.3f}s" fill="freeze"/></path>')
    A(f'<set attributeName="opacity" to="0" begin="{INTRO}s"/></g>')

    # loop layer — 94 drift bands
    A(f'<g id="loop" opacity="0"><set attributeName="opacity" to="1" begin="{INTRO}s"/>')
    for i in range(nb):
        s = lab == i
        if not s.any(): continue
        dx, dy = band_d[i]
        vals = ";".join(["0 0", f"{dx:.1f} {dy:.1f}", f"{dx:.1f} {dy:.1f}",
                         f"{dx:.1f} {dy:.1f}", f"{dx:.1f} {dy:.1f}",
                         f"{dx:.1f} {dy:.1f}", f"{dx:.1f} {dy:.1f}",
                         f"{dx:.1f} {dy:.1f}", "0 0"])
        A(f'<g><path d="{runs_path(xs[s], ys[s])}"/>'
          f'<animateTransform attributeName="transform" type="translate" '
          f'values="{vals}" keyTimes="{KTS}" dur="{DUR}s" begin="{INTRO}s" repeatCount="indefinite"/>'
          f'<animate attributeName="opacity" values="1;1;0.06;0.06;0.06;0.06;0.06;0.06;1" '
          f'keyTimes="{KTS}" dur="{DUR}s" begin="{INTRO}s" repeatCount="indefinite"/></g>')
    A("</g>")

    # travellers — OT-matched logo morph
    L1, L2, L3 = travellers
    A(f'<g id="trav" fill="{P["accent"]}" opacity="0">'
      f'<animate attributeName="opacity" values="0;0;1;1;1;1;1;1;0" keyTimes="{KTS}" '
      f'dur="{DUR}s" begin="{INTRO}s" repeatCount="indefinite"/>')
    for i in range(len(L1)):
        p3, p1, p2 = L3[i], L1[i], L2[i]
        seq = [p3, p3, p1, p1, p2, p2, p3, p3, p3]
        vals = ";".join(f"{p[0]-p3[0]:.0f} {p[1]-p3[1]:.0f}" for p in seq)
        A(f'<rect x="{p3[0]:.0f}" y="{p3[1]:.0f}" width="2" height="2">'
          f'<animateTransform attributeName="transform" type="translate" values="{vals}" '
          f'keyTimes="{KTS}" dur="{DUR}s" begin="{INTRO}s" repeatCount="indefinite"/></rect>')
    A("</g></g></g>")

    # ── info panel
    A(f'<rect x="{IP_X}" y="{FR_Y}" width="{IP_W}" height="{FR_H}" rx="10" '
      f'fill="{P["panel"]}" stroke="{P["line"]}"/>')
    hy = FR_Y + 28
    A(f'<text x="{INNER_L}" y="{hy}" font-size="13" fill="{P["chrome"]}" '
      f'letter-spacing="1.6">SYSTEM.INFO</text>')
    # LIVE badge
    bx = INNER_R - 62
    A(f'<rect x="{bx}" y="{hy-13}" width="62" height="19" rx="9.5" fill="none" stroke="#FF4D4D" opacity=".55"/>')
    A(f'<circle cx="{bx+13}" cy="{hy-3.5}" r="3.6" fill="#FF4D4D">'
      f'<animate attributeName="opacity" values="1;.15;1" dur="1.6s" repeatCount="indefinite"/></circle>')
    A(f'<text x="{bx+24}" y="{hy+1}" font-size="12" fill="#FF4D4D" letter-spacing="1.2">LIVE</text>')
    A(f'<path d="M{INNER_L} {hy+13}H{INNER_R}" stroke="{P["line"]}"/>')

    y = hy + 40
    for r in ROWS:
        if r is None:
            y += 10; continue
        lb, v = r
        lead = max(SLOTS - len(lb) - len(v) - 2, 1)
        A(f'<text x="{INNER_L}" y="{y}" font-size="14" fill="{P["dim"]}" '
          f'textLength="{len(lb)*CW:.1f}" lengthAdjust="spacingAndGlyphs">{esc(lb)}</text>')
        A(f'<text x="{INNER_L+(len(lb)+1)*CW:.1f}" y="{y}" font-size="14" fill="{P["leader"]}" '
          f'textLength="{lead*CW:.1f}" lengthAdjust="spacingAndGlyphs">{"."*lead}</text>')
        A(f'<text x="{INNER_R}" y="{y}" font-size="14" text-anchor="end" fill="{P["text"]}" '
          f'textLength="{len(v)*CW:.1f}" lengthAdjust="spacingAndGlyphs">{esc(v)}</text>')
        y += 23

    # handle pill
    pw = len(HANDLE) * CW + 30
    A(f'<rect x="{INNER_L}" y="{y+6}" width="{pw:.1f}" height="28" rx="14" fill="{P["chrome"]}"/>')
    A(f'<text x="{INNER_L+pw/2:.1f}" y="{y+25}" font-size="14" text-anchor="middle" '
      f'fill="{P["pill_fg"]}" letter-spacing=".4">{HANDLE}</text>')
    A("</svg>")
    return "".join(S)

# ══════════════════════════════════════════════════════════ main
if __name__ == "__main__":
    print(f"Using source photo: {SRC}")
    dark, light, core = portrait_bits()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(output_dir, "dark_bits.npy"), dark)
    np.save(os.path.join(output_dir, "light_bits.npy"), light)

    clouds = [logo_points(n) for n in LOGOS]
    a, b, c1 = ot_match(clouds[0], clouds[1])
    b2, c, c2 = ot_match(b, clouds[2])
    
    idx = {tuple(np.round(p, 6)): i for i, p in enumerate(b)}
    order = [idx[tuple(np.round(p, 6))] for p in b2]
    L1, L2, L3 = a[order], b2, c
    np.save(os.path.join(output_dir, "travellers.npy"), np.stack([L1, L2, L3]))
    print(f"travellers      : {len(L1)}  mean OT hop {c1:.1f} / {c2:.1f} px")

    out = {}
    target = L1.mean(0)
    for theme, bits in (("dark", dark), ("light", light)):
        xs, ys, lab, band_d = drift_bands(bits, target)
        groups = rng.integers(0, 60, len(xs))

        gn = gridness(label_field(xs, ys, lab, core))
        lab0 = kmeans2(0.42 * (target - np.stack([xs, ys], 1)), 94, minit="++", seed=11, iter=40)[1]
        gn0 = gridness(label_field(xs, ys, lab0, core))
        ev = evenness(xs, ys, groups, 60)
        spat = (np.argsort(np.argsort(ys * GW + xs)) * 60 // len(xs)).clip(0, 59)
        ev_bad = evenness(xs, ys, spat, 60)

        print(f"{theme:5s} dots={len(xs):6d} bands={len(band_d)}  "
              f"gridness={gn:.3f} (no-noise {gn0:.3f})  "
              f"evenness={ev:.3f} (spatial {ev_bad:.3f})")
        svg = build_svg(theme, bits, (xs, ys, lab, band_d), (L1, L2, L3),
                        dict(groups=groups))
        
        target_svg = os.path.join(output_dir, f"{theme}.svg")
        with open(target_svg, "w", encoding="utf-8") as f:
            f.write(svg)
        out[theme] = len(svg)

    for k, v in out.items():
        print(f"{k}.svg  {v/1024:.0f} KB")
