"""Materials-science entity tagging (fast, rule-based, no model download).

Every text-bearing element gets an ``entities`` block so the RAG layer can filter
and boost by material, property, technique, process etc. Rules are deliberately
conservative: precision matters more than recall for metadata filters.
"""
from __future__ import annotations

import re
from collections import Counter

ELEMENTS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re
Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl
Mc Lv Ts Og""".split())

# Words that parse as element strings but are almost always English / acronyms.
FORMULA_STOP = {"I", "In", "As", "At", "Be", "He", "No", "On", "Am", "Es", "Cs", "Os", "Co", "Ca", "Pa", "Ti",
                "Ho", "Ne", "Ar", "B", "C", "N", "O", "F", "P", "S", "K", "V", "Y", "U", "W", "CO", "NO", "IN",
                "AS", "AT", "BE", "HE", "ON", "Bi", "Sc", "Ga", "Ge", "Re", "Pb", "La", "Ce", "Nb", "Sb", "Pd",
                "Al", "Fe", "Ni", "Cu", "Zn", "Ag", "Au", "Mg", "Si", "Sn", "Cr", "Mn", "Mo", "Pt", "Ta", "Zr",
                "Hf", "Li", "Na", "Rb", "Sr", "Ba", "Cd", "Hg", "Tl", "Te", "Se", "Br", "Cl", "Ir", "Ru", "Rh",
                "CoN", "SPS", "PCs", "NiCe", "BCC", "FCC", "HCP", "CNC", "CPU", "NIST", "SEM", "TEM", "XRD",
                "CV", "PC", "PS", "PP", "PE", "CS", "NS", "SN", "YS", "UTS", "IP", "IR", "UV", "PI", "FIB",
                "CAD", "NSF", "DOE", "MIT", "PhD", "BSc", "MSc", "Inc", "Co.", "Ltd", "PV", "PVC", "SiNPs"}

# short all-caps formulas that look like codes but are real compounds
COMMON_SHORT_FORMULAS = {"CO2", "SO2", "NO2", "N2", "O2", "H2", "F2", "I2", "O3", "CH4", "NH3", "NO3", "SO3", "SO4",
                         "PO4", "CS2", "SF6", "BF3", "BN", "UO2", "CF4", "SiO2", "TiO2", "PH3", "HF", "HCl", "H2O",
                         "CN", "NO", "CO", "WC", "TiC", "SiC", "ZrO2", "CeO2", "HfO2", "B4C", "OH"}
_EL = "(?:" + "|".join(sorted(ELEMENTS, key=len, reverse=True)) + ")"
_NUM = r"(?:\d+(?:\.\d+)?|[xyzδ](?:[-+]\d+(?:\.\d+)?)?)?"
_GROUP = rf"(?:{_EL}{_NUM}|\((?:{_EL}{_NUM}[,]?)+\){_NUM})"
FORMULA_RE = re.compile(rf"(?<![\w\-/]){_GROUP}{{2,}}(?:[·•∙.]\d*{_GROUP}+)?(?![\w])")
# Alloy designations: Ti-6Al-4V, Al-4.5Cu, Fe-18Cr-8Ni, CoCrFeMnNi handled by FORMULA_RE
ALLOY_RE = re.compile(rf"\b{_EL}(?:\s?-\s?\d+(?:\.\d+)?\s?{_EL})+\b")
STANDARD_GRADE_RE = re.compile(
    r"\b(?:(?:AISI|SAE|ASTM|UNS|DIN|JIS|EN|ISO|GB/T)\s?[A-Z]?\d{2,5}[A-Z]?(?:-\d+)?"
    r"|\d{4}-T\d{1,4}"          # Al tempers 6061-T6
    r"|(?:SS|SUS)\s?\d{3}[LH]?"  # SS316L
    r"|\d{3}L\b"                 # 316L
    r"|Inconel\s?\d{3}|Hastelloy\s?[A-Z]-?\d*|Ti-?6Al-?4V(?:\s?ELI)?)\b")

_U = {
    "stress/pressure": r"GPa|MPa|kPa|Pa|psi|ksi|bar|mbar|Torr|atm|MPa\s?(?:√m|m\^?1/2|·m1/2)",
    "energy": r"meV|keV|MeV|eV|kJ/mol|kJ\s?mol[−-]1|J/mol|J/m2|J/m²|mJ/m2|mJ/m²|kJ|J",
    "temperature": r"°C|ºC|℃|°F|K(?![a-zA-Z])",
    "length": r"nm|µm|μm|um|mm|cm|Å|pm|m(?![a-zA-Z²³])",
    "composition": r"at\.?\s?%|wt\.?\s?%|vol\.?\s?%|mol\.?\s?%|ppm|ppb|%",
    "specific surface area": r"m2/g|m²/g|m2\s?g[−-]1|m²\s?g[−-]1",
    "density": r"g/cm3|g/cm³|g\s?cm[−-]3|kg/m3|kg/m³|kg\s?m[−-]3",
    "thermal": r"W/\(?m·?K\)?|W\s?m[−-]1\s?K[−-]1|W/mK|J/\(?g·?K\)?|J/kg·?K|K/min|°C/min|K[−-]1|1/K",
    "electrical": r"S/cm|S/m|S\s?cm[−-]1|Ω·?cm|Ω\s?cm|ohm·?cm|Ω|mΩ|kΩ|MΩ|mA/cm2|mA\s?cm[−-]2|mAh/g|mAh\s?g[−-]1|Wh/kg|mV|kV|V|mA|A(?![a-zA-Z])",
    "magnetic": r"emu/g|Oe|kOe|mT|T(?![a-zA-Z])|A/m|µB|μB",
    "hardness": r"HV\d*|HRC|HRB|HB|GPa\s?\(?hardness\)?",
    "frequency": r"GHz|MHz|kHz|Hz|cm[−-]1|cm-1",
    "time": r"ms|µs|μs|ns|ps|fs|min|h(?![a-zA-Z])|s(?![a-zA-Z])|days?|hours?",
    "dose/irradiation": r"dpa|Gy|kGy|ions/cm2",
    "rate": r"s[−-]1|/s|nm/s|mm/s|rpm",
    "mass/amount": r"mg|kg|g(?![a-zA-Z/])|mol|mmol|µmol|μmol",
    "power": r"mW|kW|W(?![a-zA-Z/])",
}
_UNIT_ALT = "|".join(f"(?P<u_{i}>{pat})" for i, pat in enumerate(_U.values()))
_UNIT_KINDS = list(_U.keys())
_NUMBER = r"[-−+]?\d+(?:[.,]\d+)?(?:\s?(?:×|x|\*)\s?10\^?[-−]?\d+|[eE][-−]?\d+)?"
QUANTITY_RE = re.compile(
    rf"(?<![\w.\-/^])(?<![A-Za-z]-)(?P<v1>{_NUMBER})(?:\s?(?:±|\+/-)\s?(?P<err>{_NUMBER}))?"
    rf"(?:\s?(?:–|-|\u2014|to|~)\s?(?P<v2>{_NUMBER}))?\s?(?:{_UNIT_ALT})(?![\w])")

PROPERTIES = [
    "yield strength", "ultimate tensile strength", "tensile strength", "compressive strength", "flexural strength",
    "fatigue strength", "fatigue life", "fracture toughness", "toughness", "hardness", "microhardness",
    "nanohardness", "young's modulus", "elastic modulus", "shear modulus", "bulk modulus", "storage modulus",
    "poisson's ratio", "elastic constants?", "ductility", "elongation", "strain hardening", "creep",
    "stress relaxation", "wear resistance", "friction coefficient", "corrosion resistance", "corrosion rate",
    "oxidation resistance", "density", "porosity", "grain size", "particle size", "crystallite size",
    "lattice parameters?", "lattice constants?", "surface energy", "formation energy", "cohesive energy",
    "activation energy", "stacking fault energy", "diffusion coefficient", "diffusivity", "solubility",
    "melting (?:point|temperature)", "glass transition temperature", "curie temperature", "neel temperature",
    "transition temperature", "thermal conductivity", "thermal diffusivity", "thermal expansion",
    "coefficient of thermal expansion", "specific heat", "heat capacity", "thermal stability",
    "electrical conductivity", "ionic conductivity", "resistivity", "electrical resistivity", "band ?gap",
    "carrier concentration", "carrier mobility", "mobility", "seebeck coefficient", "power factor",
    "figure of merit", "dielectric constant", "permittivity", "dielectric loss", "piezoelectric coefficient",
    "ferroelectric", "polarization", "magnetization", "saturation magnetization", "coercivity",
    "remanence", "permeability", "magnetic susceptibility", "refractive index", "absorption coefficient",
    "reflectance", "transmittance", "photoluminescence", "quantum efficiency", "specific capacity",
    "capacity retention", "coulombic efficiency", "energy density", "power density", "cycle life",
    "viscosity", "crystallinity", "degree of crystallinity", "molecular weight", "glass transition",
    "residual stress", "texture", "anisotropy", "biocompatibility", "wettability", "contact angle",
    "surface roughness", "impact strength", "damping", "superconductivity", "critical temperature",
]
TECHNIQUES = {
    "XRD": r"XRD|X-ray diffraction", "SEM": r"SEM|scanning electron microscop\w*",
    "TEM": r"(?<!S)TEM|transmission electron microscop\w*", "HRTEM": r"HRTEM|HR-TEM",
    "STEM": r"STEM|HAADF(?:-STEM)?", "EDS/EDX": r"EDS|EDX|energy[- ]dispersive (?:X-ray )?spectroscop\w*",
    "EBSD": r"EBSD|electron backscatter(?:ed)? diffraction", "XPS": r"XPS|X-ray photoelectron spectroscop\w*",
    "XRF": r"XRF|X-ray fluorescence", "AFM": r"AFM|atomic force microscop\w*", "FTIR": r"FTIR|FT-IR|infrared spectroscop\w*",
    "Raman": r"Raman(?: spectroscop\w*)?", "UV-Vis": r"UV-?Vis(?:ible)?", "PL": r"photoluminescence spectroscop\w*",
    "DSC": r"DSC|differential scanning calorimetr\w*", "TGA": r"TGA|thermogravimetric",
    "DTA": r"DTA|differential thermal analysis", "DMA": r"DMA|dynamic mechanical analysis",
    "NMR": r"NMR|nuclear magnetic resonance", "SAXS/WAXS": r"SAXS|WAXS|small[- ]angle X-ray scattering",
    "neutron diffraction": r"neutron diffraction|neutron scattering", "APT": r"APT|atom probe tomography",
    "FIB": r"FIB|focused ion beam", "BET": r"BET(?: surface area)?", "ICP": r"ICP-(?:MS|OES|AES)",
    "nanoindentation": r"nanoindentation", "tensile test": r"tensile test\w*|uniaxial tension",
    "hardness test": r"Vickers|Rockwell|Brinell", "impedance spectroscopy": r"impedance spectroscop\w*|EIS",
    "cyclic voltammetry": r"cyclic voltammetry", "SQUID/VSM": r"SQUID|VSM|vibrating sample magnetometr\w*",
    "XAS": r"XAS|XANES|EXAFS", "Mössbauer": r"M[öo]ssbauer", "optical microscopy": r"optical microscop\w*",
    "DFT": r"DFT|density functional theory", "molecular dynamics": r"molecular dynamics|MD simulation\w*",
    "CALPHAD": r"CALPHAD|Thermo-?Calc", "phase-field": r"phase[- ]field", "FEM/FEA": r"FEM|FEA|finite element",
    "Monte Carlo": r"Monte Carlo", "machine learning": r"machine learning|neural network\w*|deep learning",
    "high-throughput": r"high[- ]throughput|combinatorial",
}
PROCESSES = [
    "annealing", "sintering", "spark plasma sintering", "hot isostatic pressing", "hot pressing", "quenching",
    "tempering", "aging", "ageing", "solution treatment", "precipitation hardening", "heat treatment",
    "normalizing", "carburizing", "nitriding", "casting", "investment casting", "die casting", "forging",
    "rolling", "cold rolling", "hot rolling", "extrusion", "drawing", "machining", "welding", "brazing",
    "additive manufacturing", "3d printing", "selective laser melting", "laser powder bed fusion",
    "electron beam melting", "directed energy deposition", "fused deposition modeling", "stereolithography",
    "powder metallurgy", "ball milling", "mechanical alloying", "sol-gel", "hydrothermal synthesis",
    "solvothermal", "co-precipitation", "chemical vapor deposition", "physical vapor deposition",
    "sputtering", "magnetron sputtering", "atomic layer deposition", "molecular beam epitaxy",
    "pulsed laser deposition", "electrodeposition", "electroplating", "spin coating", "dip coating",
    "calcination", "doping", "ion implantation", "electrospinning", "injection molding", "curing",
    "polymerization", "crystallization", "solidification", "recrystallization", "grain growth",
    "phase transformation", "oxidation", "corrosion", "irradiation", "etching", "polishing", "lithography",
]
PHASES = [
    "fcc", "bcc", "hcp", "face-centered cubic", "body-centered cubic", "hexagonal close-packed",
    "simple cubic", "tetragonal", "orthorhombic", "monoclinic", "triclinic", "rhombohedral", "trigonal",
    "perovskite", "spinel", "rutile", "anatase", "brookite", "wurtzite", "zinc blende", "zincblende",
    "rock ?salt", "fluorite", "garnet", "olivine", "corundum", "diamond cubic", "layered", "amorphous",
    "glassy", "nanocrystalline", "single crystal", "polycrystalline", "martensite", "austenite", "ferrite",
    "pearlite", "bainite", "cementite", "ledeburite", "laves phase", "sigma phase", "intermetallic",
    "solid solution", "high[- ]entropy alloy", "eutectic", "eutectoid", "peritectic", "precipitates?",
    "dislocations?", "grain boundar(?:y|ies)", "twin boundar(?:y|ies)", "vacanc(?:y|ies)", "interstitials?",
    "point defects?", "stacking faults?",
]
MATERIAL_CLASSES = [
    "steel", "stainless steel", "cast iron", "aluminum alloy", "aluminium alloy", "titanium alloy",
    "nickel-based superalloy", "superalloy", "magnesium alloy", "copper alloy", "shape memory alloy",
    "high-entropy alloy", "metallic glass", "ceramic", "glass-ceramic", "glass", "polymer", "thermoplastic",
    "thermoset", "elastomer", "hydrogel", "composite", "nanocomposite", "fiber-reinforced", "carbon fiber",
    "graphene", "carbon nanotube", "nanoparticles?", "nanowires?", "quantum dots?", "thin films?",
    "semiconductor", "superconductor", "thermoelectric", "piezoelectric", "ferroelectric", "multiferroic",
    "biomaterial", "metal-organic framework", "MOF", "zeolite", "perovskite solar cell", "2D material",
    "MXene", "cathode", "anode", "electrolyte", "solid electrolyte", "catalyst", "coating",
]
SPACE_GROUP_RE = re.compile(
    r"\b(?:Fm-?3̄?m|Im-?3̄?m|Pm-?3̄?m|Fd-?3̄?m|Ia-?3̄?d|F-?4̄?3m|P6_?3/mmc|P63/mmc|P6/mmm|P4/mmm|I4/mmm|I4/mcm|"
    r"P4mm|R-?3̄?m|R-?3̄?c|R3c|R3m|Pnma|Pbnm|P2_?1/c|P21/c|P21/n|C2/c|C2/m|Cmcm|Pbca|P-?3̄?m1|P-?1|P1)\b")
MILLER_RE = re.compile(r"(?:\((?:\s?-?\d\s?){3,4}\)|\[(?:\s?-?\d\s?){3,4}\]|\{(?:\s?-?\d\s?){3,4}\}|⟨(?:\s?-?\d\s?){3,4}⟩|<(?:\s?-?\d){3,4}>)")
MILLER_CTX = re.compile(r"plane|direction|facet|orientation|peak|reflection|surface|textur|slip|diffraction|"
                        r"zone axis|grain|crystal|lattice|XRD|oriented", re.I)

_PROP_RE = re.compile(r"\b(" + "|".join(PROPERTIES) + r")\b", re.I)
_PROC_RE = re.compile(r"\b(" + "|".join(PROCESSES) + r")\b", re.I)
_PHASE_RE = re.compile(r"\b(" + "|".join(PHASES) + r")\b", re.I)
_CLASS_RE = re.compile(r"\b(" + "|".join(MATERIAL_CLASSES) + r")\b", re.I)
def _split_top_level(pattern: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in pattern:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


def _tech_pattern(v: str) -> re.Pattern:
    # acronyms (XRD, APT, BET ...) stay case-sensitive; spelled-out names are case-insensitive
    alts = [a if re.match(r"[A-Z]{2}", a) else f"(?i:{a})" for a in _split_top_level(v)]
    return re.compile(r"(?<![\w-])(?:" + "|".join(alts) + r")(?![\w-])")


_TECH_RES = {k: _tech_pattern(v) for k, v in TECHNIQUES.items()}


def _num(s: str) -> float | None:
    s = s.replace("−", "-").replace(",", ".").replace(" ", "")
    m = re.match(r"([-+]?\d+(?:\.\d+)?)(?:(?:×|x|\*)10\^?([-]?\d+)|[eE]([-]?\d+))?$", s)
    if not m:
        return None
    v = float(m.group(1))
    exp = m.group(2) or m.group(3)
    return v * 10 ** int(exp) if exp else v


def _is_formula(tok: str) -> bool:
    if tok in FORMULA_STOP or len(tok) < 2:
        return False
    if tok.isupper() and not any(ch.isdigit() for ch in tok):
        return False  # acronyms such as CNC, NIST, SPS
    elements = re.findall(r"[A-Z][a-z]?", tok)
    if not all(e in ELEMENTS for e in elements):
        return False
    if not any(ch.isdigit() for ch in tok) and re.match(r"[A-Z]{2}", tok):
        return False  # acronym-like: SIAs, DFTs, NPs (real digit-free formulas start NaCl, SiC, GaAs ...)
    if re.search(r"\d{3,}", tok) or re.fullmatch(r"[A-Z]{1,3}\d{1,2}", tok) and tok not in COMMON_SHORT_FORMULAS:
        return False  # grant / product codes such as FP7, H2020, A380
    return len(set(elements)) >= 2 or any(ch.isdigit() for ch in tok)


def _uniq(items: list[str]) -> list[str]:
    seen, out = set(), []
    for it in items:
        k = it.lower()
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def extract_entities(text: str) -> dict:
    """Return a dict of non-empty entity lists found in ``text``."""
    if not text or len(text) < 2:
        return {}
    ents: dict[str, list] = {}

    formulas = [m.group(0) for m in FORMULA_RE.finditer(text) if _is_formula(m.group(0))]
    alloys = [m.group(0) for m in ALLOY_RE.finditer(text) if re.search(r"\d", m.group(0))]
    grades = [m.group(0) for m in STANDARD_GRADE_RE.finditer(text)]
    if formulas:
        ents["chemical_formulas"] = _uniq(formulas)
    if alloys or grades:
        ents["alloy_designations"] = _uniq(alloys + grades)

    quants = []
    for m in QUANTITY_RE.finditer(text):
        kind = next((_UNIT_KINDS[int(k[2:])] for k, v in m.groupdict().items() if k.startswith("u_") and v), None)
        unit = next((v for k, v in m.groupdict().items() if k.startswith("u_") and v), None)
        v1 = _num(m.group("v1"))
        if v1 is None or unit is None:
            continue
        # avoid matching years, figure numbers etc. with single-letter units
        if unit in ("s", "h", "m", "A", "V", "K", "T", "g", "W", "J") and not re.search(r"\d\s?" + re.escape(unit) + r"\b", m.group(0)):
            continue
        q = {"text": m.group(0).strip(), "value": v1, "unit": unit.strip(), "quantity_kind": kind,
             "char_start": m.start(), "char_end": m.end()}
        if m.group("v2"):
            q["value_max"] = _num(m.group("v2"))
        if m.group("err"):
            q["uncertainty"] = _num(m.group("err"))
        quants.append(q)
    if quants:
        ents["quantities"] = quants[:100]

    for key, rx in (("properties", _PROP_RE), ("processes", _PROC_RE), ("phases_and_structures", _PHASE_RE),
                    ("material_classes", _CLASS_RE)):
        found = _uniq([m.group(1) for m in rx.finditer(text)])
        if found:
            ents[key] = found

    techs = [k for k, rx in _TECH_RES.items() if rx.search(text)]
    if techs:
        ents["techniques"] = techs

    sgs = _uniq([m.group(0) for m in SPACE_GROUP_RE.finditer(text)])
    if sgs:
        ents["space_groups"] = sgs

    millers = []
    for m in MILLER_RE.finditer(text):
        ctx = text[max(0, m.start() - 60): m.end() + 60]
        if MILLER_CTX.search(ctx):
            millers.append(m.group(0))
    if millers:
        ents["miller_indices"] = _uniq(millers)
    return ents


def merge_entity_summary(all_ents: list[dict], top: int = 30) -> dict:
    """Aggregate per-element entities into a document-level summary with counts."""
    out: dict[str, list] = {}
    for key in ("chemical_formulas", "alloy_designations", "properties", "techniques", "processes",
                "phases_and_structures", "material_classes", "space_groups", "miller_indices"):
        c = Counter()
        for e in all_ents:
            for v in e.get(key, []):
                c[v if key in ("chemical_formulas", "alloy_designations", "space_groups", "miller_indices", "techniques") else v.lower()] += 1
        if c:
            out[key] = [{"value": k, "count": n} for k, n in c.most_common(top)]
    kinds = Counter(q["quantity_kind"] for e in all_ents for q in e.get("quantities", []))
    if kinds:
        out["quantity_kinds"] = dict(kinds.most_common())
    return out
