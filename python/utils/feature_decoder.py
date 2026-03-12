#!/usr/bin/env python3
"""Feature name decoder: maps cryptic feature names to biological descriptions."""

# Amino acid composition features
AAC_DESCRIPTIONS = {
    "AAC_A": "Alanine composition",
    "AAC_R": "Arginine composition",
    "AAC_N": "Asparagine composition",
    "AAC_D": "Aspartate composition",
    "AAC_C": "Cysteine composition",
    "AAC_E": "Glutamate composition",
    "AAC_Q": "Glutamine composition",
    "AAC_G": "Glycine composition",
    "AAC_H": "Histidine composition",
    "AAC_I": "Isoleucine composition",
    "AAC_L": "Leucine composition",
    "AAC_K": "Lysine composition",
    "AAC_M": "Methionine composition",
    "AAC_F": "Phenylalanine composition",
    "AAC_P": "Proline composition",
    "AAC_S": "Serine composition",
    "AAC_T": "Threonine composition",
    "AAC_W": "Tryptophan composition",
    "AAC_Y": "Tyrosine composition",
    "AAC_V": "Valine composition",
}

# CTD property groups
CTD_PROPERTIES = {
    "hydrophobicity": "Hydrophobicity (Eisenberg consensus)",
    "normalized.van.der.Waals.volume": "Normalized van der Waals volume",
    "polarity": "Polarity (Grantham)",
    "polarizability": "Polarizability (Charton-Charton)",
    "charge": "Charge (positive/neutral/negative)",
    "secondary.structure": "Secondary structure preference (helix/strand/coil)",
    "solvent.accessibility": "Solvent accessibility (buried/exposed/intermediate)",
}

# Group meanings for CTDC/CTDT/CTDD
CTD_GROUPS = {
    "Group1": "Low property value residues",
    "Group2": "Medium property value residues",
    "Group3": "High property value residues",
}

# Detectability features
DET_DESCRIPTIONS = {
    "det_seq_length": "Protein sequence length",
    "det_log_mw": "Log10 molecular weight",
    "det_pI": "Isoelectric point",
    "det_charge_pH7": "Net charge at pH 7",
    "det_gravy": "Grand average of hydropathy (GRAVY)",
    "det_n_tryptic": "Number of tryptic cleavage sites",
    "det_tryptic_density": "Tryptic sites per 100 residues",
    "det_frac_aromatic": "Fraction aromatic residues (F+W+Y)",
    "det_frac_basic": "Fraction basic residues (K+R+H)",
    "det_frac_hydrophobic": "Fraction hydrophobic residues",
    "det_is_membrane": "Membrane protein (binary)",
}

# Network features
NET_DESCRIPTIONS = {
    "ppi_degree": "PPI network degree centrality",
    "ppi_betweenness": "PPI network betweenness centrality",
    "ppi_clustering_coeff": "PPI network clustering coefficient",
    "ppi_n_detected_neighbors": "Number of detected PPI neighbors",
    "ppi_frac_neighbors_up": "Fraction of PPI neighbors upregulated",
    "ppi_frac_neighbors_down": "Fraction of PPI neighbors downregulated",
    "ppi_frac_neighbors_unch": "Fraction of PPI neighbors unchanged",
    "ppi_weighted_frac_up": "Confidence-weighted fraction neighbors up",
    "ppi_weighted_frac_down": "Confidence-weighted fraction neighbors down",
    "ppi_weighted_frac_unch": "Confidence-weighted fraction neighbors unchanged",
}

# Pathway features
PW_DESCRIPTIONS = {
    "pw_n_pathways": "Number of KEGG pathways",
    "pw_max_frac_up": "Max fraction pathway members upregulated",
    "pw_max_frac_down": "Max fraction pathway members downregulated",
    "pw_mean_frac_up": "Mean fraction pathway members upregulated",
    "pw_mean_frac_down": "Mean fraction pathway members downregulated",
}

# GO-slim mapping: index -> (GO_ID, ontology, term_name)
# Generated from data/ucec/processed/go_slim_mapping.csv
# These are the top 150 most frequent GO annotations for UCEC proteins
GOSLIM_DESCRIPTIONS = {
    "1": ("GO:0005737", "CC", "cytoplasm"),
    "2": ("GO:0005515", "MF", "protein binding"),
    "3": ("GO:0005634", "CC", "nucleus"),
    "4": ("GO:0005829", "CC", "cytosol"),
    "5": ("GO:0005886", "CC", "plasma membrane"),
    "6": ("GO:0016020", "CC", "membrane"),
    "7": ("GO:0005654", "CC", "nucleoplasm"),
    "8": ("GO:0005576", "CC", "extracellular region"),
    "9": ("GO:0005739", "CC", "mitochondrion"),
    "10": ("GO:0005615", "CC", "extracellular space"),
    "11": ("GO:0046872", "MF", "metal ion binding"),
    "12": ("GO:0070062", "CC", "extracellular exosome"),
    "13": ("GO:0005783", "CC", "endoplasmic reticulum"),
    "14": ("GO:0003723", "MF", "RNA binding"),
    "15": ("GO:0042802", "MF", "identical protein binding"),
    "16": ("GO:0005794", "CC", "Golgi apparatus"),
    "17": ("GO:0003677", "MF", "DNA binding"),
    "18": ("GO:0005856", "CC", "cytoskeleton"),
    "19": ("GO:0008270", "MF", "zinc ion binding"),
    "20": ("GO:0005789", "CC", "endoplasmic reticulum membrane"),
    "21": ("GO:0016787", "MF", "hydrolase activity"),
    "22": ("GO:0000166", "MF", "nucleotide binding"),
    "23": ("GO:0045944", "BP", "positive regulation of transcription by RNA polymerase II"),
    "24": ("GO:0007165", "BP", "signal transduction"),
    "25": ("GO:0016740", "MF", "transferase activity"),
    "26": ("GO:0005524", "MF", "ATP binding"),
    "27": ("GO:0031012", "CC", "extracellular matrix"),
    "28": ("GO:0006357", "BP", "regulation of transcription by RNA polymerase II"),
    "29": ("GO:0000122", "BP", "negative regulation of transcription by RNA polymerase II"),
    "30": ("GO:0000981", "MF", "DNA-binding transcription factor activity, RNA polymerase II-specific"),
    "31": ("GO:0005509", "MF", "calcium ion binding"),
    "32": ("GO:0000785", "CC", "chromatin"),
    "33": ("GO:0005925", "CC", "focal adhesion"),
    "34": ("GO:0005730", "CC", "nucleolus"),
    "35": ("GO:0007155", "BP", "cell adhesion"),
    "36": ("GO:0042995", "CC", "cell projection"),
    "37": ("GO:0009986", "CC", "cell surface"),
    "38": ("GO:0048471", "CC", "perinuclear region of cytoplasm"),
    "39": ("GO:0005768", "CC", "endosome"),
    "40": ("GO:0006355", "BP", "regulation of DNA-templated transcription"),
    "41": ("GO:0045893", "BP", "positive regulation of DNA-templated transcription"),
    "42": ("GO:0042803", "MF", "protein homodimerization activity"),
    "43": ("GO:0005813", "CC", "centrosome"),
    "44": ("GO:0006508", "BP", "proteolysis"),
    "45": ("GO:0000978", "MF", "RNA polymerase II cis-regulatory region sequence-specific DNA binding"),
    "46": ("GO:0000139", "CC", "Golgi membrane"),
    "47": ("GO:0005743", "CC", "mitochondrial inner membrane"),
    "48": ("GO:0031410", "CC", "cytoplasmic vesicle"),
    "49": ("GO:0030154", "BP", "cell differentiation"),
    "50": ("GO:0006915", "BP", "apoptotic process"),
    "51": ("GO:0004674", "MF", "protein serine/threonine kinase activity"),
    "52": ("GO:0045202", "CC", "synapse"),
    "53": ("GO:0005759", "CC", "mitochondrial matrix"),
    "54": ("GO:0006629", "BP", "lipid metabolic process"),
    "55": ("GO:0003779", "MF", "actin binding"),
    "56": ("GO:0032991", "CC", "protein-containing complex"),
    "57": ("GO:0006974", "BP", "DNA damage response"),
    "58": ("GO:0015031", "BP", "protein transport"),
    "59": ("GO:0045087", "BP", "innate immune response"),
    "60": ("GO:0003924", "MF", "GTPase activity"),
    "61": ("GO:0005764", "CC", "lysosome"),
    "62": ("GO:0098978", "CC", "glutamatergic synapse"),
    "63": ("GO:0003700", "MF", "DNA-binding transcription factor activity"),
    "64": ("GO:0004672", "MF", "protein kinase activity"),
    "65": ("GO:0016491", "MF", "oxidoreductase activity"),
    "66": ("GO:0045892", "BP", "negative regulation of DNA-templated transcription"),
    "67": ("GO:0005694", "CC", "chromosome"),
    "68": ("GO:0005788", "CC", "endoplasmic reticulum lumen"),
    "69": ("GO:0043066", "BP", "negative regulation of apoptotic process"),
    "70": ("GO:0016324", "CC", "apical plasma membrane"),
    "71": ("GO:0005085", "MF", "guanyl-nucleotide exchange factor activity"),
    "72": ("GO:0008284", "BP", "positive regulation of cell population proliferation"),
    "73": ("GO:0005929", "CC", "cilium"),
    "74": ("GO:0003682", "MF", "chromatin binding"),
    "75": ("GO:0016301", "MF", "kinase activity"),
    "76": ("GO:0035556", "BP", "intracellular signal transduction"),
    "77": ("GO:0002376", "BP", "immune system process"),
    "78": ("GO:0005765", "CC", "lysosomal membrane"),
    "79": ("GO:0005525", "MF", "GTP binding"),
    "80": ("GO:0019901", "MF", "protein kinase binding"),
    "81": ("GO:0070161", "CC", "anchoring junction"),
    "82": ("GO:0015629", "CC", "actin cytoskeleton"),
    "83": ("GO:0008233", "MF", "peptidase activity"),
    "84": ("GO:0007399", "BP", "nervous system development"),
    "85": ("GO:0030424", "CC", "axon"),
    "86": ("GO:0008285", "BP", "negative regulation of cell population proliferation"),
    "87": ("GO:0010628", "BP", "positive regulation of gene expression"),
    "88": ("GO:0030425", "CC", "dendrite"),
    "89": ("GO:0006954", "BP", "inflammatory response"),
    "90": ("GO:0001228", "MF", "DNA-binding transcription activator activity, RNA polymerase II-specific"),
    "91": ("GO:0016567", "BP", "protein ubiquitination"),
    "92": ("GO:0061630", "MF", "ubiquitin protein ligase activity"),
    "93": ("GO:0016887", "MF", "ATP hydrolysis activity"),
    "94": ("GO:0003676", "MF", "nucleic acid binding"),
    "95": ("GO:0006338", "BP", "chromatin remodeling"),
    "96": ("GO:0004252", "MF", "serine-type endopeptidase activity"),
    "97": ("GO:0005096", "MF", "GTPase activator activity"),
    "98": ("GO:0005912", "CC", "adherens junction"),
    "99": ("GO:0051301", "BP", "cell division"),
    "100": ("GO:0001525", "BP", "angiogenesis"),
    "101": ("GO:0005769", "CC", "early endosome"),
    "102": ("GO:0005874", "CC", "microtubule"),
    "103": ("GO:0008017", "MF", "microtubule binding"),
    "104": ("GO:0016477", "BP", "cell migration"),
    "105": ("GO:0006281", "BP", "DNA repair"),
    "106": ("GO:0008289", "MF", "lipid binding"),
    "107": ("GO:0005102", "MF", "signaling receptor binding"),
    "108": ("GO:0016607", "CC", "nuclear speck"),
    "109": ("GO:0043065", "BP", "positive regulation of apoptotic process"),
    "110": ("GO:0019899", "MF", "enzyme binding"),
    "111": ("GO:1990837", "MF", "sequence-specific double-stranded DNA binding"),
    "112": ("GO:0005201", "MF", "extracellular matrix structural constituent"),
    "113": ("GO:0051015", "MF", "actin filament binding"),
    "114": ("GO:0005741", "CC", "mitochondrial outer membrane"),
    "115": ("GO:0030036", "BP", "actin cytoskeleton organization"),
    "116": ("GO:0043025", "CC", "neuronal cell body"),
    "117": ("GO:0016323", "CC", "basolateral plasma membrane"),
    "118": ("GO:0006325", "BP", "chromatin organization"),
    "119": ("GO:0045296", "MF", "cadherin binding"),
    "120": ("GO:0007283", "BP", "spermatogenesis"),
    "121": ("GO:0010008", "CC", "endosome membrane"),
    "122": ("GO:0016192", "BP", "vesicle-mediated transport"),
    "123": ("GO:0036064", "CC", "ciliary basal body"),
    "124": ("GO:0098609", "BP", "cell-cell adhesion"),
    "125": ("GO:0007186", "BP", "G protein-coupled receptor signaling pathway"),
    "126": ("GO:0005178", "MF", "integrin binding"),
    "127": ("GO:0008201", "MF", "heparin binding"),
    "128": ("GO:0031267", "MF", "small GTPase binding"),
    "129": ("GO:0030335", "BP", "positive regulation of cell migration"),
    "130": ("GO:0004867", "MF", "serine-type endopeptidase inhibitor activity"),
    "131": ("GO:0009897", "CC", "external side of plasma membrane"),
    "132": ("GO:0003713", "MF", "transcription coactivator activity"),
    "133": ("GO:0006886", "BP", "intracellular protein transport"),
    "134": ("GO:0010468", "BP", "regulation of gene expression"),
    "135": ("GO:0043565", "MF", "sequence-specific DNA binding"),
    "136": ("GO:0000398", "BP", "mRNA splicing, via spliceosome"),
    "137": ("GO:0030018", "CC", "Z disc"),
    "138": ("GO:0000776", "CC", "kinetochore"),
    "139": ("GO:0006897", "BP", "endocytosis"),
    "140": ("GO:0005604", "CC", "basement membrane"),
    "141": ("GO:0007507", "BP", "heart development"),
    "142": ("GO:0044877", "MF", "protein-containing complex binding"),
    "143": ("GO:0006457", "BP", "protein folding"),
    "144": ("GO:0030027", "CC", "lamellipodium"),
    "145": ("GO:0031625", "MF", "ubiquitin protein ligase binding"),
    "146": ("GO:0007160", "BP", "cell-matrix adhesion"),
    "147": ("GO:0051607", "BP", "defense response to virus"),
    "148": ("GO:0051726", "BP", "regulation of cell cycle"),
    "149": ("GO:0005911", "CC", "cell-cell junction"),
    "150": ("GO:0010629", "BP", "negative regulation of gene expression"),
}


def decode_feature(name):
    """Return human-readable description for a feature name."""
    # Direct lookup
    for d in [AAC_DESCRIPTIONS, DET_DESCRIPTIONS, NET_DESCRIPTIONS, PW_DESCRIPTIONS]:
        if name in d:
            return d[name]

    # Dipeptide composition
    if name.startswith("DC_") and len(name) == 5:
        aa1, aa2 = name[3], name[4]
        return f"{_aa_name(aa1)}-{_aa_name(aa2)} dipeptide frequency"

    # CTD descriptors
    if name.startswith("CTDC_"):
        parts = name[5:].rsplit(".", 1)
        if len(parts) == 2:
            prop, group = parts[0], parts[1]
            prop_desc = CTD_PROPERTIES.get(prop, prop)
            group_desc = CTD_GROUPS.get(group, group)
            return f"Composition: {prop_desc} - {group_desc}"
        return f"CTD composition: {name[5:]}"

    if name.startswith("CTDT_"):
        parts = name[5:].rsplit(".", 1)
        if len(parts) == 2:
            prop = parts[0]
            prop_desc = CTD_PROPERTIES.get(prop, prop)
            return f"Transition: {prop_desc}"
        return f"CTD transition: {name[5:]}"

    if name.startswith("CTDD_"):
        parts = name[5:].rsplit(".", 1)
        if len(parts) == 2:
            prop_desc = CTD_PROPERTIES.get(parts[0], parts[0])
            return f"Distribution: {prop_desc} - {parts[1]}"
        return f"CTD distribution: {name[5:]}"

    # PseAAC
    if name.startswith("PseAAC_"):
        return f"Pseudo amino acid composition (lambda={name[7:]})"
    if name.startswith("APseAAC_"):
        return f"Amphiphilic pseudo amino acid composition (lambda={name[8:]})"

    # CTriad
    if name.startswith("CTriad_"):
        return f"Conjoint triad descriptor {name[7:]}"

    # QSO / SOCN
    if name.startswith("QSO_"):
        return f"Quasi-sequence-order descriptor {name[4:]}"
    if name.startswith("SOCN_"):
        return f"Sequence-order coupling number {name[5:]}"

    # GO features
    if name.startswith("GOslim_"):
        idx = name[7:]
        if idx in GOSLIM_DESCRIPTIONS:
            go_id, ontology, term = GOSLIM_DESCRIPTIONS[idx]
            return f"GO-slim [{ontology}] {term} ({go_id})"
        return f"GO-slim term #{idx} (unmapped)"
    if name.startswith("GO_"):
        parts = name.split("_")
        if len(parts) >= 4 and parts[2] == "sim":
            return f"GO {parts[1]} semantic similarity to {parts[3]} group"
        return f"GO feature: {name[3:]}"

    return name  # fallback

# Amino acid full names
_AA_NAMES = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}

def _aa_name(code):
    return _AA_NAMES.get(code, code)


# Property grouping for enrichment analysis
PROPERTY_GROUPS = {
    "hydrophobicity": ["AAC_A", "AAC_V", "AAC_I", "AAC_L", "AAC_M", "AAC_F", "AAC_W", "AAC_P",
                        "det_gravy", "det_frac_hydrophobic"],
    "charge": ["AAC_K", "AAC_R", "AAC_H", "AAC_D", "AAC_E", "det_charge_pH7", "det_pI",
               "det_frac_basic"],
    "size": ["det_seq_length", "det_log_mw", "det_n_tryptic", "det_tryptic_density"],
    "aromaticity": ["AAC_F", "AAC_W", "AAC_Y", "det_frac_aromatic"],
    "disorder_prone": ["AAC_P", "AAC_G", "AAC_S", "AAC_Q", "AAC_E", "AAC_K"],
    "structure": [],  # CTD secondary structure features will be matched dynamically
}

def get_property_group(feature_name):
    """Return which property group(s) a feature belongs to."""
    groups = []
    for group_name, members in PROPERTY_GROUPS.items():
        if feature_name in members:
            groups.append(group_name)
    # Dynamic matching for CTD features
    if "hydrophobicity" in feature_name.lower():
        groups.append("hydrophobicity")
    if "charge" in feature_name.lower():
        groups.append("charge")
    if "secondary.structure" in feature_name.lower():
        groups.append("structure")
    if "polarity" in feature_name.lower():
        groups.append("charge")  # related
    if "solvent" in feature_name.lower():
        groups.append("hydrophobicity")  # related
    return groups if groups else ["other"]
