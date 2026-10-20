"""Photocatalyst database — class and wavelength lookup."""

PHOTOCATALYST_DB: dict[str, dict] = {
    "Ir(ppy)3":       {"class": "Ir_cyclometalated", "wavelength_nm": 450},
    "fac-Ir(ppy)3":   {"class": "Ir_cyclometalated", "wavelength_nm": 450},
    "[Ir(ppy)2(dtbbpy)]PF6": {"class": "Ir_cyclometalated", "wavelength_nm": 450},
    "[Ir(dF(CF3)ppy)2(dtbbpy)]PF6": {"class": "Ir_cyclometalated", "wavelength_nm": 450},
    "Ru(bpy)3Cl2":    {"class": "Ru_polypyridyl",   "wavelength_nm": 450},
    "[Ru(bpy)3]Cl2":  {"class": "Ru_polypyridyl",   "wavelength_nm": 450},
    "[Ru(bpy)3]2+":   {"class": "Ru_polypyridyl",   "wavelength_nm": 450},
    "4CzIPN":         {"class": "organic_DA",        "wavelength_nm": 435},
    "Eosin Y":        {"class": "organic_xanthene",  "wavelength_nm": 530},
    "Rose Bengal":     {"class": "organic_xanthene",  "wavelength_nm": 530},
    "TPT":            {"class": "organic_DA",        "wavelength_nm": 420},
    "Acr-Mes":        {"class": "organic_acridinium", "wavelength_nm": 450},
    "Methylene Blue":  {"class": "organic_thiazine",  "wavelength_nm": 660},
    "3DPA2FBN":       {"class": "organic_DA",        "wavelength_nm": 400},
}


def lookup_photocatalyst(name: str | None) -> dict | None:
    """Look up photocatalyst by exact or substring match."""
    if not name:
        return None
    if name in PHOTOCATALYST_DB:
        return PHOTOCATALYST_DB[name]
    name_lower = name.lower()
    for key, val in PHOTOCATALYST_DB.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return val
    return None
