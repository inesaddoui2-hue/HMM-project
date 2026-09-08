import pandas as pd

DOSSIER = "data/chr19-data"

def charger(nom):
    """Relit un fichier chr19 déjà extrait par awk."""
    return pd.read_csv(
        f"{DOSSIER}/{nom}.chr19.tsv",
        sep="\t",
        dtype={"chrom": "category", "start": "int64",
               "end": "int64", "strand": "category"},
    )