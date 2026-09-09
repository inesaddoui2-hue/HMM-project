import pandas as pd

DOSSIER = "data/chr19-data"

def charger(nom):
    """Relit un fichier chr19 déjà extrait par awk."""
    return pd.read_csv(
        f"{DOSSIER}/{nom}.chr19.tsv",  # # f-string : charger("ES") -> "data/chr19-data/ES.chr19.tsv"
        sep="\t",# une seule valeur repetee ("chr19") -> stockee une fois
        dtype={"chrom": "category", "start": "int64",    # coordonnees imposees en entier, pandas ne peut pas deviner un float
               "end": "int64", "strand": "category"},
    )

## POURQUOI PANDAS ET PAS open()
# Chaque fichier contient plusieurs centaines de milliers de reads. Une boucle
# Python qui lit ligne par ligne et decoupe sur les tabulations mettrait
# plusieurs secondes ; read_csv fait le meme travail. Surtout, il rend des
# colonnes vectorisees, sur lesquelles preprocessing.py travaille d'un coup
# avec numpy au lieu de reboucler.

# POURQUOI IMPOSER LE dtype :
# Sans lui, pandas devine en lisant un echantillon du fichier.