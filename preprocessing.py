import numpy as np
import pandas as pd

#pour lalecture data chr19
from readfile import charger 


BIN_SIZE = 1000
SHIFT = 100
CHR19_LENGTH = 61_321_190          # assemblage mm8
M = CHR19_LENGTH // BIN_SIZE + 1

#Extremite sequencee du fragment : start sur le brin +, end sur le brin - .
def position_tag(df):
    return np.where(df["strand"] == "+", df["start"], df["end"])


#Un seul read par couple (tag_pos, brin).
#Plusieurs reads au même endroit viennent très probablement d'un même
# #fragment recopié par la PCR. 
#On compare uniquement tag_pos et strand :
#deux reads de longueurs différentes issus du même fragment n'ont pas le
#même start, comparer les lignes entieres les laisserait passer.

def deduplicate(df):
    return df.drop_duplicates(subset=["tag_pos", "strand"]).reset_index(drop=True)


def shift_to_center(df, shift=SHIFT):
    return np.where(df["strand"] == "+",
                    df["tag_pos"] + shift,
                    df["tag_pos"] - shift)

#Fichier chr19 -> tableau avec une colonne 'centre', un centre par fragment.

def fragments(nom):
    tag = charger(nom)
    tag["tag_pos"] = position_tag(tag)
    tag = deduplicate(tag)
    tag["centre"] = shift_to_center(tag)
    return tag

# la on calcule combien de fragment(leur centres) y'a t'il par bin ? 
def bin_counts(centres, m=M, bin_size=BIN_SIZE):
    numero_bin = centres // bin_size
    numero_bin = numero_bin[(numero_bin >= 0) & (numero_bin < m)]   # un centre décalé peut sortir du chromosome
    return np.bincount(numero_bin, minlength=m)

#on cree notre vecteur : Nom de librairie -> vecteur de comptages par bin.
def comptages(nom):
    return bin_counts(fragments(nom)["centre"].to_numpy())

 
if __name__ == "__main__":
    table = pd.DataFrame({nom: comptages(nom) for nom in ("ES", "NP")})
    table.index.name = "bin"
    table.insert(0, "debut", table.index * BIN_SIZE)
    print(table)
