"""Fusionne les deux exports XAUUSD en conservant les valeurs brutes.

Les virgules décimales des exports sont conservées. La reconstruction des
colonnes de prix et de volume se fait après cette étape.
"""

import argparse
import gzip
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


DOSSIER_DONNEES = Path(__file__).resolve().parent.parent / "data"
ANCIEN = DOSSIER_DONNEES / "XAUUSD_Ticks_2004.01.01_2021.10.15.csv"
RECENT = DOSSIER_DONNEES / "XAUUSD_Ticks_2021.01.04_2026.11.20.csv"
SORTIE = DOSSIER_DONNEES / "XAUUSD_Ticks_fusion_brute.csv.gz"
ENTETE = b"Time (EET),Ask,Bid,AskVolume,BidVolume"
TAILLE_DATE = 23
TAILLE_BLOC = 8 * 1024 * 1024
FORMAT_DATE = re.compile(rb"\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}\.\d{3}")


def lire_entete(fichier):
    entete = fichier.readline().removeprefix(b"\xef\xbb\xbf").rstrip(b"\r\n")
    if entete != ENTETE:
        raise ValueError(f"En-tête inattendu dans {fichier.name}.")
    return fichier.tell()


def trouver_debut_recouvrement(ancien, debut_ancien, premiere_date_recente):
    """Cherche le premier tick ancien daté du début du second fichier ou après."""
    gauche = debut_ancien
    droite = ancien.seek(0, 2)

    # On cherche d'abord par position dans le fichier pour éviter de relire 25 Go.
    while droite - gauche > TAILLE_BLOC:
        milieu = (gauche + droite) // 2
        ancien.seek(milieu - 1)
        ancien.readline()  # Repart au début d'une ligne entière.
        ligne = ancien.readline()
        if ligne and ligne[:TAILLE_DATE] < premiere_date_recente:
            gauche = ancien.tell()
        else:
            droite = milieu

    ancien.seek(gauche)
    position = gauche
    while ligne := ancien.readline():
        if ligne[:TAILLE_DATE] >= premiere_date_recente:
            return position
        position = ancien.tell()
    return position


def verifier_recouvrement(ancien, recent, debut_recouvrement, debut_recent):
    """Vérifie que la fin de l'ancien fichier est déjà couverte par le récent."""
    ancien.seek(debut_recouvrement)
    recent.seek(debut_recent)
    doublons = 0
    dernier_bloc = b""

    while bloc_ancien := ancien.read(TAILLE_BLOC):
        bloc_recent = recent.read(len(bloc_ancien))
        if bloc_ancien != bloc_recent:
            raise ValueError(
                "Les sources diffèrent dans le chevauchement, près du tick "
                f"{doublons + 1:,} de l'export récent. Fusion interrompue."
            )
        doublons += bloc_ancien.count(b"\n")
        dernier_bloc = bloc_ancien

    # Le fichier ancien peut s'arrêter au milieu d'une ligne qui continue dans le récent.
    fin_tronquee = bool(dernier_bloc and not dernier_bloc.endswith(b"\n"))
    if fin_tronquee and not recent.readline().endswith(b"\n"):
        raise ValueError("Le dernier tick incomplet manque aussi dans l'export récent.")
    return doublons, fin_tronquee


def copier_ticks(source, destination, fin):
    """Copie les octets jusqu'à la fin de la partie à conserver."""
    nombre_ticks = 0
    dernier_octet = b""
    octets_restants = fin - source.tell()

    while octets_restants:
        bloc = source.read(min(TAILLE_BLOC, octets_restants))
        if not bloc:
            raise ValueError(f"Fin inattendue dans {source.name}.")
        destination.write(bloc)
        octets_restants -= len(bloc)
        ticks_du_bloc = bloc.count(b"\n")
        nombre_ticks += ticks_du_bloc
        dernier_octet = bloc[-1:]
        if nombre_ticks // 50_000_000 != (nombre_ticks - ticks_du_bloc) // 50_000_000:
            print(f"  {nombre_ticks:,} ticks copiés", flush=True)

    if dernier_octet and dernier_octet != b"\n":
        raise ValueError(f"Dernière ligne incomplète dans {source.name}.")
    return nombre_ticks


def fusionner(chemin_ancien, chemin_recent, sortie):
    if sortie.exists():
        raise FileExistsError(f"Le fichier de sortie existe déjà : {sortie}")

    with chemin_ancien.open("rb") as ancien, chemin_recent.open("rb") as recent:
        debut_ancien = lire_entete(ancien)
        debut_recent = lire_entete(recent)
        premiere_date_recente = recent.readline()[:TAILLE_DATE]
        if not FORMAT_DATE.fullmatch(premiere_date_recente):
            raise ValueError(f"Premier tick invalide dans {recent.name}.")

        print("Vérification du chevauchement…", flush=True)
        debut_recouvrement = trouver_debut_recouvrement(ancien, debut_ancien, premiere_date_recente)
        doublons, fin_tronquee = verifier_recouvrement(ancien, recent, debut_recouvrement, debut_recent)
        print(f"{doublons:,} ticks communs vérifiés.", flush=True)
        if fin_tronquee:
            print("Le tick tronqué est récupéré depuis l'export récent.", flush=True)

        fin_recent = recent.seek(0, 2)
        ancien.seek(debut_ancien)
        recent.seek(debut_recent)

        fichier = NamedTemporaryFile(
            dir=sortie.parent, prefix=f"{sortie.name}.", suffix=".part", delete=False
        )
        temporaire = Path(fichier.name)
        try:
            with fichier, gzip.GzipFile(
                filename=sortie.stem, fileobj=fichier, mode="wb", compresslevel=1
            ) as csv:
                csv.write(ENTETE + b"\r\n")
                print("Écriture de la partie ancienne…", flush=True)
                ticks_anciens = copier_ticks(ancien, csv, debut_recouvrement)
                print("Écriture de la partie récente…", flush=True)
                ticks_recents = copier_ticks(recent, csv, fin_recent)
            if sortie.exists():
                raise FileExistsError(f"Le fichier de sortie existe déjà : {sortie}")
            temporaire.rename(sortie)
        finally:
            temporaire.unlink(missing_ok=True)

    print(f"{ticks_anciens + ticks_recents:,} ticks écrits dans {sortie}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ancien", type=Path, default=ANCIEN, help="CSV historique ancien")
    parser.add_argument("--recent", type=Path, default=RECENT, help="CSV historique récent")
    parser.add_argument("--sortie", type=Path, default=SORTIE, help="Chemin du CSV compressé en .gz")
    args = parser.parse_args()
    try:
        fusionner(args.ancien, args.recent, args.sortie)
    except (OSError, ValueError) as erreur:
        parser.exit(1, f"Erreur : {erreur}\n")


if __name__ == "__main__":
    main()
