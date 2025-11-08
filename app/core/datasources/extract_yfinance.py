from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf


class YahooExtractor:
    """
    Extraction avec Yfinance et stockage dans un cache local.
    """

    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self.repertoire_cache = Path(cache_dir)
        self.repertoire_cache.mkdir(exist_ok=True)

    def _chemin_cache(self, actif: str) -> Path:
        return self.repertoire_cache / f"{actif.upper()}.parquet"

    def _extraire_close(self, df: pd.DataFrame, actif: str) -> pd.Series:
        """
        Isoler la colonne Close.
        """
        if "Close" in df.columns:
            return df["Close"].rename(actif)

        raise ValueError("Structure de colonnes inattendue (pas de 'Close')")
    
    def extraire_volume(self, df: pd.DataFrame, actif: str) -> pd.Series:
        """
        Isoler la colonne Volume.
        """
        if "Volume" in df.columns:
            return df["Volume"].rename(actif)

        raise ValueError("Structure de colonnes inattendue (pas de 'Volume')")
    

    def fetch(self, actif: str, start: date, end: date) -> pd.Series:
        """
        Renvoie la série Close ajustée pour un actif.
        """
        actif = actif.upper().strip()
        chemin_fichier = self._chemin_cache(actif)

        if chemin_fichier.exists():
            serie_cache = pd.read_parquet(chemin_fichier)
            if isinstance(serie_cache, pd.DataFrame):
                if serie_cache.shape[1] == 1:
                    serie_cache = serie_cache.iloc[:, 0]
            if isinstance(serie_cache, pd.Series):
                date_min_index = pd.to_datetime(serie_cache.index.min()).date()
                date_max_index = pd.to_datetime(serie_cache.index.max()).date()
                if date_min_index <= start and date_max_index >= end:
                    return serie_cache.loc[str(start) : str(end)]

        donnees_telecharge = yf.download(
            actif,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,  
            multi_level_index=False,
        )
        if donnees_telecharge is None or donnees_telecharge.empty:
            raise ValueError(f"Aucune donnée pour {actif} sur la période demandée")

        print(donnees_telecharge.head())

        serie_close = self._extraire_close(donnees_telecharge, actif)

        serie_close.to_frame().to_parquet(chemin_fichier, compression="snappy")
        return serie_close.loc[str(start) : str(end)]
    
    def fetch_volume(self, actif: str, start: date, end: date) -> pd.Series:
        """
        Renvoie la série Volume pour un actif.
        """
        actif = actif.upper().strip()
        

        try:
            self.fetch(actif, start, end)
        except Exception:
            pass
        
        donnees_telecharge = yf.download(
            actif,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,  
            multi_level_index=False,
        )
        if donnees_telecharge is None or donnees_telecharge.empty:
            raise ValueError(f"Aucune donnée pour {actif} sur la période demandée")

        serie_volume = self.extraire_volume(donnees_telecharge, actif)
        return serie_volume.loc[str(start) : str(end)]


    def batch_fetch(self, actifs: List[str], start: date, end: date) -> pd.DataFrame:
        """
        Concatène les séries Close de plusieurs actifs.
        """
        series_actifs = [self.fetch(actif, start, end) for actif in actifs]
        return pd.concat(series_actifs, axis=1).sort_index().ffill()

    def batch_fetch_volume(self, actifs: List[str], start: date, end: date) -> pd.DataFrame:
        """
        Concatène les séries de volumes de plusieurs actifs.
        """
        series_actifs = [self.fetch_volume(actif, start, end) for actif in actifs]
        return pd.concat(series_actifs, axis=1).sort_index().ffill()
    
    def _charger_prix_et_volume(self, actifs: List[str], start: date, end: date) -> tuple[pd.Series, pd.Series]:
        """
        Charge les prix moyens et volumes moyens pour une liste d'actifs.
        """

        donnees_prix = self.extracteur.batch_fetch(actifs, start, end)
        prix_moyen = donnees_prix.mean(axis=1)
        donnees_volume = self.extracteur.batch_fetch_volume(actifs, start, end)
        volume_moyen = donnees_volume.mean(axis=1)
        return prix_moyen, volume_moyen

if __name__ == "__main__":
    extracteur = YahooExtractor()
    date_debut = date(2019, 12, 31)
    date_fin = date(2023, 1, 1)
    donnees_actif = extracteur.fetch("ACIM", date_debut, date_fin)
    