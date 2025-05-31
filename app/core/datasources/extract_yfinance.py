from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

__all__ = ["YahooExtractor"]


class YahooExtractor:
    """Wrapper POO autour de yfinance + cache Parquet."""

    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _chemin_cache(self, actif: str) -> Path:
        return self.cache_dir / f"{actif.upper()}.parquet"

    def _extraire_close(self, df: pd.DataFrame, actif: str) -> pd.Series:
        """Isoler la colonne de clôture quelle que soit la structure."""
        if "Close" in df.columns:
            return df["Close"].rename(actif)

        raise ValueError("Structure de colonnes inattendue (pas de 'Close')")

    # Extraction de données
    def fetch(self, actif: str, start: date, end: date) -> pd.Series:
        """Renvoie la série **Close** ajustée (index DateTime) pour un actif."""

        actif = actif.upper().strip()
        chemin = self._chemin_cache(actif)

        # Verifier le cache
        if chemin.exists():
            serie = pd.read_parquet(chemin)
            if isinstance(serie, pd.DataFrame):
                if serie.shape[1] == 1:
                    serie = serie.iloc[:, 0]
            if isinstance(serie, pd.Series):
                idx_min = pd.to_datetime(serie.index.min()).date()
                idx_max = pd.to_datetime(serie.index.max()).date()
                if idx_min <= start and idx_max >= end:
                    return serie.loc[str(start) : str(end)]

        # Téléchargement
        df = yf.download(
            actif,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,  
            multi_level_index=False,
        )
        if df is None or df.empty:
            raise ValueError(f"Aucune donnée pour {actif} sur la période demandée")

        serie = self._extraire_close(df, actif)

        # Stockage dans le Cache
        serie.to_frame().to_parquet(chemin, compression="snappy")
        return serie.loc[str(start) : str(end)]

    def batch_fetch(self, actifs: List[str], start: date, end: date) -> pd.DataFrame:
        """Concatène les séries Close de plusieurs actifs."""
        frames = [self.fetch(a, start, end) for a in actifs]
        return pd.concat(frames, axis=1).sort_index().ffill()


if __name__ == "__main__":
    extractor = YahooExtractor()
    start_date = date(2020, 1, 1)
    end_date = date(2023, 1, 1)
    data = extractor.fetch("AAPL", start_date, end_date)
    print(data.shape)
