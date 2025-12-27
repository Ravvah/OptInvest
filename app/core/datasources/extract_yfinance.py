from datetime import date, timedelta
from pathlib import Path
from typing import List
import pandas as pd
import yfinance as yf
from functools import lru_cache
from loguru import logger

import app.api.schemas.constants as cst

class YahooExtractor:
    """
    Extraction avec Yfinance et stockage dans un cache local.
    """
 
    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self.repertoire_cache = Path(cache_dir)
        self.repertoire_cache.mkdir(exist_ok=True)

    def extract_and_rename_close_from_dataframe(self, df: pd.DataFrame, actif: str) -> pd.DataFrame:
        """
        Isoler les colonnes Close et Volume.
        """
        if "Close" in df.columns:
            return df[["Close"]].rename(columns={"Close": actif})

    def get_cache_path(self, actif: str) -> Path:
        """
        Retourne le chemin du fichier de données parquet du cache
        """

        return self.repertoire_cache / f"{actif.upper()}.parquet"

    # def rename_close_from_dataframe(self, df: pd.DataFrame, actif: str) -> pd.DataFrame:
    #     """
    #     Renommer la colonne Close.
    #     """
        
    #     if "Close" in df.columns:
    #         return df[["Close"]].rename(columns={"Close": actif})

        # raise ValueError("Structure de colonnes inattendue (pas de 'Close')")
        #raise ValueError("Structure de colonnes inattendue (pas de 'Volume')")

    def _save_df_cache(self, df: pd.DataFrame, actif: str) -> None:
        """
        Sauvegarde une série dans le cache local.
        """
        chemin_fichier = self.get_cache_path(actif)
        df.to_parquet(chemin_fichier, compression="snappy")
        logger.info(f"Données en cache sauvegardées pour {actif} dans {chemin_fichier}")

    def _fetch_with_cache(self, actif: str, start: date, end: date) -> pd.DataFrame | None:
        """
        Récupère les données en utilisant le cache local si possible.
        """

        logger.info(f"Chargement des données en cache pour {actif} ...")
        chemin = self.get_cache_path(actif)
        if not chemin.exists():
            return None
        
        df_cache = pd.read_parquet(chemin)
        if not isinstance(df_cache, pd.DataFrame):
            raise ValueError("Données en cache dans un format inattendu")

        if not df_cache[f"{actif}"].any():
            raise ValueError(f"Aucune donnée Close ou Volume dans le fichier existant pour {actif}")

        cache_start = pd.to_datetime(df_cache.index.min()).date()
        cache_end = pd.to_datetime(df_cache.index.max()).date()

        if cache_start <= start and end <= cache_end:
            logger.info("Données disponibles en cache !")
            return df_cache.loc[str(start) : str(end)]

        return None
    
    def _fetch_with_api(self, actif: str, start:date, end:date) -> pd.DataFrame | None:
        """
        Méthode pour récuperer la série Close pour un actif en appellant l'api YFinance
        """

        logger.info(f"Téléchargement des données pour {actif} de {start} à {end} ...")
        donnees_telechargees = yf.download(
            actif,
            start=start,
            end=end,
            progress=True,
            auto_adjust=True,  
            multi_level_index=False,
        )
        if donnees_telechargees is None or donnees_telechargees.empty:
            raise ValueError(f"Aucune donnée pour {actif} sur la période demandée")
        
        if donnees_telechargees.index.max().date() > end:
            logger.warning(f"Données de YFinance à date dépassantes pour {actif} : fin des données au {donnees_telechargees.index.max().date()} au lieu de {end}") 

        df_cleaned = self.extract_and_rename_close_from_dataframe(donnees_telechargees, actif)

        if actif in cst.ACTIFS_A_CONVERTIR: 
            df_cleaned = self._convert_close_to_eur(actif, df_cleaned, start, end)
        
        return df_cleaned.loc[str(start): str(end)]

    @lru_cache(maxsize=32)
    def fetch_close_actif_main(self, actif: str, start: date, end: date) -> pd.DataFrame:
        """
        Méthode globale pour récuperer la série Close ajustée pour un actif.
        Optimisation avec Least Recently Used Cache
        """

        # Si données dejà en cache
        df_cache = self._fetch_with_cache(actif, start, end)
        if df_cache is not None:
            return df_cache
        
        # Si absence en cache, on telecharge les données sur 10 ans pour remplir le cache
        date_ten_years_ago = date.today() - timedelta(days= 365 * 10)
        date_start_for_cache = min(start, date_ten_years_ago)

        df_augmented_close_for_cache = self._fetch_with_api(actif=actif ,start=date_start_for_cache, end=end)
        
        self._save_df_cache(df_augmented_close_for_cache, actif)

        return df_augmented_close_for_cache

        # df_close_actif = self._fetch_with_api(actif, start, end)

        # donnees_telecharge = yf.download(
        #     actif,
        #     start=start,
        #     end=end,
        #     progress=True,
        #     auto_adjust=True,  
        #     multi_level_index=False,
        # )
        # if donnees_telecharge is None or donnees_telecharge.empty:
        #     raise ValueError(f"Aucune donnée pour {actif} sur la période demandée")
        
        # if donnees_telecharge.index.max().date() > end:
        #     logger.warning(f"Données de YFinance à date dépassantes pour {actif} : fin des données au {donnees_telecharge.index.max().date()} au lieu de {end}")
        # df_cleaned = self.extract_and_rename_close_from_dataframe(donnees_telecharge, actif)
        # # df_cleaned = self.rename_close_from_dataframe(df_cleaned, actif)   
         
        # if actif in cst.ACTIFS_A_CONVERTIR: 
        #     df_cleaned = self._convert_close_to_eur(actif, df_cleaned, start, end)
        
        # self._save_df_cache(df_close_actif, actif)
        # return df_close_actif

    def concat_multiple_closes_from_dataframes(self, actifs: List[str], start: date, end: date) -> pd.DataFrame:
        """
        Concatène les séries Close de plusieurs actifs.
        """

        dfs = [self.fetch_close_actif_main(actif, start, end) for actif in actifs]

        # on remplit les données manquantes par la dernière valeur connue passée
        return pd.concat(dfs, axis=1).sort_index().ffill()

    def _fetch_average_close_from_multiple_actifs(self, actifs: List[str], start: date, end: date) -> pd.Series:
        """
        Charge les prix moyens pour une liste d'actifs.
        """

        df = self.concat_multiple_closes_from_dataframes(actifs, start, end)
        prix_moyen = df[[actif for actif in actifs]].mean(axis=1).rename("Prix_Moyen")
        return prix_moyen
    
    def fetch_exchange_rate_us_to_eur(self, start:date, end:date) -> pd.Series:
        logger.info(f"Téléchargement du taux de change {cst.A} (USD -> EUR) de {start} à {end} ...")

        df_taux = self.fetch_close_actif_main(cst.ACTIF_CONVERSION_DOLLARD_EURO, start, end)
        col_name = cst.ACTIF_CONVERSION_DOLLARD_EURO.replace("=X", "")
        return 1 / (df_taux[col_name].rename("Taux_EUR_USD"))
        
        # chemin_fichier = self.get_cache_path(cst.ACTIF_CONVERSION_DOLLARD_EURO.replace("=X", ""))
        # try:
        #     if chemin_fichier.exists():
        #         df_cache = self._fetch_with_cache(cst.ACTIF_CONVERSION_DOLLARD_EURO.replace("=X", ""), start, end, chemin_fichier)
        #         if df_cache is not None:
        #             taux_eur_usd = df_cache[cst.ACTIF_CONVERSION_DOLLARD_EURO.replace("=X", "")].rename("Taux_EUR_USD")
        #             return 1 / taux_eur_usd 

        # except Exception as e:
        #     logger.warning(f"Erreur lors du chargement du taux en cache: {e}. Téléchargement direct.")
            
        # df_tx_change = yf.download(cst.ACTIF_CONVERSION_DOLLARD_EURO, start=start, end=end, progress=False, auto_adjust=True)

        # if df_tx_change.empty:
        #     raise ValueError(f"Impossible de télécharger le taux de change {cst.ACTIF_CONVERSION_DOLLARD_EURO}.")

        # taux_eur_usd = df_tx_change['Close'].rename("Taux_EUR_USD")

        # taux_usd_eur = 1 / taux_eur_usd
        
        # df_save = pd.DataFrame(taux_eur_usd).rename(columns={"Taux_EUR_USD": cst.ACTIF_CONVERSION_DOLLARD_EURO.replace("=X", "")})
        # self._save_df_cache(df_save, cst.ACTIF_CONVERSION_DOLLARD_EURO.replace("=X", ""))
        
        # return taux_usd_eur
    
    def _convert_close_to_eur(self, actif: str, df_data: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """
        Convertit les prix Close d'un actif USD vers l'EUR.
        """
        
        taux_usd_eur = self.fetch_exchange_rate_us_to_eur(start, end)
        taux_usd_eur = taux_usd_eur.reindex(df_data.index).ffill() 
        
        df_data[actif] = df_data[actif] * taux_usd_eur
        
        logger.info(f"Conversion réussie de {actif} de USD à EUR.")
        return df_data

if __name__ == "__main__":
    extracteur = YahooExtractor()
    date_debut = date(2019, 12, 31)
    date_fin = date(2023, 1, 1)
    # donnees_actif = extracteur._fetch_average_close_and_volume(["ACIM", "AAPL"], date_debut, date_fin)
    # print(donnees_actif)
    donnees_actif = extracteur.fetch_close_actif_main("ACIM", date_debut, date_fin)
    print(donnees_actif)
    