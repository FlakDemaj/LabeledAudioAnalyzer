import pandas as pd
import logging

from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from src.Entity.LabeledAudioWithSolutionEntity import LabeledAudioWithSolutionEntity
from src.Utils import Logger

class ExcelHandler:

    def __init__(self) -> None:
        # Initialize logger
        self.Logger: logging.Logger = Logger.get_logger(__name__)

        # Cached DataFrame for real labeled audio data
        self.Real_labeled_df: Optional[pd.DataFrame] = None

    def read_and_create_excel(self,
                              labeled_audios_input_path: Path,
                              result_excel_path: Path,
                              real_labeled_audios_path: Path) -> List[LabeledAudioWithSolutionEntity]:
        """
        Reads labeled audio entries from Excel and enriches them with ground truth labels.

        :param labeled_audios_input_path: The Path where the labeled audios are saved.
        :type labeled_audios_input_path: Path
        :param result_excel_path: The Path where the resulting Excel file will be saved.
        :type result_excel_path: Path
        :param real_labeled_audios_path: The path where the audios with the real label are saved
        :type real_labeled_audios_path: Path

        :return: List of LabeledAudioWithSolutionEntity objects containing data and real labels.
        :rtype: List[LabeledAudioWithSolutionEntity]
        :raises RuntimeError: If reading labeled audio data fails.
        """
        try:
            result: List[LabeledAudioWithSolutionEntity] = self.read_excel_file(labeled_audios_input_path, real_labeled_audios_path)
            if result_excel_path.exists():
                return result

            self.write_excel_result_file(result, result_excel_path)
            self.Logger.info("Write labeled audio file done.")
            return result

        except Exception as e:
            self.Logger.exception("Error while reading labeled audio Excel file")
            raise RuntimeError("Failed to read labeled audio data") from e

    def read_real_audio_deepfake_information(self,
                                             audio_title: str,
                                             real_audios_excel_path: Path) -> bool:
        """
        Retrieves the ground truth label for a given audio title from the reference Excel file.

        :param audio_title: The title of the audio to lookup.
        :type audio_title: str
        :param real_audios_excel_path: The path to the excel with the real solution
        :type real_audios_excel_path: Path
        :return: Boolean indicating whether the audio is bona-fide (True) or spoof (False).
        :rtype: bool
        :raises RuntimeError: If reading the real label fails or audio title is not found.
        """
        try:
            if self.Real_labeled_df is None:
                self.Real_labeled_df = pd.read_excel(real_audios_excel_path)

            row = self.Real_labeled_df.loc[self.Real_labeled_df.file == audio_title]

            if row.empty:
                self.Logger.info(f"No match found for audio: {audio_title}")
                raise ValueError(f"No entry found for audio title: {audio_title}")

            return self._convert_to_boolean(row.iloc[0]["label"])

        except Exception as e:
            self.Logger.exception("Error while reading real audio label")
            raise RuntimeError("Failed to read real label for audio") from e

    def write_excel_result_file(self,
                                entities: List[LabeledAudioWithSolutionEntity],
                                result_excel_path: Path) -> None:
        """
        Creates an Excel result file from a list of labeled audio entities.

        :param entities: List of LabeledAudioWithSolutionEntity objects to be saved.
        :type entities: List[LabeledAudioWithSolutionEntity]
        :param result_excel_path: The path where the excel file should be saved
        :type result_excel_path: Path
        :return: None
        :raises RuntimeError: If writing the Excel file fails.
        """
        try:

            self.Logger.info("Write excel file...")
            mapped_audios_df: pd.DataFrame = self._entities_to_dataframe(entities)

            result_excel_path.parent.mkdir(parents=True, exist_ok=True)
            mapped_audios_df.to_excel(result_excel_path, index=False)

            self.Logger.info(f"Excel file saved to {result_excel_path}")
        except Exception as e:
            self.Logger.exception("Error while writing real audio label")
            raise RuntimeError("Failed to write real label for audio") from e

    def read_excel_file(self, labeled_audios_path: Path, real_audios_path: Path) -> List[LabeledAudioWithSolutionEntity]:
        """
                Reads labeled audio with solution excel file

                :param labeled_audios_path: The path to the excel file
                :type labeled_audios_path: Path
                :param real_audios_path: The path to the real_labeled_audios
                :return: List of LabeledAudioWithSolutionEntity objects containing data and real labels.
                :rtype: List[LabeledAudioWithSolutionEntity]
                :raises RuntimeError: If reading labeled audio data fails.
        """

        try:
            self.Logger.info("Read excel file...")
            df: pd.DataFrame = pd.read_excel(labeled_audios_path)
            result: List[LabeledAudioWithSolutionEntity] = []

            for row in tqdm(df.itertuples(index=False), total=len(df)):
                real_label: bool = self.read_real_audio_deepfake_information(row.audio_title, real_audios_path)
                result.append(
                    LabeledAudioWithSolutionEntity(
                        row.audio_title,
                        row.naturalness,
                        row.emotions,
                        row.rhythm,
                        row.is_deepfake,
                        real_label,
                    )
                )
            self.Logger.info("Read excel file done.")
            return result

        except Exception as e:
            self.Logger.exception("Error while reading labeled audio Excel file")
            raise RuntimeError("Failed to read labeled audio data") from e

    @staticmethod
    def _convert_to_boolean(label_value: str) -> bool:
        """
        Converts a label string (e.g., 'bona-fide' or 'spoof') to a boolean value.

        :param label_value: Label string to convert.
        :type label_value: str
        :return: True if label_value is 'bona-fide', else False.
        :rtype: bool
        """
        return True if label_value == "bona-fide" else False

    @staticmethod
    def _entities_to_dataframe(entities: List[LabeledAudioWithSolutionEntity]) -> pd.DataFrame:
        """
        Converts a list of LabeledAudioWithSolutionEntity objects to a pandas DataFrame.

        :param entities: List of LabeledAudioWithSolutionEntity objects.
        :type entities: List[LabeledAudioWithSolutionEntity]
        :return: DataFrame representing the entities.
        :rtype: pd.DataFrame
        """
        data = []
        for e in entities:
            data.append({
                "audio_title": e.audio_title,
                "naturalness_score": e.naturalness_score,
                "emotionality_score": e.emotionality_score,
                "rhythm_score": e.rhythm_score,
                "human_df_label": e.human_df_label,
                "is_truly_df": e.is_truly_df,
            })
        return pd.DataFrame(data)
