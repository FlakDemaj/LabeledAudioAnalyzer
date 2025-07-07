import pandas as pd
import logging

from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from src.Entity.LabeledAudioWithSolutionEntity import LabeledAudioWithSolutionEntity
from src.Utils import Logger
from src.Utils.Constants import base_input_excel_path, result_excel_path

class ExcelHandler:
    def __init__(self) -> None:

        # Define paths to the Excel files
        self.labeled_audios_path: Path = base_input_excel_path / "labeled_audios.xlsx"
        self.real_audios_path: Path = base_input_excel_path / "real_labeled_audios.xlsx"

        # Initialize logger
        self.logger: logging.Logger = Logger.get_logger(__name__)

        # Cached DataFrame for real labeled audio data
        self._real_labeled_df: Optional[pd.DataFrame] = None

    def read_and_create_excel(self) -> List[LabeledAudioWithSolutionEntity]:
        """
        Reads labeled audio entries from Excel and enriches them with ground truth labels.

        :return: List of LabeledAudioWithSolutionEntity objects containing data and real labels.
        :rtype: List[LabeledAudioWithSolutionEntity]
        :raises RuntimeError: If reading labeled audio data fails.
        """
        try:
            result: List[LabeledAudioWithSolutionEntity] = self.read_excel_file()
            if result_excel_path.exists():
                return result

            self.write_excel_result_file(result)
            self.logger.info("Write labeled audio file done.")
            return result

        except Exception as e:
            self.logger.exception("Error while reading labeled audio Excel file")
            raise RuntimeError("Failed to read labeled audio data") from e

    def read_real_audio_deepfake_information(self, audio_title: str) -> bool:
        """
        Retrieves the ground truth label for a given audio title from the reference Excel file.

        :param audio_title: The title of the audio to lookup.
        :type audio_title: str
        :return: Boolean indicating whether the audio is bona-fide (True) or spoof (False).
        :rtype: bool
        :raises RuntimeError: If reading the real label fails or audio title is not found.
        """
        try:
            if self._real_labeled_df is None:
                self._real_labeled_df = pd.read_excel(self.real_audios_path)

            row = self._real_labeled_df.loc[self._real_labeled_df.file == audio_title]

            if row.empty:
                self.logger.info(f"No match found for audio: {audio_title}")
                raise ValueError(f"No entry found for audio title: {audio_title}")

            return self.convert_to_boolean(row.iloc[0]["label"])

        except Exception as e:
            self.logger.exception("Error while reading real audio label")
            raise RuntimeError("Failed to read real label for audio") from e

    def write_excel_result_file(self, entities: List[LabeledAudioWithSolutionEntity]) -> None:
        """
        Creates an Excel result file from a list of labeled audio entities.

        :param entities: List of LabeledAudioWithSolutionEntity objects to be saved.
        :type entities: List[LabeledAudioWithSolutionEntity]
        :return: None
        :raises RuntimeError: If writing the Excel file fails.
        """
        try:
            self.logger.info("Write excel file...")
            mapped_audios_df: pd.DataFrame = self.entities_to_dataframe(entities)

            result_excel_path.parent.mkdir(parents=True, exist_ok=True)
            mapped_audios_df.to_excel(result_excel_path, index=False)

            self.logger.info(f"Excel file saved to {result_excel_path}")
        except Exception as e:
            self.logger.exception("Error while writing real audio label")
            raise RuntimeError("Failed to write real label for audio") from e

    def read_excel_file(self) -> List[LabeledAudioWithSolutionEntity]:
        """
                Reads labeled audio with solution excel file

                :return: List of LabeledAudioWithSolutionEntity objects containing data and real labels.
                :rtype: List[LabeledAudioWithSolutionEntity]
                :raises RuntimeError: If reading labeled audio data fails.
        """

        try:
            self.logger.info("Read excel file...")
            df: pd.DataFrame = pd.read_excel(self.labeled_audios_path)
            result: List[LabeledAudioWithSolutionEntity] = []

            rows_to_iterate = df.iloc[1:]
            for row in tqdm(rows_to_iterate.itertuples(index=False), total=len(rows_to_iterate)):
                real_label: bool = self.read_real_audio_deepfake_information(row.audio_title)
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
            self.logger.info("Read excel file done.")
            return result

        except Exception as e:
            self.logger.exception("Error while reading labeled audio Excel file")
            raise RuntimeError("Failed to read labeled audio data") from e

    @staticmethod
    def convert_to_boolean(label_value: str) -> bool:
        """
        Converts a label string (e.g., 'bona-fide' or 'spoof') to a boolean value.

        :param label_value: Label string to convert.
        :type label_value: str
        :return: True if label_value is 'bona-fide', else False.
        :rtype: bool
        """
        return True if label_value == "bona-fide" else False

    @staticmethod
    def entities_to_dataframe(entities: List[LabeledAudioWithSolutionEntity]) -> pd.DataFrame:
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
