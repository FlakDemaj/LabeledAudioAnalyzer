import logging

from typing import List

from Entity.LabeledAudioWithSolutionEntity import LabeledAudioWithSolutionEntity
from Evaluation.Analyzer import Analyzer
from Handler.ExcelHandler import ExcelHandler
from src.Utils.Logger import get_logger

class Starter:

    def __init__(self):
        self.Logger: logging.Logger = get_logger(__name__)
        self.Analyzer: Analyzer = Analyzer()

        # region Entry Point

    def start(self):
        """
        Entry point of the script.

        Loads labeled audio data from an Excel source and runs a complete analysis
        pipeline on it.
        """
        try:
            labeled_audios = self.load_labeled_audios()
            self.run_analysis(labeled_audios)
        except Exception as e:
            self.Logger.critical("Execution aborted due to a fatal error: ", str(e))

    # endregion

    #region Helper
    def load_labeled_audios(self) -> List[LabeledAudioWithSolutionEntity]:
        """
        Loads labeled audio data from an Excel file and returns it as a list of entities.

        The ExcelHandler reads the input file and constructs instances of
        LabeledAudioWithSolutionEntity.

        Returns:
            List[LabeledAudioWithSolutionEntity]: A list of labeled audio entities.

        Raises:
            Exception: If reading the Excel file fails or the format is invalid.
        """
        try:
            self.Logger.info("Starting the reader...")
            excel_reader = ExcelHandler()
            labeled_data = excel_reader.read_and_create_excel()
            self.Logger.info("Reading is done.")
            return labeled_data
        except Exception as e:
            self.Logger.error("Failed to load labeled audios: %s", str(e))
            raise
    #endregion


    #region Analysis
    def run_analysis(self,labeled_audios: List[LabeledAudioWithSolutionEntity]) -> None:
        """
        Runs various analysis steps on the given labeled audio data.

        This includes:
        - Confusion matrix generation
        - Logistic regression
        - Correlation matrix generation
        - Boxplot visualization

        Args:
            labeled_audios (List[LabeledAudioWithSolutionEntity]):
                A list of labeled audio entities to analyze.

        Raises:
            Exception: If any step in the analysis process fails.
        """
        try:
            self.Logger.info("Starting the analysis...")

            self.Analyzer.create_confusion_matrix(labeled_audios)

            self.Analyzer.starting_logistic_regression(labeled_audios)

            self.Analyzer.create_correlation_matrix(labeled_audios)

            self.Analyzer.create_boxplots(labeled_audios)

            self.Logger.info("Analysis complete.")

        except Exception as e:
            self.Logger.critical("Analysis process failed: %s", str(e))
            raise
    #endregion