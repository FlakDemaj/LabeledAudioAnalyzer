import logging

from datetime import datetime
from typing import List

from src.Entity.LabeledAudioWithSolutionEntity import LabeledAudioWithSolutionEntity
from src.Evaluation.Analyzer import Analyzer
from src.Handler.ExcelHandler import ExcelHandler
from src.Utils.Logger import get_logger
from src.Utils.Constants import (roc_pdf_output_path, recall_precision_pdf_output_path, recall_precision_with_human_labels_pdf_output_path,
    confusion_matrix_base_pdf_output_path, correlations_pdf_output_path, logistic_regression_coefficients_text_output_path,
                                 confusion_matrix_metric_values_text_output_path)


class Starter:

    def __init__(self):
        self.Logger: logging.Logger = get_logger(__name__)
        self.Analyzer: Analyzer = Analyzer()
        self.ExcelHandler: ExcelHandler = ExcelHandler()

        # region Entry Point

    def start(self):
        """
        Entry point of the script.

        Loads labeled audio data from an Excel source and runs a complete analysis
        pipeline on it.
        """
        try:
            starting_time: datetime = datetime.now()

            self.Logger.info(f"Starting at {starting_time}...")

            labeled_audios = self.load_labeled_audios()
            self.run_analysis(labeled_audios)

            end_time = datetime.now()
            self.Logger.info(f"Completed at {end_time}.")
            self.Logger.info(f"Analysis took {end_time - starting_time} seconds.")

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
            labeled_data = self.ExcelHandler.read_and_create_excel()
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

            self.Logger.info(f"Starting the analysis...")

            self.Analyzer.create_confusion_matrix(confusion_matrix_base_pdf_output_path / "human_audio_labeling.pdf",
                                                  confusion_matrix_metric_values_text_output_path,
                                                  labeled_audios,
                                                  "Confusion Matrix Human Audio Labeling")

            self.Analyzer.create_confusion_matrix_with_raw_data("./Data/Input/ConfusionMatrixData/ai_labeling_confusion_matrix.txt",
                                                                confusion_matrix_base_pdf_output_path / "ai_audio_labeling.pdf",
                                                                confusion_matrix_metric_values_text_output_path,
                                                                "Confusion Matrix AI Audio Labeling")

            self.Analyzer.create_confusion_matrix_with_raw_data("./Data/Input/ConfusionMatrixData/ai_human_emulation_confusion_matrix.txt",
                                                                confusion_matrix_base_pdf_output_path / "ai_human_emulation_confusion_matrix.pdf",
                                                                confusion_matrix_metric_values_text_output_path,
                                                                "Confusion Matrix AI Human Emulation")


            self.Analyzer.starting_logistic_regression(
                logistic_regression_coefficients_text_output_path,
                labeled_audios)

            self.Analyzer.create_correlation_matrix(correlations_pdf_output_path,
                                                    labeled_audios,
                                                    "Correlation between the assessment criterion")

            self.Analyzer.create_boxplot(labeled_audios,
                                          "Deepfakes vs. real voices")


            self.Analyzer.create_roc("./Data/Input/Results/ai_result.txt",
                                     roc_pdf_output_path,
                                     "ROC AI Labeling Curve")

            self.Analyzer.create_precision_recall("./Data/Input/Results/ai_result.txt",
                                                  recall_precision_pdf_output_path,
                                                  "Precision-Recall Curve")

            self.Analyzer.create_precision_recall_with_human_labels("./Data/Input/Results/ai_result.txt",
                                                                    recall_precision_with_human_labels_pdf_output_path,
                                                                    labeled_audios,
                                                                    "Precision-Recall-Curve with human-performance")

            #self.Analyzer.start_cluster_analyse(labeled_audios,
            #                        "./Data/Input/Audios/release_in_the_wild",
            #                                    cluster_analyse_pdf_output_path)

            self.Logger.info(f"Analysis complete.")

        except Exception as e:
            self.Logger.critical("Analysis process failed: %s", str(e))
            raise
    #endregion