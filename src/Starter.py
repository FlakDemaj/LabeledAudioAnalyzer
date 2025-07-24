import logging

from datetime import datetime
from typing import List

from matplotlib.axes import Axes

from src.Entity.LabeledAudioWithSolutionEntity import LabeledAudioWithSolutionEntity
from src.Evaluation.Analyzer import Analyzer
from src.Handler.ExcelHandler import ExcelHandler
from src.Utils.Logger import get_logger
from src.Utils.Constants import *

class Starter:

    def __init__(self, display_plots: bool):
        """
         Args:
            display_plots (bool): Whether or not to display the plots.

        """
        self.Logger: logging.Logger = get_logger(__name__)
        self.Analyzer: Analyzer = Analyzer(display_plots)
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
            self.Logger.critical("Execution aborted due to a fatal error: %s", e)

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
            labeled_data = self.ExcelHandler.read_and_create_excel(
                labeled_audios_excel_input_path,
                result_excel_output_path,
                real_labeled_audios_excel_input_path)
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

            self.Analyzer.create_confusion_matrix_with_raw_data(ai_labeling_confusion_matrix_input_path,
                                                                confusion_matrix_base_pdf_output_path / "ai_audio_labeling.pdf",
                                                                confusion_matrix_metric_values_text_output_path,
                                                                "Confusion Matrix AI Audio Labeling")

            self.Analyzer.create_confusion_matrix_with_raw_data(ai_human_emulation_confusion_matrix_input_path,
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

            roc_ax: Axes = self.Analyzer.create_roc(ai_results_input_path,
                                                    roc_ai_pdf_output_path,
                                                    roc_threshold_text_output_path,
                                                    "AI ROC labeling curve")

            self.Analyzer.create_roc_with_human(roc_ax,
                                                ai_results_input_path,
                                                roc_with_human_perception_pdf_output_path,
                                                roc_threshold_text_output_path,
                                                labeled_audios,
                                            "AI ROC labeling curve with human perception")

            precision_recall_ax: Axes = self.Analyzer.create_precision_recall(ai_results_input_path,
                                                  recall_precision_ai_pdf_output_path,
                                                  recall_precision_human_text_output_data,
                                                  "AI Precision-Recall-Curve")

            self.Analyzer.create_precision_recall_with_human_labels(precision_recall_ax,
                                                         ai_results_input_path,
                                                                    recall_precision_with_human_labels_pdf_output_path,
                                                                    recall_precision_human_text_output_data,
                                                                    labeled_audios,
                                                                    "Precision-Recall-Curve with human-performance")

            self.Logger.info(f"Analysis complete.")

        except Exception as e:
            self.Logger.critical("Analysis process failed: %s", str(e))
            raise
    #endregion