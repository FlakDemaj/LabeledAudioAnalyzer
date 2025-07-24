import json
import logging
import os.path

import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from sklearn import metrics
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, auc, precision_recall_curve, \
    precision_score, recall_score, confusion_matrix
from scipy.stats import ttest_ind
from typing import List, Dict
from pathlib import Path
from ast import literal_eval

from src.Utils.Logger import get_logger
from src.Entity.LabeledAudioWithSolutionEntity import LabeledAudioWithSolutionEntity
from src.Entity.ConfusionMatrixDataEntity import ConfusionMatrixDataEntity
from src.Handler.PDFHandler import PDFHandler
from src.Utils.Constants import base_results_path


class Analyzer:
    def __init__(self, display_plots: bool):
        """
         Args:
            display_plots (bool): Whether or not to display the plots.
        """

        self.PDFHandler = PDFHandler()
        self.Logger: logging.Logger = get_logger(__name__)
        self.Logistic_regression_model: LogisticRegression = LogisticRegression()
        self.Display_plots = display_plots
        # self.FeatureExtractor: FeatureExtractor = FeatureExtractor()

    # region confusionmatrix

    def create_confusion_matrix(self,
                                output_pdf_path: Path,
                                output_metrix_txt_path: Path,
                                labeled_audios: List[LabeledAudioWithSolutionEntity],
                                title: str) -> None:
        """
        Generates and displays a confusion matrix for human-labeled audio data compared to ground truth.

        Optionally saves the confusion matrix as a PDF file to the predefined results directory.

        Args:
            labeled_audios(List[LabeledAudioWithSolutionEntity]): List of audio entities with human labels and ground truth labels.
            output_pdf_path (Path): Path where the resulting PDF plot should be saved.
            output_metrix_txt_path (Path): Path where the resulting metrix text file should be saved.
            title (str): Title of the confusion matrix.

        Returns:
            None
        Raises:
            RuntimeError: If saving the PDF fails due to file system issues.
        """
        self.Logger.info("Create confusion matrix...")
        # Extract true labels and predicted labels from the data
        y_true = [audio.is_truly_df for audio in labeled_audios]
        y_pred = [audio.human_df_label for audio in labeled_audios]

        # Define class labels
        labels = ["Negative", "Positive"]

        # Compute the confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

        # Create the plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, cbar=True)

        # Set axis labels and title
        plt.xlabel("Human Prediction")
        plt.ylabel("True Label")
        plt.title(title)
        plt.tight_layout()
        self.Logger.info("Create confusion matrix done.")

        self.PDFHandler.safe_pdf(output_pdf_path, plt)

        # Creates the metrics for the confusion matrix
        tn, fp = confusion_matrix[0]
        fn, tp = confusion_matrix[1]

        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

        self.create_confusion_matrix_metric(
            output_metrix_txt_path,
            ConfusionMatrixDataEntity(tp, tn, fp, fn),
            title)

        # Show the plot
        self._show_plot_if_enabled()

    def create_confusion_matrix_with_raw_data(self,
                                              input_path: str,
                                              output_pdf_path: Path,
                                              output_metrix_txt_path: Path,
                                              title: str) -> None:
        """
        Generates a confusion matrix based on raw TP, TN, FP, FN values loaded from a file.

        This method constructs and visualizes a confusion matrix using manually
        specified true/false positives and negatives, without requiring the full list
        of prediction labels. The resulting matrix is saved as a PDF and also displayed.

        Args:
            input_path (str): Path to the file containing confusion matrix data (e.g., JSON or TXT).
            output_pdf_path (Path): Path where the resulting PDF plot should be saved.
            output_metrix_txt_path (Path): Path where the resulting metrix text file should be saved.
            title (str): Title to display on the confusion matrix plot.

        Returns:
            None

        Raises:
            RuntimeError: If saving the PDF fails due to file system issues.
        """
        self.Logger.info("Creating confusion matrix from raw data...")

        # Load raw counts from file (assumed to return an entity with TP, TN, FP, FN)
        confusion_matrix_data: ConfusionMatrixDataEntity = self._load_confusion_matrix_data(input_path)

        # Extract values
        tp: int = confusion_matrix_data.true_positive
        tn: int = confusion_matrix_data.true_negative
        fp: int = confusion_matrix_data.false_positive
        fn: int = confusion_matrix_data.false_negative

        # Creates the metrics for the confusion matrix
        self.create_confusion_matrix_metric(
            output_metrix_txt_path,
            ConfusionMatrixDataEntity(tp, tn, fp, fn),
            title)

        # Build confusion matrix as a 2D array
        confusion_matrix = np.array([[tn, fp],
                                     [fn, tp]])

        # Define label names for the axes
        labels = ["Negative", "Positive"]

        # Create the heatmap plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, cbar=True)

        # Label axes and title
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.tight_layout()

        # Save the figure to PDF and show it
        self.PDFHandler.safe_pdf(output_pdf_path, plt)
        self._show_plot_if_enabled()

        self.Logger.info("Confusion matrix created and saved.")

    def create_confusion_matrix_metric(
            self,
            output_path: Path,
            confusion_matrix_entity: ConfusionMatrixDataEntity,
            title: str) -> None:
        """
       Generates metrics for the given confusion metric.

        The metrics include:
        - Sensitivity
	- Negative Predictive Value 
        - Specificity
        - Precision
        - F-Score

        These values are visualized in a simple bar chart and saved as a PDF.

        Args:
            confusion_matrix_entity (ConfusionMatrixDataEntity): Confusion matrix entity.
            output_path (Path): Path where the resulting metrix txt should be saved.
            title (str): The Title for the plot.
        """

        self.Logger.info(f"Creating metrics for {title}...")

        metrics_dict: Dict[str, float] = {
            "Accuracy": confusion_matrix_entity.accuracy,
	    "Negative Predictive Value": confusion_matrix_entity.false_predictive_value,
            "Sensitivity": confusion_matrix_entity.sensitive,
            "Specificity": confusion_matrix_entity.specificity,
            "Precision": confusion_matrix_entity.precision,
            "F1-Score": confusion_matrix_entity.f_score,
        }

        self.Logger.info(
            f"\n Metrics for '{title}':\n" + tabulate(list(metrics_dict.items()), headers=["Metric", "Value"],
                                                       tablefmt="fancy_grid", floatfmt=".4f"))

        self.Logger.info(f"Metrics created for {title}.")

        self._write_result(self._create_file_name(output_path, title),
                           metrics_dict)

    # endregion

    # region logistic regression

    def starting_logistic_regression(self,
                                     output_path: Path,
                                     labeled_audios_list: List[LabeledAudioWithSolutionEntity]
                                     ) -> None:
        """
        Performs logistic regression analysis to predict human-assigned deepfake labels
        using subjective audio scores (naturalness, emotionality, rhythm).

        Trains model on 80% and tests on 20%, logs accuracy and feature coefficients with confidence intervals.
        Outputs odds ratios interpreted as odds that the voice is perceived as *human* (non-deepfake).

        Args:
            labeled_audios_list (List[LabeledAudioWithSolutionEntity]): labeled audio data
            output_path (Path): path for saving results (not changed here)

        Returns:
            None
        """

        if not labeled_audios_list:
            self.Logger.warning("Input list is empty. Aborting analysis.")
            return

        self.Logger.info("Converting input list to pandas DataFrame...")
        labeled_audios_df: pd.DataFrame = self._list_to_dataframe_converter(labeled_audios_list)
        self.Logger.info("DataFrame created successfully.")

        features: List[str] = ["Naturalness", "Emotionality", "Rhythm"]

        # Features and target
        X = labeled_audios_df[features]
        y = labeled_audios_df["human_label"]

        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Add intercept for statsmodels
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)

        # Fit logistic regression with statsmodels
        model = sm.Logit(y_train, X_train_sm)
        result = model.fit(disp=False)

        # Prediction probabilities on test set
        y_pred_prob = result.predict(X_test_sm)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Confidence intervals
        conf_int = result.conf_int(alpha=0.05)
        conf_int.columns = ['2.5%', '97.5%']

        # Coefficients and odds ratios (Deepfake odds)
        coef = result.params
        odds_ratios = np.exp(coef)
        conf_int_exp = np.exp(conf_int)

        # Odds ratios for *human* perception (non-deepfake) = inverse of deepfake odds
        odds_ratios_human = 1 / odds_ratios
        ci_lower_human = 1 / conf_int_exp['97.5%']  # upper bound inverted -> new lower
        ci_upper_human = 1 / conf_int_exp['2.5%']  # lower bound inverted -> new upper

        # Replace odds ratio for intercept with NaN (no meaningful interpretation)
        odds_ratios_human['const'] = np.nan
        ci_lower_human['const'] = np.nan
        ci_upper_human['const'] = np.nan

        # Prepare summary DataFrame with human-odds interpretation
        summary_df = pd.DataFrame({
            'Coefficient': coef,
            'Odds Ratio (human)': odds_ratios_human,
            'CI Lower (2.5%)': ci_lower_human,
            'CI Upper (97.5%)': ci_upper_human
        })

        # Add accuracy row
        accuracy_df = pd.DataFrame({
            'Coefficient': [np.nan],
            'Odds Ratio (human)': [np.nan],
            'CI Lower (2.5%)': [np.nan],
            'CI Upper (97.5%)': [accuracy]
        }, index=['Accuracy'])

        summary_df = pd.concat([summary_df, accuracy_df])

        self.Logger.info(
            "\nLogistic regression results (interpreted as odds that the voice is perceived as human) with 95% confidence intervals:\n" +
            tabulate(summary_df, headers='keys', tablefmt='fancy_grid', floatfmt=".4f")
        )

        self._write_result(output_path, summary_df.to_dict(orient="index"))

    # endregion

    # region correlationmatrix

    def create_correlation_matrix(self,
                                  output_path: Path,
                                  labeled_audios_list: List[LabeledAudioWithSolutionEntity],
                                  title: str) -> None:
        """
        Computes and visualizes the Pearson correlation matrix between selected features
        ('emotionality', 'rhythm', 'naturalness') based on labeled audio data.

        The correlation matrix is shown as a heatmap with annotated values, saved as a PDF,
        and displayed interactively.

        Args:
            labeled_audios_list (List[LabeledAudioWithSolutionEntity]): List of labeled audio data objects.
            output_path (Path): Path where the resulting PDF plot should be saved.
            title (str): Title of the plot.

        Returns:
            None
        """
        self.Logger.info("Creating correlation matrix...")

        # Convert the list of labeled audios to a DataFrame
        labeled_audios_df: pd.DataFrame = self._list_to_dataframe_converter(labeled_audios_list)

        # Define the features for which to compute correlations
        features: List[str] = ["Emotionality", "Rhythm", "Naturalness"]

        # Calculate the Pearson correlation matrix for the selected features
        corr_matrix: pd.DataFrame = labeled_audios_df[features].corr(method="pearson")

        # Create a heatmap visualization of the correlation matrix
        sns.heatmap(corr_matrix, annot=True, cmap="Blues", vmin=0.7, vmax=1)
        plt.title(title)

        self.Logger.info("Created correlation matrix.")

        # Save the heatmap plot as a PDF using the PDF handler
        self.PDFHandler.safe_pdf(output_path, plt)

        # Display the heatmap interactively
        self._show_plot_if_enabled()

    # endregion

    # region boxplot
    def create_boxplot(self,
                       labeled_audios_list: List[LabeledAudioWithSolutionEntity],
                       title: str) -> None:
        """
        Creates boxplot comparing Deepfake and Real audio samples for multiple features
        based on both perceived (human_label) and true (true_label) labels,
        and performs independent t-tests to check for significant differences.

        Each boxplot visualizes the distribution of one feature (naturalness, emotionality, rhythm)
        across the two groups. The t-statistic and p-value from the t-test are displayed on the plot.

        The plots are saved as PDFs named after each feature and label type.

        Args:
            labeled_audios_list (List[LabeledAudioWithSolutionEntity]): List of labeled audio data objects.
            title (str): Title of the plot.

        Returns:
            None
        """
        # Convert the list of labeled audios to a DataFrame for easier processing
        labeled_audios_df: pd.DataFrame = self._list_to_dataframe_converter(labeled_audios_list)

        for feature in ["Naturalness", "Emotionality", "Rhythm"]:
            for label_type in ["human_label", "true_label"]:
                self.Logger.info(f"Create Boxplot for feature: {feature} using {label_type}...")

                # Extract feature values for both groups based on label_type
                group1 = labeled_audios_df[labeled_audios_df[label_type] == False][feature]
                group2 = labeled_audios_df[labeled_audios_df[label_type] == True][feature]

                # Perform independent t-test (unequal variances) between the two groups
                t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
                self.Logger.info(f"{feature.capitalize()} ({label_type}) - T = {t_stat:.3f}, p = {p_value:.4f}")

                # Create the boxplot
                sns.boxplot(x=label_type, y=feature, data=labeled_audios_df)
                label_desc = "Human Perception" if label_type == "human_label" else "Real Result"
                plt.title(f"{feature.capitalize()}: {title}\n({label_desc})")
                plt.xlabel("Deepfake = True")
                plt.ylabel(feature.capitalize())

                # Annotate with t-statistic and p-value
                plt.text(
                    0.5,
                    plt.ylim()[1] * 0.95,
                    f"T = {t_stat:.2f}, p = {p_value:.4f}",
                    horizontalalignment='center',
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
                )

                self.Logger.info(f"Boxplot for feature: {feature} ({label_type}) is done.")

                # Define output path
                output_path: Path = base_results_path / "PDFs" / "boxplots" / f"{feature}_{label_type}_boxplot.pdf"

                # Save and show
                self.PDFHandler.safe_pdf(output_path, plt)

                self._show_plot_if_enabled()

    # endregion

    # region roc

    def create_roc(
            self,
            input_path: str,
            output_pdf_path: Path,
            output_threshold_text_path: Path,
            title: str
    ) -> Axes:
        """
        Loads prediction results from a file, computes the ROC curve, and displays it.

        Args:
            input_path (str): Path to the file containing prediction results.
            output_pdf_path (Path): Path where the PDF should be saved.
            output_threshold_text_path (Path): Path where the text file should be saved.
            title (str): The title for the plot.

        Returns:
            Figure: The matplotlib Figure object of the ROC plot.
        """

        self.Logger.info("Starting creating ROC analysis...")

        # Load prediction results
        result: List[Dict[str, float | int]] = self._read_result_list(input_path)
        y_score: List[float] = [float(entry["result"]) for entry in result]
        y_true: List[int] = [int(entry["label"]) for entry in result]

        # Calculate ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc: float = auc(fpr, tpr)

        self.Logger.info("ROC analysis is done.")

        # Prepare a list with the threshild data and the tpr and fpr
        roc_data = [
            {"Threshold": float(th), "TPR": float(t), "FPR": float(f)}
            for th, t, f in zip(thresholds, tpr, fpr)
        ]

        # Log completion
        self.Logger.info("ROC analysis with human label is done.")
        self.Logger.info(
            "\nThresholds:\n" + tabulate(roc_data, headers="keys", tablefmt="fancy_grid", floatfmt=".4f")
        )

        # Save the ROC data to files
        roc_dict = {entry["Threshold"]: {"TPR": entry["TPR"], "FPR": entry["FPR"]} for entry in roc_data}
        self._write_result(self._create_file_name(output_threshold_text_path, title), roc_dict)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="upper left")

        # Save as PDF
        self.PDFHandler.safe_pdf(output_pdf_path, fig)

        # Show plot
        self._show_plot_if_enabled()

        return ax

    def create_roc_with_human(
            self,
            ax: Axes,  # pass in the existing pyplot object returned by create_roc
            input_path: str,
            output_pdf_path: Path,
            output_threshold_text_path: Path,
            labeled_audios_list: List[LabeledAudioWithSolutionEntity],
            title: str
    ) -> None:
        """
        Adds human performance information to an already plotted ROC curve.

        Args:
            ax (Axes): The matplotlib Axes object of the plot.
            input_path (str): Path to the file containing prediction results.
            output_pdf_path (Path): Path where the PDF should be saved.
            output_threshold_text_path (Path): Path where the threshold data should be saved.
            labeled_audios_list (List[LabeledAudioWithSolutionEntity]): List of labeled audio data.
            title (str): Title for the plot.
        """

        # Log starting message
        self.Logger.info("Starting creating ROC analysis with human labels...")

        # Load model prediction results from file
        result: List[Dict[str, float | int]] = self._read_result_list(input_path)
        y_score = [entry["result"] for entry in result]  # predicted scores
        y_true = [entry["label"] for entry in result]  # ground truth labels

        # Convert the list of labeled audio objects to a DataFrame to access human labels
        labeled_audios_df: pd.DataFrame = self._list_to_dataframe_converter(labeled_audios_list)
        human_labels = labeled_audios_df["human_label"].tolist()

        # Compute ROC thresholds (we already plotted the ROC curve in create_roc)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # Compute human performance via confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, human_labels).ravel()
        human_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate for humans
        human_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate for humans

        self.Logger.info("ROC analysis with human label is done.")

        # Log human metrics
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        self.Logger.info("=" * 60)
        self.Logger.info(f"{GREEN} Human TPR: {human_tpr:.3f}{RESET}")
        self.Logger.info(f"{YELLOW} Human FPR: {human_fpr:.3f}{RESET}")
        self.Logger.info("=" * 60)

        # Save the human performance data to files
        self._write_result(
            self._create_file_name(output_threshold_text_path, "roc_human_data"),
            {"tpr": human_tpr, "fpr": human_fpr},
        )

        ax.set_title(title)
        # Add a scatter point for human performance on the existing plot
        ax.scatter(human_fpr, human_tpr, color='red', s=100, label='Human Performance', zorder=5)
        ax.legend(loc="upper left")  # Update legend to include the human performance

        # Save updated plot as PDF and show it
        ax.figure.savefig(output_pdf_path)
        self._show_plot_if_enabled(ax.figure)

    # endregion

    # region Precision-Recall

    def create_precision_recall(self,
                                input_path: str,
                                output_pdf_path: Path,
                                output_threshold_text_path: Path,
                                title: str) -> Axes:
        """
           Creates and saves a precision-recall curve.

            This method loads model results from a file and then computes precision and recall
            for the model scores. It plots the precision-recall curve for the model.
            The plot is saved as a PDF file.

           Args:
               input_path (str): Path to the input file containing model results.
               output_pdf_path (Path): Path where the generated PDF plot will be saved.
               output_threshold_text_path (Path): Path where the generated text data will be saved.
               title (str): Title of the plot.
            Returns:
               None
           """

        self.Logger.info("Starting creating precision recall analysis...")

        result: List[Dict[str, float | int]] = self._read_result_list(input_path)

        true_labels = [entry["label"] for entry in result]
        ki_scores = [entry["result"] for entry in result]

        precision_ki, recall_ki, thresholds = precision_recall_curve(true_labels, ki_scores)

        self.Logger.info("Precision recall analysis is done.")

        precision_recall_data = [
            {"Threshold": float(th), "Precision": float(t), "Recall": float(f)}
            for th, t, f in zip(thresholds, precision_ki, recall_ki)
        ]

        self.Logger.info(
            "\nThe corresponding threshold values for the precision recall analysis:\n" + (
                tabulate(precision_recall_data, headers="keys",
                         tablefmt="fancy_grid",
                         floatfmt=".4f")))

        roc_dict = {entry["Threshold"]: {"Precision": entry["Precision"], "Recall": entry["Recall"]} for entry in
                    precision_recall_data}

        self._write_result(self._create_file_name(output_threshold_text_path, title), roc_dict)

        fig, ax = plt.subplots()

        ax.plot(recall_ki, precision_ki, label = "AI-Model PR-Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        self.PDFHandler.safe_pdf(output_pdf_path, fig)
        self._show_plot_if_enabled()

        return ax

    def create_precision_recall_with_human_labels(self,
                                                  ax: Axes,
                                                  input_path: str,
                                                  output_pdf_path: Path,
                                                  output_threshold_text_path: Path,
                                                  labeled_audios_list: List[LabeledAudioWithSolutionEntity],
                                                  title: str) -> None:
        """
           Creates and saves a precision-recall curve comparing model predictions with human labels.

           This method loads model results from a file, converts a list of labeled audio entities
           into a DataFrame, and then computes precision and recall for both the model scores
           and human labels. It plots the precision-recall curve for the model and marks
           the human performance as a single point on the plot. The plot is saved as a PDF file.

           Args:
               ax (Axes): The matlib Axes object for the created plot to be extended
               input_path (str): Path to the input file containing model results.
               output_pdf_path (Path): Path where the generated PDF plot will be saved.
               output_threshold_text_path (Path): Path where the text file should be saved.
               labeled_audios_list (List[LabeledAudioWithSolutionEntity]): List of labeled audio entities with human labels.
               title (str): Title of the plot.
           Returns:
               None
           """

        self.Logger.info("Starting creating with human labels precision recall analysis...")

        result: List[Dict[str, float | int]] = self._read_result_list(input_path)
        labeled_audios_df: pd.DataFrame = self._list_to_dataframe_converter(labeled_audios_list)

        true_labels = [entry["label"] for entry in result]
        human_labels = labeled_audios_df["human_label"]

        precision_human = precision_score(true_labels, human_labels)
        recall_human = recall_score(true_labels, human_labels)

        self.Logger.info("Precision recall with human labels analysis is done.")

        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        self.Logger.info("=" * 60)
        self.Logger.info(f"{GREEN} The human precision: {precision_human:.3f}{RESET}")
        self.Logger.info(f"{YELLOW} The human recall is {recall_human:.3f}{RESET}")
        self.Logger.info("=" * 60)

        self._write_result(self._create_file_name(output_threshold_text_path, "precision_recall_human_data"),
                           {
            "human_precision": precision_human,
            "human_recall": recall_human,
        })

        ax.set_title(title)
        ax.scatter(recall_human, precision_human, color='red', s=100, label='Human Performance', zorder=5)
        ax.legend(loc="upper right")  # Update legend to include the human performance

        ax.figure.savefig(output_pdf_path)
        self._show_plot_if_enabled(ax.figure)

    # endregion

    # region Helper
    def _list_to_dataframe_converter(
            self, labeled_audios_list: List[LabeledAudioWithSolutionEntity]
    ) -> pd.DataFrame:
        """
        Converts a list of LabeledAudioWithSolutionEntity objects into a pandas DataFrame
        suitable for logistic regression analysis.

        Each object is transformed into a dictionary containing:
        - naturalness score
        - emotionality score
        - rhythm score
        - human-assigned label
        - ground truth label
        - audio title

        Args:
            labeled_audios_list (List[LabeledAudioWithSolutionEntity]):
                The list of labeled audio entities to convert.

        Returns:
            pd.DataFrame: A DataFrame with the relevant features and labels.
        """

        self.Logger.info("Converting input list to pandas DataFrame...")

        df: pd.DataFrame = pd.DataFrame([{
            "Naturalness": x.naturalness_score,
            "Emotionality": x.emotionality_score,
            "Rhythm": x.rhythm_score,
            "human_label": x.human_df_label,
            "true_label": x.is_truly_df,
            "audio_title": x.audio_title
        } for x in labeled_audios_list])

        self.Logger.info("DataFrame created successfully.")

        return df

    def _write_result(self, output_path: Path, dictionary) -> None:
        """
         Writes a dictionary into a text file

         The data is expected to be a dictionary

         Args:
             output_path (Path): Path to the output file
             dictionary (dict): Dictionary to be written

         Returns:
             None
         """

        if output_path.exists():
            return None
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.Logger.info(f"Writing text file to {output_path}...")

        with open(output_path, "w") as file:
            file.write(json.dumps(dictionary, indent=4))

        self.Logger.info(f"Text file saved to {output_path}")


    def _show_plot_if_enabled(self, fig: Axes = None):
        """
        Shows the given figure (or the current plot)
        if display_plots is enabled, otherwise closes it.
        """
        if self.Display_plots:
            if fig is not None:
                fig.show()
            else:
                plt.show()
        else:
            if fig is not None:
                plt.close(fig)
            else:
                plt.close()

    @staticmethod
    def _read_result_list(input_path: str) -> List[Dict[str, float | int]]:
        """
        Loads a list of dictionaries from a file.

        The file is expected to contain a string representation of a list
        of dictionaries, each with keys 'result' and 'label'.

        Args:
            input_path (Path): Path to the input file.

        Returns:
            List[Dict[str, float | int]]: A list where each element is a dictionary
            containing a float 'result' and an integer 'label'.
        """
        with open(input_path, "r") as file:
            data = literal_eval(file.read())

        return data

    @staticmethod
    def _load_confusion_matrix_data(input_path: str) -> ConfusionMatrixDataEntity:
        """
        Loads confusion matrix data from a JSON file and returns a populated entity.

        The expected JSON structure should include the following keys:
        - 'tp' (true positives)
        - 'tn' (true negatives)
        - 'fp' (false positives)
        - 'fn' (false negatives)

        Args:
            input_path (str): Path to the JSON file containing confusion matrix values.

        Returns:
            ConfusionMatrixDataEntity: Object containing all confusion matrix values.
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        return ConfusionMatrixDataEntity(
            data['tp'],
            data['tn'],
            data['fp'],
            data['fn']
        )

    @staticmethod
    def _create_file_name(output_path: Path, title: str) -> Path:
        """
        Creates a file name from the title and the output_file.

        Args:
            title (str): The title of the file which is shown in the logs
            output_path (Path): Path to the output file

        Returns:
            str: The file name as path in lowercase, consisting of "_" instead of " "
        """
        return Path(os.path.join(output_path, title.lower().replace(" ", "_") + ".txt"))

    # endregion