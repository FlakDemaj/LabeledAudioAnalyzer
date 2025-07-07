import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as mcolors

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from typing import List, Optional
from pathlib import Path

from src.Utils.Constants import confusion_matrix_pdf_output_path
from src.Utils.Logger import get_logger
from src.Utils.Constants import logistic_regression_coefficients_pdf_output_path
from src.Entity.LabeledAudioWithSolutionEntity import (LabeledAudioWithSolutionEntity)
from src.Handler.PDFHandler import PDFHandler
from src.Utils.Constants import correlations_pdf_output_path, base_results_path

class Analyzer:
    def __init__(self):
        self.PDFHandler = PDFHandler()
        self.Logger = get_logger(__name__)
        self.Logistic_regression_model: LogisticRegression = LogisticRegression()
    #region confusionmatrix

    def create_confusion_matrix(self, labeled_audios: List[LabeledAudioWithSolutionEntity]) -> None:
        """
        Generates and displays a confusion matrix for human-labeled audio data compared to ground truth.

        Optionally saves the confusion matrix as a PDF file to the predefined results directory.

        :param labeled_audios: List of audio entities with human labels and ground truth labels.
        :type labeled_audios: List[LabeledAudioWithSolutionEntity]
        :return: None
        :raises RuntimeError: If saving the PDF fails due to file system issues (e.g., permissions).
        """
        self.Logger.info("Create confusion matrix...")
        # Extract true labels and predicted labels from the data
        y_true = [audio.is_truly_df for audio in labeled_audios]
        y_pred = [audio.human_df_label for audio in labeled_audios]

        # Define class labels
        labels = ["negative", "positive"]

        # Compute the confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

        # Create the plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, cbar=True)

        # Set axis labels and title
        plt.xlabel("Human Prediction")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        self.Logger.info("Create confusion matrix done.")

        self.PDFHandler.safe_pdf(confusion_matrix_pdf_output_path, plt)
        # Show the plot
        plt.show()

    #endregion

    #region logistic regression

    def starting_logistic_regression(self, labeled_audios_list: List[LabeledAudioWithSolutionEntity]) -> None:
        """
        Performs logistic regression analysis to predict human-assigned deepfake labels
        using subjective audio scores (naturalness, emotionality, rhythm).

        The method trains the model on 80% of the input data and tests on the remaining 20%.
        It logs the accuracy on the test set and the feature coefficients.

        Args:
            labeled_audios_list (List[LabeledAudioWithSolutionEntity]):
                A list of labeled audio samples with both subjective scores and ground truth labels.

        Returns:
            None
        """

        if not labeled_audios_list:
            self.Logger.warning("Input list is empty. Aborting analysis.")
            return

        self.Logger.info("Converting input list to pandas DataFrame...")
        labeled_audios_df: pd.DataFrame = self.list_to_dataframe_converter(labeled_audios_list)
        self.Logger.info("DataFrame created successfully.")

        self.Logger.info("Preparing training and testing data...")
        # Feature matrix
        x: pd.DataFrame = labeled_audios_df[["naturalness", "emotionality", "rhythm"]]
        # Target: human-assigned deepfake label
        y: pd.Series = labeled_audios_df["human_label"]

        # Split data into training (80%) and testing (20%) sets
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        # Train the logistic regression model
        self.Logistic_regression_model.fit(x_train, y_train)
        y_pred: Optional[pd.Series] = pd.Series(self.Logistic_regression_model.predict(x_test))

        self.Logger.info("Model training and prediction completed.")

        # Log accuracy on the test set
        accuracy: float = accuracy_score(y_test, y_pred)
        self.Logger.info(f"Accuracy (test set): {accuracy:.4f}")

        # Log feature coefficients
        coeff_df: pd.DataFrame = pd.DataFrame({
            "Feature": x.columns,
            "Coefficient": self.Logistic_regression_model.coef_[0]
        })
        self.Logger.info(f"Model coefficients:\n{coeff_df}")
        self.plot_coefficients()

    def plot_coefficients(self) -> None:
        """
        Plots the coefficients of the trained logistic regression model
        to visualize feature importance and direction (positive/negative).

        Returns:
            None
        """
        if not hasattr(self.Logistic_regression_model, "coef_"):
            self.Logger.warning("Model has not been trained yet. Cannot plot coefficients.")
            return

        # Extract feature names and coefficients
        features: List[str] = ["naturalness", "emotionality", "rhythm"]
        coefficients: List[float] = self.Logistic_regression_model.coef_[0].tolist()

        abs_coeffs: np.ndarray = np.abs(coefficients)
        stretched: np.ndarray = np.power(abs_coeffs, 0.7)  # stretch small values slightly
        norm: np.ndarray = (stretched - stretched.min()) / (np.ptp(stretched) + 1e-6)

        # --- Create custom colormap
        cmap: mcolors.Colormap = cm.get_cmap("Blues")
        custom_cmap: mcolors.ListedColormap = mcolors.ListedColormap(
            cmap(np.linspace(0.2, 1.5, 256))  # avoid very light blue
        )
        colors: List[tuple] = [custom_cmap(val) for val in norm]

        # Create horizontal bar chart
        plt.figure(figsize=(8, 4))
        plt.barh(features, coefficients, color=colors)
        plt.xlabel("Coefficient Value")
        plt.title("Logistic Regression Coefficients")
        plt.axvline(x=0, color="gray", linestyle="--")
        plt.tight_layout()

        self.PDFHandler.safe_pdf(logistic_regression_coefficients_pdf_output_path, plt)

        plt.show()

    #endregion

    #region correlationmatrix

    def create_correlation_matrix(self, labeled_audios_list: List[LabeledAudioWithSolutionEntity]) -> None:
        """
        Computes and visualizes the Pearson correlation matrix between selected features
        ('emotionality', 'rhythm', 'naturalness') based on labeled audio data.

        The correlation matrix is shown as a heatmap with annotated values, saved as a PDF,
        and displayed interactively.

        Args:
            labeled_audios_list (List[LabeledAudioWithSolutionEntity]): List of labeled audio data objects.

        Returns:
            None
        """
        self.Logger.info("Creating correlation matrix...")

        # Convert the list of labeled audios to a DataFrame
        labeled_audios_df: pd.DataFrame = self.list_to_dataframe_converter(labeled_audios_list)

        # Define the features for which to compute correlations
        features: List[str] = ["emotionality", "rhythm", "naturalness"]

        # Calculate the Pearson correlation matrix for the selected features
        corr_matrix: pd.DataFrame = labeled_audios_df[features].corr(method="pearson")

        # Create a heatmap visualization of the correlation matrix
        sns.heatmap(corr_matrix, annot=True, cmap="Blues", vmin=0.3, vmax=1)
        plt.title("Korrelationen zwischen Bewertungskriterien")

        self.Logger.info("Created correlation matrix.")

        # Save the heatmap plot as a PDF using the PDF handler
        self.PDFHandler.safe_pdf(correlations_pdf_output_path, plt)

        # Display the heatmap interactively
        plt.show()

    #endregion

    #region boxplot
    def create_boxplots(self, labeled_audios_list: List[LabeledAudioWithSolutionEntity]) -> None:
        """
        Creates boxplots comparing Deepfake and Real audio samples for multiple features
        based on both perceived (human_label) and true (true_label) labels,
        and performs independent t-tests to check for significant differences.

        Each boxplot visualizes the distribution of one feature (naturalness, emotionality, rhythm)
        across the two groups. The t-statistic and p-value from the t-test are displayed on the plot.

        The plots are saved as PDFs named after each feature and label type.

        Args:
            labeled_audios_list (List[LabeledAudioWithSolutionEntity]): List of labeled audio data objects.

        Returns:
            None
        """
        # Convert the list of labeled audios to a DataFrame for easier processing
        labeled_audios_df: pd.DataFrame = self.list_to_dataframe_converter(labeled_audios_list)

        for feature in ["naturalness", "emotionality", "rhythm"]:
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
                label_desc = "Menschliche Wahrnehmung" if label_type == "human_label" else "Tatsächliches Resultat"
                plt.title(f"{feature.capitalize()}: Deepfakes vs. echte Stimmen\n({label_desc})")
                plt.xlabel("Deepfake (True = ja)")
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
                plt.show()

    #endregion

    #region Helper
    def list_to_dataframe_converter(
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

        Args:
            labeled_audios_list (List[LabeledAudioWithSolutionEntity]):
                The list of labeled audio entities to convert.

        Returns:
            pd.DataFrame: A DataFrame with the relevant features and labels.
        """

        self.Logger.info("Converting input list to pandas DataFrame...")

        df: pd.DataFrame = pd.DataFrame([{
            "naturalness": x.naturalness_score,
            "emotionality": x.emotionality_score,
            "rhythm": x.rhythm_score,
            "human_label": x.human_df_label,
            "true_label": x.is_truly_df
        } for x in labeled_audios_list])

        self.Logger.info("DataFrame created successfully.")

        return df
    #endregion