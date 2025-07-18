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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, \
    precision_score, recall_score
from scipy.stats import ttest_ind
from typing import List, Dict
from pathlib import Path
from ast import literal_eval

from src.Utils.Logger import get_logger
from src.Entity.LabeledAudioWithSolutionEntity import LabeledAudioWithSolutionEntity
from src.Entity.ConfusionMatrixDataEntity import ConfusionMatrixDataEntity
#from src.Extractor.FeatureExtractor import FeatureExtractor
from src.Handler.PDFHandler import PDFHandler
from src.Utils.Constants import base_results_path

class Analyzer:
    def __init__(self):

        self.PDFHandler = PDFHandler()
        self.Logger: logging.Logger = get_logger(__name__)
        self.Logistic_regression_model: LogisticRegression = LogisticRegression()
        #self.FeatureExtractor: FeatureExtractor = FeatureExtractor()

    #region confusionmatrix

    def create_confusion_matrix(self,
                                output_pdf_path:Path,
                                output_metrix_txt_path:Path,
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
        # Show the plot
        plt.show()

        # Creates the metrics for the confusion matrix
        tn, fp = confusion_matrix[0]
        fn, tp = confusion_matrix[1]

        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

        self.create_confusion_matrix_metric(
            output_metrix_txt_path,
            ConfusionMatrixDataEntity(tp, tn, fp, fn),
            title)


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
        confusion_matrix_data: ConfusionMatrixDataEntity = self.load_confusion_matrix_data(input_path)

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
        plt.show()

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
            "Sensitivity": confusion_matrix_entity.sensitive,
            "Specificity": confusion_matrix_entity.specificity,
            "Precision": confusion_matrix_entity.precision,
            "F1-Score": confusion_matrix_entity.f_score,
        }

        self.Logger.info(f"\n📊 Metrics for '{title}':\n" + tabulate(list(metrics_dict.items()), headers=["Metric", "Value"],
                                                                    tablefmt="fancy_grid", floatfmt=".4f"))

        self.Logger.info(f"Metrics created for {title}.")

        self.write_result(self.create_file_name(output_path, title),
                          metrics_dict)

    #endregion

    #region logistic regression

    def starting_logistic_regression(self,
                                     output_path: Path,
                                     labeled_audios_list: List[LabeledAudioWithSolutionEntity]
                                     ) -> None:
        """
        Performs logistic regression analysis to predict human-assigned deepfake labels
        using subjective audio scores (naturalness, emotionality, rhythm).

        Trains model on 80% and tests on 20%, logs accuracy and feature coefficients with confidence intervals.

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
        labeled_audios_df: pd.DataFrame = self.list_to_dataframe_converter(labeled_audios_list)
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
        result = model.fit(disp=False)  # disp=False unterdrückt Output im Terminal

        # Vorhersage auf Testdaten (Wahrscheinlichkeiten)
        y_pred_prob = result.predict(X_test_sm)
        # Klassifizieren mit Schwellenwert 0.5
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Accuracy berechnen
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        self.Logger.info(f"Accuracy (test set): {accuracy:.4f}")

        # Konfidenzintervalle berechnen
        conf_int = result.conf_int(alpha=0.05)
        conf_int.columns = ['2.5%', '97.5%']

        # Koeffizienten, Odds Ratios und Konfidenzintervalle in DataFrame
        coef = result.params
        odds_ratios = np.exp(coef)
        conf_int_exp = np.exp(conf_int)

        summary_df = pd.DataFrame({
            'Coefficient': coef,
            'Odds Ratio': odds_ratios,
            'CI Lower (2.5%)': conf_int_exp['2.5%'],
            'CI Upper (97.5%)': conf_int_exp['97.5%']
        })

        accuracy_df = pd.DataFrame({
            'Coefficient': [np.nan],
            'Odds Ratio': [np.nan],
            'CI Lower (2.5%)': [np.nan],
            'CI Upper (97.5%)': [accuracy]
        }, index=['Accuracy'])

        summary_df = pd.concat([summary_df, accuracy_df])

        self.Logger.info("\nLogistic regression results with 95% confidence intervals:\n" +
                         tabulate(summary_df, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))

        self.write_result(output_path, summary_df.to_dict(orient="index"))


    #endregion

    #region correlationmatrix

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
        labeled_audios_df: pd.DataFrame = self.list_to_dataframe_converter(labeled_audios_list)

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
        plt.show()

    #endregion

    #region boxplot
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
        labeled_audios_df: pd.DataFrame = self.list_to_dataframe_converter(labeled_audios_list)

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
                plt.show()

    #endregion

    #regoion roc

    def create_roc(self,
                   input_path: str,
                   output_path: Path,
                   title: str) -> None:
        """
        Loads prediction results from a file, computes the ROC curve, and displays it using matplotlib.

        The method assumes the input file contains a list of dictionaries, each with:
        - "result": the predicted score or probability
        - "label": the ground truth label (0 or 1)

        Args:
            input_path (str): Path to the file containing prediction results.
            output_path (Path): Path where the file should be saved.
            title (str): The Title for the plot.
        """

        self.Logger.info("Starting creating roc analysis...")
        # Load prediction results from the specified file path
        result: List[Dict[str, float | int]] = self.read_result_list(input_path)

        # Extract the predicted scores and true labels from the result list
        y_score = [entry["result"] for entry in result]
        y_pred = [entry["label"] for entry in result]

        # Compute the false positive rate, true positive rate, and thresholds for the ROC curve
        fpr, tpr, thresholds = roc_curve(y_pred, y_score)

        # Calculate the Area Under the Curve (AUC) for the ROC curve
        roc_auc = auc(fpr, tpr)

        self.Logger.info("Roc analysis is done.")

        # Create a new figure for the plot
        plt.figure()

        # Plot the ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')

        # Plot the diagonal line representing a random classifier
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # Set axis limits and labels
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # Add plot title and legend
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)

        self.PDFHandler.safe_pdf(output_path, plt)

        # Display the plot
        plt.show()

    #endregion

    #region Precision-Recall

    def create_precision_recall(self,
                                input_path: str,
                                output_path: Path,
                                title: str) -> None:
        """
        Generates a Precision-Recall curve based on prediction results and saves it as a PDF.

        This method loads a list of dictionaries containing prediction scores ('result') and
        true labels ('label') from a specified input file. It calculates the precision and
        recall values across various thresholds, plots the Precision-Recall curve, and includes
        the Average Precision (AP) score in the legend. The resulting plot is saved to a PDF file.

        Args:
            input_path (str): Path to the input file containing prediction results.
            output_path (Path): Path where the resulting PDF plot will be saved.
            title (str): Title of the plot.
        Returns:
            None
        """
        self.Logger.info("Starting creating precision recall analysis...")

        # Load prediction results from the input file
        result: List[Dict[str, float | int]] = self.read_result_list(input_path)

        # Extract predicted scores and true labels from the result list
        y_score = [entry["result"] for entry in result]
        y_pred = [entry["label"] for entry in result]

        # Compute precision-recall pairs for different probability thresholds
        precision, recall, thresholds = precision_recall_curve(y_pred, y_score)

        # Compute the average precision score across all thresholds
        avg_precision = average_precision_score(y_pred, y_score)

        self.Logger.info("Precision recall analysis is done.")

        # Plot the Precision-Recall curve
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a PDF using the custom PDF handler
        self.PDFHandler.safe_pdf(output_path, plt)

        # Show the plot (useful during development or interactive use)
        plt.show()

    def create_precision_recall_with_human_labels(self,
                                                  input_path: str,
                                                  output_path: Path,
                                                  labeled_audios_list: List[LabeledAudioWithSolutionEntity],
                                                  title: str) -> None:
        """
           Creates and saves a precision-recall curve comparing model predictions with human labels.

           This method loads model results from a file, converts a list of labeled audio entities
           into a DataFrame, and then computes precision and recall for both the model scores
           and human labels. It plots the precision-recall curve for the model and marks
           the human performance as a single point on the plot. The plot is saved as a PDF file.

           Args:
               input_path (str): Path to the input file containing model results.
               output_path (Path): Path where the generated PDF plot will be saved.
               labeled_audios_list (List[LabeledAudioWithSolutionEntity]): List of labeled audio entities with human labels.
               title (str): Title of the plot.
           Returns:
               None
           """

        self.Logger.info("Starting creating with human labels precision recall analysis...")

        result: List[Dict[str, float | int]] = self.read_result_list(input_path)
        labeled_audios_df: pd.DataFrame = self.list_to_dataframe_converter(labeled_audios_list)

        true_labels = [entry["label"] for entry in result]
        human_labels = labeled_audios_df["human_label"]
        ki_scores = [entry["result"] for entry in result]

        precision_ki, recall_ki, thresholds = precision_recall_curve(true_labels, ki_scores)

        precision_human = precision_score(true_labels, human_labels)
        recall_human = recall_score(true_labels, human_labels)

        self.Logger.info("Precision recall with human labels analysis is done.")

        plt.figure(figsize=(8, 6))
        plt.plot(recall_ki, precision_ki, label='AI-Model PR-Curve')
        plt.scatter(recall_human, precision_human, color='red', label='Human-Performance', zorder=5, s=100)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)

        self.PDFHandler.safe_pdf(output_path, plt)
        plt.show()

    #endregion
    '''
    #region ClusterAnalyse
    def start_cluster_analyse(self,
                              labeled_audios: List[LabeledAudioWithSolutionEntity],
                              input_path: str,
                              output_path: Path) -> None:
        """     
        Performs a clustering analysis based on the PPQ55_Jitter feature and
        plots the results with color-coding based on true labels (blue = real voice, red = deepfake).

        Args:
            labeled_audios (List[LabeledAudioWithSolutionEntity]): List of labeled audio entities with human labels.
            input_path (str): Path to the folder containing input .wav audio files.
            output_path (Path): Path where the resulting PDF plot should be saved.
                
        Returns:
            None
        """

        self.Logger.info("Starting cluster analysis using the PPQ55_Jitter feature...")

        ppq55_jitter_values: List[float] = []  # Stores jitter values for each file
        file_names: List[str] = []  # Stores corresponding file names

        # --- Extract PPQ55_Jitter values for all .wav files
        for file in tqdm(labeled_audios):
            if file.audio_title.endswith(".wav"):
                file_path = os.path.join(input_path, file.audio_title)
                ppq55_jitter: float = self.FeatureExtractor.get_ppq55_jitter(file_path)

                if ppq55_jitter is not None:
                    ppq55_jitter_values.append(ppq55_jitter)
                    file_names.append(file.audio_title)

        # --- Safety check
        if not ppq55_jitter_values:
            self.Logger.warning("No valid jitter values were found.")
            return

        # --- Convert list to 2D array for clustering
        X: np.ndarray = np.array(ppq55_jitter_values).reshape(-1, 1)

        # --- Run clustering (e.g., KMeans)
        labels = self.KMeansClusterClient.fit_predict(X)
        self.Logger.info("Clustering based on PPQ55_Jitter completed.")

        # --- Load true labels from the labeled audio objects
        labeled_audios_df: pd.DataFrame = self.list_to_dataframe_converter(labeled_audios)
        true_labels_map = labeled_audios_df.set_index("audio_title")["true_label"].to_dict()
        true_labels = [true_labels_map.get(filename, None) for filename in file_names]

        # --- Assign colors based on true labels: red for deepfake, blue for real
        colors = ["red" if label == 1 else "blue" for label in true_labels]

        # --- Create scatter plot without sorting
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(X)), X.flatten(), c=colors, edgecolors='k')
        plt.xlabel("Audio File Index (unsorted)")
        plt.ylabel("PPQ55_Jitter Value")
        plt.title("PPQ55_Jitter – Real Voice (blue) vs Deepfake (red)")
        plt.axhline(np.mean(X), color="gray", linestyle="--", label="Mean Jitter")
        plt.legend()

        # --- Save PDF plot
        self.PDFHandler.safe_pdf(output_path, plt)
        plt.show()

        # --- Log results for inspection
        for fname, ppq, cluster, label in zip(file_names, ppq55_jitter_values, labels, true_labels):
            self.Logger.info(f"{fname}: PPQ55={ppq:.5f}, Cluster={cluster}, TrueLabel={label}")

    #endregion
    '''
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

    def write_result(self, output_path: Path, dictionary) -> None:
        """
         Writes a dictionary into a text file

         The data is expected to be a dictionary

         Args:
             output_path (Path): Path to the output file
             dictionary (dict): Dictionary to be written

         Returns:
             None
         """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.Logger.info(f"Writing text file to {output_path}...")

        with open(output_path, "w") as file:
            file.write(str(dictionary))

        self.Logger.info(f"Text file saved to {output_path}")

    @staticmethod
    def read_result_list(input_path: str) -> List[Dict[str, float | int]]:
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
    def load_confusion_matrix_data(input_path: str) -> ConfusionMatrixDataEntity:
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
    def create_file_name(output_path: Path, title: str) -> Path:
        """
        Creates a file name from the title and the output_file.

        Args:
            title (str): The title of the file which is shown in the logs
            output_path (Path): Path to the output file

        Returns:
            str: The file name as path in lowercase, consisting of "_" instead of " "
        """
        return Path(os.path.join(output_path, title.lower().replace(" ", "_") + ".txt"))

    #endregion