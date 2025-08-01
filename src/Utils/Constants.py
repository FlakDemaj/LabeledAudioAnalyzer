import os

from pathlib import Path

# region general

base_path: Path = Path.cwd()

# endregion

# region input

# base
base_input_path: str = os.path.join(base_path, 'src','Data', 'Input')
base_excel_input_path: str = os.path.join(base_input_path, 'Excel')
base_confusion_matrix_data_input_path: str = os.path.join(base_input_path, 'ConfusionMatrixData')

# Excels
labeled_audios_excel_input_path: Path = Path(base_excel_input_path) / "labeled_audios.xlsx"
real_labeled_audios_excel_input_path: Path = Path(base_excel_input_path) / "real_labeled_audios.xlsx"

# Txts
ai_human_emulation_confusion_matrix_input_path: str = os.path.join(base_confusion_matrix_data_input_path, "ai_human_emulation_confusion_matrix.txt")
ai_labeling_confusion_matrix_input_path: str = os.path.join(base_confusion_matrix_data_input_path, "ai_labeling_confusion_matrix.txt")
human_labeling_confusion_matrix_input_path: str = os.path.join(base_confusion_matrix_data_input_path, "human_labeling_confusion_matrix.txt")

ai_results_input_path: str = os.path.join(base_input_path, "Results", "ai_result.txt")


# endregion

# region output

# Base_paths
base_results_path = base_path / "src" / "Data" / "Results"
base_pdf_path = base_results_path / "PDFs"
base_txt_path = base_results_path / "Texts"
base_input_excel_path = base_path / "Data" / "Input" / "Excel"

# Excels
result_excel_output_path: Path = base_results_path / "Excel" / "labeled_audios_with_real_df_label.xlsx"

# PDFs
confusion_matrix_base_pdf_output_path: Path = base_pdf_path / "ConfusionMatrix"
correlations_pdf_output_path: Path = base_pdf_path / "correlations.pdf"
roc_ai_pdf_output_path : Path = base_pdf_path / "ROC" / "roc_ai_perception.pdf"
roc_with_human_perception_pdf_output_path : Path = base_pdf_path / "ROC" / "roc_ai_with_human_perception.pdf"

# Txts
logistic_regression_coefficients_text_output_path: Path = base_txt_path / "logistic_regression_coefficients.txt"
confusion_matrix_metric_values_text_output_path: Path = base_txt_path
roc_threshold_text_output_path: Path = base_txt_path
recall_precision_human_text_output_data: Path = base_txt_path

# endregion