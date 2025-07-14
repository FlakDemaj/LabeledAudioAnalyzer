from pathlib import Path

# Base_paths
base_path: Path = Path.cwd()
base_results_path = base_path / "Data" / "Results"
base_pdf_path = base_results_path / "PDFs"
base_txt_path = base_results_path / "Texts"

# Excels
base_input_excel_path = base_path / "Data" / "Input" / "Excel"
result_excel_path: Path = base_results_path / "Excel" / "labeled_audios_with_real_df_label.xlsx"

# PDFs
confusion_matrix_base_pdf_output_path: Path = base_pdf_path / "ConfusionMatrix"
correlations_pdf_output_path: Path = base_pdf_path / "correlations.pdf"
roc_pdf_output_path : Path = base_pdf_path / "ROC.pdf"
recall_precision_pdf_output_path : Path = base_pdf_path / "recall_precision.pdf"
recall_precision_with_human_labels_pdf_output_path : Path = base_pdf_path / "recall_precision_with_human_labels.pdf"
cluster_analyse_pdf_output_path: Path = base_pdf_path / "ClusterAnalyse" / "cluster_analyse_ppq55_jitter.pdf"

# Txts
logistic_regression_coefficients_text_output_path: Path = base_txt_path / "logistic_regression_coefficients.txt"
confusion_matrix_metric_values_text_output_path: Path = base_txt_path