from pathlib import Path

# Base_paths
base_path: Path = Path.cwd()
base_results_path = base_path / "Data" / "Results"
base_pdf_path = base_results_path / "PDFs"

# Excels
base_input_excel_path = base_path / "Data" / "Input" / "Excel"
result_excel_path: Path = base_results_path / "Excel" / "labeled_audios_with_real_df_label.xlsx"

# PDFs
confusion_matrix_base_pdf_output_path: Path = base_pdf_path / "ConfusionMatrix"
logistic_regression_coefficients_pdf_output_path: Path = base_pdf_path / "logistic_regression_coefficients.pdf"
correlations_pdf_output_path: Path = base_pdf_path / "correlations.pdf"
barplot_human_labeling_pdf_output_path: Path = base_pdf_path / "barplots" /"barplot_human_labeling.pdf"
barplot_ai_labeling_pdf_output_path : Path = base_pdf_path / "barplots" /"barplot_ai_labeling.pdf"
roc_pdf_output_path : Path = base_pdf_path / "ROC.pdf"
recall_precision_pdf_output_path : Path = base_pdf_path / "recall_precision.pdf"
recall_precision_with_human_labels_pdf_output_path : Path = base_pdf_path / "recall_precision_with_human_labels.pdf"
cluster_analyse_pdf_output_path: Path = base_pdf_path / "ClusterAnalyse" / "cluster_analyse_ppq55_jitter.pdf"