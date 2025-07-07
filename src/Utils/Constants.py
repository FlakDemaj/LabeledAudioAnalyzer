from pathlib import Path

# Base_paths
base_path: Path = Path.cwd()
base_results_path = base_path / "Data" / "Results"

# Excels
base_input_excel_path = base_path / "Data" / "Input" / "Excel"
result_excel_path: Path = base_results_path / "Excel" / "labeled_audios_with_real_df_label.xlsx"

# PDFs
confusion_matrix_pdf_output_path: Path = base_results_path / "PDFs" / "confusion_matrix.pdf"
logistic_regression_coefficients_pdf_output_path: Path = base_results_path / "PDFs" / "logistic_regression_coefficients.pdf"
correlations_pdf_output_path: Path = base_results_path / "PDFs" / "correlations.pdf"
