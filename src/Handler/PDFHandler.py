from pathlib import Path
from src.Utils.Logger import get_logger
from matplotlib import pyplot

class PDFHandler:

    def __init__(self):
        # Initialize the logger for this handler
        self.Logger = get_logger(__name__)

    def safe_pdf(self, output_path: Path, plt: pyplot) -> None:
        """
        Saves a Matplotlib plot as a PDF to the given path.

        Args:
            output_path (Path): Full path including the filename (e.g., 'results/plot.pdf')
            plt: Matplotlib pyplot object (usually passed as 'plt')

        Raises:
            RuntimeError: If saving the PDF fails due to an exception.
        """
        try:
            if output_path.exists():
                pass
            else:
                # Ensure the parent directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                self.Logger.info("Saving plot as PDF...")
                plt.savefig(output_path, format="pdf")
                self.Logger.info(f"PDF saved to: {output_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to save PDF: {e}")
