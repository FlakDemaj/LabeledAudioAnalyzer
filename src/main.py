from Entity.LabeledAudioWithSolutionEntity import LabeledAudioWithSolutionEntity
from Evaluation.Analyzer import Analyzer
from Handler.ExcelHandler import ExcelHandler

from typing import List

if __name__ == '__main__':
    excel_reader = ExcelHandler()

    labeled_audios: List[LabeledAudioWithSolutionEntity] = excel_reader.read_and_create_excel()

    analyzer = Analyzer()

    analyzer.create_confusion_matrix(labeled_audios)

    analyzer.starting_logistic_regression(labeled_audios)

    analyzer.create_correlation_matrix(labeled_audios)

    analyzer.create_boxplots(labeled_audios)
