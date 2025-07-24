class ConfusionMatrixDataEntity:
    def __init__(self,
                 true_positive: int,
                 true_negative:int,
                 false_positive: int,
                 false_negative: int,):
        self.true_positive:int = true_positive
        self.true_negative:int = true_negative
        self.false_positive:int = false_positive
        self.false_negative:int = false_negative

        self.accuracy:float = self.calculate_accuracy()
        self.sensitive:float = self.calculate_sensitive()
        self.false_predictive_value:float = self.calculate_false_predictive_value()
        self.specificity:float = self.calculate_specificity()
        self.precision:float = self.calculate_precision()
        self.f_score:float = self.calculate_f_score()


    def calculate_accuracy(self) -> float:
        return (self.true_positive + self.true_negative)/ (self.true_positive + self.true_negative + self.false_positive + self.false_negative)

    def calculate_sensitive(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative)

    def calculate_false_predictive_value(self) -> float:
        return self.true_negative / (self.true_negative + self.false_negative)

    def calculate_specificity(self) -> float:
        return self.true_negative / (self.true_negative + self.false_positive)

    def calculate_precision(self) -> float:
        return self.true_positive / (self.true_positive + self.false_positive)

    def calculate_f_score(self) -> float:
        return (2 * self.sensitive * self.precision) / (self.sensitive + self.precision)
