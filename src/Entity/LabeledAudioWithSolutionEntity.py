class LabeledAudioWithSolutionEntity:
    """
    Represents a labeled audio sample with subjective scores
    and both human and ground truth deepfake labels.
    """

    def __init__(
            self,
            audio_title: str,
            naturalness_score: int,
            emotionality_score: int,
            rhythm_score: int,
            human_df_label: bool,
            is_truly_df: bool):
        """
        Initializes a LabeledAudioWithSolutionEntity instance.

        :param audio_title: Title or identifier of the audio sample.
        :param naturalness_score: Rating for naturalness (e.g., 1-5).
        :param emotionality_score: Rating for emotionality (e.g., 1-5).
        :param rhythm_score: Rating for rhythm (e.g., 1-5).
        :param human_df_label: Human-assigned deepfake label (True if labeled as deepfake).
        :param is_truly_df: Ground truth deepfake label (True if actually a deepfake).
        """
        self.audio_title: str = audio_title
        self.naturalness_score: int = naturalness_score
        self.emotionality_score: int = emotionality_score
        self.rhythm_score: int = rhythm_score
        self.human_df_label: bool = human_df_label
        self.is_truly_df: bool = is_truly_df
