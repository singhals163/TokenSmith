import re
from typing import List, Optional

from tests.metrics.base import MetricBase


class BleuMetric(MetricBase):
    """Sentence-level BLEU-4 with NIST method-1 smoothing."""

    def __init__(self):
        self._available = False
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
            self._sentence_bleu = sentence_bleu
            self._smoothing = SmoothingFunction().method1
            self._available = True
        except Exception as e:
            print(f"BleuMetric initialization failed: {e}")

    @property
    def name(self) -> str:
        return "bleu"

    @property
    def weight(self) -> float:
        return 0.0

    def is_available(self) -> bool:
        return self._available

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def calculate(self, answer: str, expected: str, keywords: Optional[List[str]] = None) -> float:
        if not self._available or not answer.strip() or not expected.strip():
            return 0.0
        ref = self._tokenize(expected)
        hyp = self._tokenize(answer)
        if not ref or not hyp:
            return 0.0
        try:
            return float(self._sentence_bleu([ref], hyp, smoothing_function=self._smoothing))
        except Exception as e:
            print(f"BLEU calculation failed: {e}")
            return 0.0
