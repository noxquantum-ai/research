import unittest
from types import SimpleNamespace

import torch

from nox.mmlu import (
    compare_mmlu_reference,
    format_question,
    prepare_example,
    resolve_mmlu_reference,
    score_batch,
)


class FixtureTokenizer:
    def encode(self, text, add_special_tokens=True):
        if len(text) >= 2 and text[-2] == " " and text[-1] in "ABCD":
            return list(range(len(text) - 2)) + [10000 + "ABCD".index(text[-1])]
        return list(range(len(text)))


class MMLUProtocolTests(unittest.TestCase):
    def setUp(self):
        self.row = {"question": "Which option?", "choices": ["one", "two", "three", "four"],
                    "answer": 2, "subject": "test_subject"}
        self.dev = [dict(self.row, question=f"Demonstration {index}?") for index in range(5)]

    def test_target_label_does_not_enter_prompt(self):
        tokenizer = FixtureTokenizer()
        first = prepare_example(tokenizer, self.row, self.dev)
        second = prepare_example(tokenizer, dict(self.row, answer=0), self.dev)
        self.assertEqual(first, second)
        self.assertEqual(first["shots"], 5)
        self.assertEqual(first["choice_token_ids"], [10000, 10001, 10002, 10003])
        self.assertTrue(format_question(self.row).endswith("Answer:"))
        self.assertTrue(format_question(self.row, True).endswith("Answer: C\n\n"))

    def test_context_reduction_removes_only_demonstrations(self):
        tokenizer = FixtureTokenizer()
        full = prepare_example(tokenizer, self.row, self.dev)
        shorter = prepare_example(tokenizer, self.row, self.dev, max_context=len(full["input_ids"]))
        self.assertLess(shorter["shots"], 5)
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            prepare_example(tokenizer, self.row, self.dev, max_context=5)
        with self.assertRaisesRegex(ValueError, "same subject"):
            prepare_example(tokenizer, self.row, [dict(d, subject="wrong") for d in self.dev])

    def test_scoring_uses_last_nonpadding_position_and_answer_choices(self):
        examples = [{"input_ids": [7, 8], "choice_token_ids": [2, 3, 4, 5]},
                    {"input_ids": [7, 8, 9, 10], "choice_token_ids": [2, 3, 4, 5]}]

        def model(**kwargs):
            self.assertFalse(kwargs["use_cache"])
            self.assertEqual(kwargs["attention_mask"].tolist(), [[1, 1, 0, 0], [1, 1, 1, 1]])
            logits = torch.zeros(2, 4, 16)
            logits[:, :, 0] = 100
            logits[0, 1, 3] = 10
            logits[1, 3, 5] = 10
            return SimpleNamespace(logits=logits)

        scores = score_batch(model, examples, 0, "cpu")
        self.assertEqual(scores.argmax(dim=1).tolist(), [1, 3])

    def test_external_reference_is_explicitly_marked_unmatched(self):
        reference = resolve_mmlu_reference(
            reference_accuracy=0.453,
            reference_source="Llama 2 paper Table 19, five-shot base 7B",
        )
        self.assertEqual(reference["kind"], "published_external_score")
        self.assertFalse(reference["protocol_match"])
        self.assertEqual(reference["accuracy"], 0.453)

    def test_matched_dense_reference_requires_complete_same_protocol(self):
        baseline = {"status": "COMPLETE", "variant": "dense", "accuracy": 0.44,
                    "protocol_fingerprint": "same"}
        reference = resolve_mmlu_reference(dense_result=baseline, protocol_fingerprint="same")
        self.assertTrue(reference["protocol_match"])
        with self.assertRaisesRegex(ValueError, "different protocol"):
            resolve_mmlu_reference(dense_result=dict(baseline, protocol_fingerprint="other"),
                                    protocol_fingerprint="same")
        with self.assertRaisesRegex(ValueError, "Choose either"):
            resolve_mmlu_reference(dense_result=baseline, reference_accuracy=0.453,
                                    protocol_fingerprint="same")

    def test_external_reference_requires_valid_score_and_source(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            resolve_mmlu_reference(reference_accuracy=45.3, reference_source="paper")
        with self.assertRaisesRegex(ValueError, "nonempty source"):
            resolve_mmlu_reference(reference_accuracy=0.453)

    def test_published_threshold_uses_only_the_complete_overall_score(self):
        reference = resolve_mmlu_reference(
            reference_accuracy=0.453,
            reference_source="Llama 2 paper Table 19, five-shot base 7B",
        )
        failed = compare_mmlu_reference(0.43, reference, full_test=True)
        self.assertFalse(failed["passes_published_reference_threshold"])
        self.assertAlmostEqual(failed["published_reference_95pct_threshold"], 0.43035)

        passed = compare_mmlu_reference(0.43035, reference, full_test=True)
        self.assertTrue(passed["passes_published_reference_threshold"])
        partial = compare_mmlu_reference(0.45, reference, full_test=False)
        self.assertFalse(partial["passes_published_reference_threshold"])
