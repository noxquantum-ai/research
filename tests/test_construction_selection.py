import unittest
from nox.construction_selection import construction_improves


class ConstructionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(mean_nll=2., teacher_kl_nats_per_token=.5,
                         next_token_correct=60, dense_agreement_correct=70, scored_tokens=100)

    def test_all_metrics_are_required_and_equal_candidates_are_not_progress(self):
        better = {**self.base, 'mean_nll':1.9, 'teacher_kl_nats_per_token':.4}
        self.assertTrue(construction_improves(better, self.base))
        self.assertFalse(construction_improves(self.base, self.base))
        for key, value in [('mean_nll',2.1), ('next_token_correct',59),
                           ('dense_agreement_correct',69), ('teacher_kl_nats_per_token',float('nan'))]:
            self.assertFalse(construction_improves({**better,key:value},self.base))

    def test_different_sample_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            construction_improves({**self.base,'scored_tokens':99}, self.base)


if __name__=='__main__':unittest.main()
