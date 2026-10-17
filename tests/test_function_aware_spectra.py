import torch

from nox.spectra import (
    minimal_rank_for_tail,
    singular_spectrum,
    spectra_summary,
    tt_parameter_count,
    tt_rank_profile,
    tt_site_unfoldings,
)


def test_small_nec_tensorization_cut_dimensions_and_orderings():
    generator = torch.Generator().manual_seed(511)
    weight = torch.randn(16, 16, generator=generator)
    cuts_a, physical_a = tt_site_unfoldings(
        weight, (2, 2, 2, 2), (2, 2, 2, 2), (0, 1, 2, 3),
    )
    cuts_b, physical_b = tt_site_unfoldings(
        weight, (2, 2, 2, 2), (2, 2, 2, 2), (3, 2, 1, 0),
    )
    assert physical_a == physical_b == (4, 4, 4, 4)
    assert [tuple(x.shape) for x in cuts_a] == [(4, 64), (16, 16), (64, 4)]
    assert [tuple(x.shape) for x in cuts_b] == [(4, 64), (16, 16), (64, 4)]
    # Each cut is an exact reshaping/permutation, so squared singular values
    # sum to the original Frobenius energy.
    for cut in cuts_a + cuts_b:
        values = torch.linalg.svdvals(cut.double())
        assert torch.allclose(values.square().sum(), weight.double().square().sum(), rtol=1e-10, atol=1e-10)


def test_tt_rank_profile_meets_conservative_global_error_bound():
    generator = torch.Generator().manual_seed(512)
    weight = torch.randn(16, 16, generator=generator)
    cuts, physical = tt_site_unfoldings(
        weight, (2, 2, 2, 2), (2, 2, 2, 2), (0, 1, 2, 3),
    )
    spectra = [singular_spectrum(cut) for cut in cuts]
    summaries = [spectra_summary(s) for s in spectra]
    profile = tt_rank_profile(
        [s["singular_values"] for s in spectra], physical, 0.2,
        total_squares=[s["frobenius_norm_squared"] for s in spectra],
    )
    assert profile["coefficient_count"] == tt_parameter_count(physical, profile["ranks"])
    assert profile["tt_error_bound"] <= 0.2 + 1e-10
    assert all(
        item["min_rank_by_residual_energy_fraction"]["0.2"]["minimum_rank"] >= 1
        for item in summaries
    )


def test_tt_rank_profile_counts_unresolved_randomized_spectral_energy():
    values = [torch.tensor([0.6**0.5])] * 3
    profile = tt_rank_profile(
        values, (2, 2, 2, 2), 0.2, total_squares=(1.0, 1.0, 1.0),
    )
    assert profile["tt_error_bound"] > 0.2


def test_minimal_rank_residual_energy_accounting():
    values = torch.tensor([4.0, 2.0, 1.0])
    rank = minimal_rank_for_tail(values, 0.05)
    total = values.square().sum()
    residual = (total - values[:rank].square().sum()) / total
    assert rank >= 1
    assert residual <= 0.05
