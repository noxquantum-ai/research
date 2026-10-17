import torch

from nox.spectra import (
    llama_projection_modes,
    minimal_rank_for_tail,
    named_tt_spectrum_audit,
    singular_spectrum,
    spectra_summary,
    tt_parameter_count,
    tt_rank_profile,
    tt_site_unfoldings,
)
from nox.tensor_order import mode_index_permutation
from scripts.audit_nec_schmidt_components import _artifact_components


def test_llama_projection_modes_match_nec_compiler_layout():
    assert llama_projection_modes(4096, 4096) == ((8, 8, 8, 8), (8, 8, 8, 8))
    assert llama_projection_modes(11008, 4096) == ((43, 4, 8, 8), (8, 8, 8, 8))
    assert llama_projection_modes(4096, 11008) == ((8, 8, 8, 8), (43, 4, 8, 8))


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


def test_named_tt_spectrum_audit_compares_matched_variants():
    generator = torch.Generator().manual_seed(513)
    weight = torch.randn(16, 16, generator=generator)
    separable = torch.outer(
        torch.randn(16, generator=generator),
        torch.randn(16, generator=generator),
    )
    report = named_tt_spectrum_audit(
        {"weight": weight, "separable": separable},
        (2, 2, 2, 2), (2, 2, 2, 2), (1, 2, 3, 2, 1),
        exact_large_cuts=True,
    )
    assert report["physical_dimensions"] == [4, 4, 4, 4]
    assert list(report["variants"]) == ["weight", "separable"]
    for cuts in report["variants"].values():
        assert len(cuts) == 3
        for cut_index, cut in enumerate(cuts):
            assert cut["cut_after_site"] == cut_index
            assert cut["configured_retained_rank"] == (2, 3, 2)[cut_index]
            assert 0 <= cut["configured_rank_residual_energy_fraction"] <= 1
            assert cut["schmidt_entropy_nats"] is not None
            assert cut["effective_schmidt_rank"] >= 1
            assert cut["reported_spectrum_partial_entropy_nats"] >= 0


def test_component_audit_restores_nonidentity_artifact_coordinates():
    modes = (2, 2, 2, 2)
    order = (3, 2, 1, 0)
    out_perm = mode_index_permutation(modes, order)
    in_perm = mode_index_permutation(modes, order)
    weight = torch.zeros(16, 16)
    sparse = torch.zeros_like(weight)
    sparse[3, 11] = 2.5
    left = torch.arange(1, 17, dtype=torch.float32)
    right = torch.arange(17, 33, dtype=torch.float32)
    bridge = torch.outer(left, right)
    scale = torch.linspace(0.5, 2.0, 16)
    stored_sparse = sparse.index_select(0, out_perm).index_select(1, in_perm)
    indices = torch.nonzero(stored_sparse)
    artifact = {
        "orig_shape": (16, 16),
        "original_o_modes": modes,
        "original_i_modes": modes,
        "site_order": order,
        "sparse_indices": indices,
        "sparse_values": stored_sparse[indices[:, 0], indices[:, 1]],
        "A_bridge": left.index_select(0, out_perm).unsqueeze(1),
        "B_bridge": right.index_select(0, in_perm).unsqueeze(0),
        "d_awt": scale.index_select(0, in_perm),
    }
    restored_sparse, restored_bridge, restored_scale, out_modes, in_modes, restored_order = (
        _artifact_components(weight, artifact, device=torch.device("cpu"))
    )
    torch.testing.assert_close(restored_sparse, sparse)
    torch.testing.assert_close(restored_bridge, bridge)
    torch.testing.assert_close(restored_scale, scale)
    assert out_modes == in_modes == modes
    assert restored_order == order
