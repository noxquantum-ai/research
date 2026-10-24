# Strata review — GPT-6 Luna, xhigh

Reviewed read-only at the user’s request on October 24, 2026. No model weights
were downloaded and no external implementation code was copied.

[Strata’s Gyro documentation](https://github.com/Niko1221/Strata/blob/main/docs/GYRO.md)
describes trellis-coded MoE weights, not a tensor-network compressor. Its
[trellis layout](https://github.com/Niko1221/Strata/blob/main/third_party/ggml/ggml-common-tq.h)
and [QTIP’s primary paper](https://papers.neurips.cc/paper_files/paper/2024/file/6de2e84b8da47bb2eb5e2ac96c63d2b0-Paper-Conference.pdf)
suggest a potential codec for fixed-shape NEC amplitudes/cores. This would
change coefficient values and require model-level validation. It cannot be
reported as additional structural reduction. Codebooks, scales and framing
must be counted. No such codec is accepted or deployed here.

The [signed-Hadamard identity](https://github.com/Niko1221/Strata/blob/main/include/strata/kernels/cpu/hadamard.hpp)
uses a diagonal sign matrix D with D²=I: W′=WDHᵀ and x′=HDx, so W′x′=Wx.
Product-local orthogonal transforms preserve the Schmidt spectrum at a cut;
nonlocal signs/transforms require measurement. An independently implemented
Hadamard screen preserved the linear function to roundoff but worsened the
compressed layer-0 MLP control. It is not promoted.

The source packer copies already-quantized blobs; it is not a ready NEC
encoder. Strata’s relevant kernels are MIT licensed; the official QTIP
implementation is GPL-3.0. The review used papers and source inspection, not
implementation copying.
