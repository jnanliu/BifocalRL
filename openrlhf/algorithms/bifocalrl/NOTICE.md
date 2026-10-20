# Provenance

The `mixed_reward/`, `normalization.py`, `voting.py`, and `legacy/` modules were
migrated from the local BifocalRL implementation. Their runtime imports are local
to this package. Historical fixed and reliability fusion are covered by parity
fixtures generated from the source implementation. Current BifocalRL uses
fixed-threshold GD2PO SNR filtering on the original normalized MV/SC signals,
attenuates SC answer-cluster means among retained responses, then applies equal
fusion weights, query reweighting and retained-batch normalization as documented
in the example README. The answer-conditioned SC decomposition is a local
extension, not part of the GD2PO paper.

Code evaluation delegates to VERL's optional Sandbox Fusion `compute_score`
implementation. Its fractional test score is logged separately; benchmark
pass metrics use binary all-tests-passed outcomes. No VERL source is vendored.

`legacy/math.py` retains the PRIME/TTRL Apache-2.0 notice in its source. Other
ported files follow their source repository's license. No CoVo source files were
copied; its paper formulas were independently implemented. The CIRCA
cluster/format rewards, GDPO advantages and KL-Cov loss were independently
implemented from the paper's equations and hyperparameters; no author code
was available at the linked paper when this implementation was written.

References:

- BifocalRL: https://github.com/jnanliu/BifocalRL
- slime checkout base: https://github.com/jnanliu/slime/tree/9f40ac0c59e14acdee336f31ab0fe07d496d000e
- OpenRLHF checkout base: https://github.com/jnanliu/OpenRLHF/tree/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53
- ETTRL: https://arxiv.org/abs/2508.11356 (v2)
- CoVo: https://arxiv.org/abs/2506.08745 (v1), https://github.com/sastpg/CoVo
- PRISM: https://arxiv.org/abs/2601.04700 (v1)
- GenPRM interface: https://github.com/RyanLiu112/GenPRM
- CIRCA + KL-Cov: https://arxiv.org/abs/2605.22620 (v1), Eqs. 2--8 and Appendix F
- GD2PO filtering/reweighting: https://arxiv.org/abs/2606.16771 (v1), Eqs. 10--12 and 19
- VERL code scoring: https://github.com/volcengine/verl/tree/main/verl/utils/reward_score/sandbox_fusion
