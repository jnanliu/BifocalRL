# Provenance

The `mixed_reward/`, `normalization.py`, `voting.py`, and `legacy/` modules were
migrated from the local BifocalRL implementation. Their runtime imports are local
to this package. The numerical fusion behavior is retained and covered by parity
fixtures generated from the source implementation.

`legacy/math.py` retains the PRIME/TTRL Apache-2.0 notice in its source. Other
ported files follow their source repository's license. No CoVo source files were
copied; its formulas and the documented released reward conventions were
independently implemented after consulting the paper and public repository.

References:

- BifocalRL: https://github.com/jnanliu/BifocalRL
- slime checkout base: https://github.com/jnanliu/slime/tree/9f40ac0c59e14acdee336f31ab0fe07d496d000e
- OpenRLHF checkout base: https://github.com/jnanliu/OpenRLHF/tree/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53
- ETTRL: https://arxiv.org/abs/2508.11356 (v2)
- CoVo: https://arxiv.org/abs/2506.08745 (v1), https://github.com/sastpg/CoVo
- PRISM: https://arxiv.org/abs/2601.04700 (v1)
- GenPRM interface: https://github.com/RyanLiu112/GenPRM
