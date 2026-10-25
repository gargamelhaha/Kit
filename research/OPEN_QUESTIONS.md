# Open questions
1. Do standard interpretability findings (two-site factual recall, induction heads, logit-lens dynamics) transfer to an
   architecture where 57% of layers cannot write K/V (Gemma 4 E2B)? (E002–E004)
2. What computation do the 20 KV-shared, double-wide-MLP layers perform, given they act per position? Are they mostly
   "readout" (answer formation, formatting) as the logit lens would suggest?
3. Role of per-layer embeddings: token-identity memory, detokenization shortcut, or something else? (H-006)
4. Are the community `decoderesearch/gemma-4-saes` (E2B base, L6/L17/L28) faithful (FVU, ΔCE vs random-decoder baseline)?
5. Root cause of the sandbox resets (PL-4).
6. Which architectures for data-efficient / autonomous learning have controlled evidence that can be checked at small scale
   on CPU? (LIT-05 pending)
7. ECC: the task asks to apply an "ECC principle" but no definition exists in the task or repository. Not applied; needs a
   definition from the user.
8. (cycle 2) Why does a single layer (25) of the 20 KV-shared layers, and a single head (25.0), do subject extraction — and what
   backup route keeps recall at 0.84 when its subject-specific content is removed? (K-0007, H-015)
9. (cycle 2) What happens when the subject lies outside layer 13's 512-token sliding window — rerouting through layer-14 K/V or
   failure? Do in-context associations share the parametric route? (H-011, H-013)
10. (cycle 2) Is the low-rank structure of intervention-effect matrices a usable discovery principle (H-014)?
