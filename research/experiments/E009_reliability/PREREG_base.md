# E009-base — same protocol on google/gemma-4-E2B (tests H-017; registered 2026-10-25 after seeing E009-IT, before any base data)
Identical run.py/analyze.py with VARIANT=base. Additional H-017 predictions (exploratory script `exploratory_errors.py`, parameterised):
(i) share of base errors whose final top-1 is a markdown token (' **', '**', ' *', ' #') < 2%;
(ii) TEST accuracy at residual 33 minus final accuracy ≤ 2 points (IT: +6.1);
(iii) lost-answer count (target top-1 at some residual 20–34, final wrong) ≤ 75 (IT: 151).
Also re-tests P-a1..P-c1 of PREREG.md on a second checkpoint (transfer). Known validity limit of factset_v1 (found in E009-IT):
first-token correctness mixes knowledge with answer form (first names, initials, articles); reported, not corrected, for comparability.
