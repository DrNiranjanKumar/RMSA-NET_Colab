# PSG analysis (R1.6, R2.8)

**Sign-flip check.** Negating the last layer of the polarity estimator changed the regression output by at most 0.00e+00 over 105 trained models (float rounding only). The sign of p is therefore not identifiable; only |p| is meaningful.

Token-to-word mapping uses the BERT token ids stored in the data file. On average 11.2% of PSG weight falls on [CLS]/[SEP] tokens; word weights below are renormalised over real words.

**|p| vs. VADER lexicon intensity.** Spearman ρ = 0.189 (p = 2.7e-09) over 978 lexicon tokens in 686 test utterances whose word count matches the token count. The lexicon is used only for this analysis.

**Qualitative examples (top-3 tokens by PSG weight).**

- label -3.0, prediction -1.84: “yeap a horrible protagonist” → top tokens: yeap (0.46), horrible (0.30), protagonist (0.13)

- label +3.0, prediction +2.49: “it was incredible see all the different color palettes smooth hand drawn animation you know” → top tokens: palettes (0.10), incredible (0.10), it (0.10)

- label +3.0, prediction -0.06: “first of all id like to say little wolver james or jimmy hes so cute hes so xxx” → top tokens: hes (0.10), wolver (0.09), hes (0.08)

- label +3.0, prediction -1.36: “lip smacking anyway lip smacking i love sabre tooth in this movie” → top tokens: smacking (0.15), smacking (0.14), i (0.10)

- label -2.8, prediction +1.93: “he um had all the charm of a narcissist xxx boy the whole film” → top tokens: narcissist (0.17), he (0.10), all (0.10)

- label +3.0, prediction -1.36: “lip smacking anyway lip smacking i love sabre tooth in this movie” → top tokens: smacking (0.15), smacking (0.14), i (0.10)
