# Interview notes — offline ranking eval pipeline

**Context:** Portfolio demo for Habib Noor. **SYNTHETIC_DEMO_DATA only.** Independent project; no employer affiliation.

## Module map

| Module | Responsibility | Tradeoff |
|--------|----------------|----------|
| `synthetic.py` | Labelled fake query/user/item logs | Fast to run; risk of overfitting narrative to generator |
| `schema.py` | Fail-closed validation | Strict CSV reload needs `.meta.json` sidecar |
| `split.py` | Entity-disjoint + time guard | Drops mixed-bucket rows; simpler than nested CV |
| `features.py` | Four numeric features | No text embeddings or cross-features |
| `baselines.py` | Item mean relevance | Strong enough to show lift sometimes; not a random ranker |
| `train.py` | HGBR + scaler, joblib export | Pointwise, not listwise; joblib trust boundary |
| `metrics.py` | NDCG@k by query | Needs enough items per query in eval |
| `cli.py` | Orchestration | Single process; no Airflow/Beam |
| `artifacts.py` | JSON provenance | No W&B / MLflow |

## Failure experiments (documented honestly)

1. **Removed synthetic meta** → `validate` fails (by design).
2. **Mixed entity buckets** → rows dropped; if volume too low, split raises empty-set error.
3. **Single-class eval clicks** → ROC/PR set to `nan` instead of fake 0.99 scores.
4. **Untrusted joblib** → loader checks `trusted_local_only` flag (still requires trusted path discipline).

## Split semantics (elevator pitch)

"We hash each query, user, and item into train or eval. An interaction only enters one side if all three entities agree. That prevents the classic leak where the same item embedding appears in train features and eval labels. We add a time cutoff so train is earlier within that framework. Limitation: it's not user-session grouped VideoDNA; it's a small-data teaching split."

## 30 interview Q&As

### Data & schema

1. **Q:** Why fail-closed validation?  
   **A:** Bad columns or nulls silently becoming 0.0 destroy metric trust; we stop immediately with `SchemaValidationError`.

2. **Q:** Why a synthetic banner?  
   **A:** Prevents accidentally demoing unlabelled CSVs as if they were production logs.

3. **Q:** Why relevance 0–3?  
   **A:** Graded labels support NDCG; binary clicks are derived for auxiliary AUC metrics.

4. **Q:** What is `query_term_hash`?  
   **A:** Deterministic numeric stand-in for query text features without shipping NLP scope.

5. **Q:** Why `.meta.json` beside CSV?  
   **A:** Pandas CSV loses `attrs`; sidecar preserves synthetic provenance on reload.

### Leakage & splits

6. **Q:** Query-only split vs yours?  
   **A:** Query-only still shares items/users across splits; entity-disjoint is stricter for interviews.

7. **Q:** What rows get dropped?  
   **A:** Any interaction where query/user/item buckets disagree (e.g. train query + eval item).

8. **Q:** Time split role?  
   **A:** Train uses timestamps before cutoff; eval entities use later times — mimics delayed eval without crossing entities.

9. **Q:** Can hash split be biased?  
   **A:** Yes — it's uniform by hash mod, not stratified by volume or geography.

10. **Q:** Session leakage?  
    **A:** Not addressed; sessions aren't modeled in synthetic schema.

### Modeling

11. **Q:** Why pointwise HGBR?  
    **A:** CPU-fast, sklearn-native, explainable; tradeoff is ignoring list context in the loss.

12. **Q:** Why not LambdaMART?  
    **A:** Scope control; this repo targets pipeline skeleton, not SOTA LTR library integration.

13. **Q:** Feature scaling?  
    **A:** `StandardScaler` in pipeline for fair boosting splits across feature scales.

14. **Q:** Clip predictions 0–3?  
    **A:** Keeps MAE comparable to label support; regressor can extrapolate otherwise.

15. **Q:** Cold-start items in eval?  
    **A:** Disjoint item sets mean popularity baseline may fall back to global mean — good talking point.

### Metrics

16. **Q:** Primary metric?  
    **A:** Mean NDCG@10 grouped by `query_id` on held-out eval lists.

17. **Q:** Why MAE?  
    **A:** Regressor calibration check alongside ranking metric.

18. **Q:** When is ROC-AUC valid here?  
    **A:** Only when both click and non-click exist in eval (`relevance >= 2`).

19. **Q:** Why not MRR?  
    **A:** Graded relevance → NDCG is natural; MRR is for single relevant doc.

20. **Q:** Small query lists?  
    **A:** NDCG@k uses `min(k, len(list))`; variance is high — mention confidence intervals in a real job.

### Baselines & fairness

21. **Q:** Popularity baseline definition?  
    **A:** Per-item mean training relevance; unseen items use global train mean.

22. **Q:** Do you guarantee beating baseline?  
    **A:** No. README shows one seed where model wins; other seeds may differ.

23. **Q:** Global mean baseline?  
    **A:** Weakest sensible constant scorer — shows model isn't cheating via metric code.

### Ops & safety

24. **Q:** Why document joblib risk?  
    **A:** Pickle/joblib from internet is RCE; artifacts are trusted-local-only demos.

25. **Q:** Reproducibility?  
    **A:** Fixed seed in synthetic, split hash, and HGBR `random_state`; pinned deps.

26. **Q:** CI scope?  
    **A:** pytest for schema, disjoint split, metrics edge, CLI repeatability — no GPU.

27. **Q:** Docker purpose?  
    **A:** Reproducible test image; not production serving.

28. **Q:** What would you add for TikTok LIVE-style infra?  
    **A:** Feature store contracts, near-line freshness SLAs, shadow metrics, and listwise eval at request grain — out of scope here.

### Product / ethics

29. **Q:** Real user data?  
    **A:** Never in this repo — synthetic only.

30. **Q:** How to present in interviews?  
    **A:** Walk through leakage story, metric choice, baseline honesty, and what you'd harden for production (monitoring, rollback, consent).

---

*Generated as part of the ranking-eval-pipeline portfolio repository. Metrics in README are from an actual local `run` with seed 42; re-run to refresh.*
