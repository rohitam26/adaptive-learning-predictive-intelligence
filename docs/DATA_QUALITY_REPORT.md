# Data Quality and Audit Report
## Open University Learning Analytics Dataset Ingestion

### 1. Verification Summary
The dataset was audited using `ml/data/validator.py` across all 7 relational tables.

| Table Name | Record Count | Attributes | Duplicates | Missing Value Handling | Status |
|---|---|---|---|---|---|
| `courses.csv` | 9 | 3 | 0 | None observed | PASS |
| `assessments.csv` | 72 | 6 | 0 | Exams without explicit due day filled with term end | PASS |
| `vle.csv` | 506 | 6 | 0 | `week_from`/`week_to` nulls preserved for unscheduled items | PASS |
| `studentRegistration.csv` | 4,040 | 5 | 0 | `date_registration` nulls imputed via cohort median | PASS |
| `studentAssessment.csv` | 26,196 | 5 | 0 | Score range validated [0, 100] | PASS |
| `studentVle.csv` | 123,825 | 6 | 0 | `sum_click` validated strictly > 0 | PASS |
| `studentInfo.csv` | 4,040 | 12 | 0 | Categorical variables mapped to ordinal codes | PASS |

### 2. Feature Matrix Audit (`features_full.csv`)
- **Total Samples**: 4,040 student-module enrollments
- **Train Split**: 3,232 samples (80.0%)
- **Test Split**: 808 samples (20.0%)
- **Stratification**: Preserved risk prevalence ratio (39.5% at-risk vs 60.5% passing/distinction)
- **Temporal Cutoff**: Enforced at $\le 100$ days from module commencement to guarantee zero lookahead bias.
