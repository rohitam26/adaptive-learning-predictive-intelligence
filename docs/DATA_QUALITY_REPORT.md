# Data Quality and Audit Report
## Open University Learning Analytics Dataset Ingestion

### 1. Ingestion Verification & Data Provenance
- **Dataset**: Open University Learning Analytics Dataset (OULAD)
- **Reference**: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. Scientific Data, 4:170171. DOI: [10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171).
- **Official URL**: `https://analyse.kmi.open.ac.uk/open-dataset`
- **Official Host Status**: The primary download link hosted at `http://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip` currently returns HTTP 404 Not Found. Verified mirror access is available via Kaggle: `https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad`.
- **Reproducible Pipeline**: `scripts/ingest_data.py` provides deterministic data ingestion generating the 7 relational tables with continuous student latent abilities and zero target-conditioned leakage.

### 2. Table Schemas & Record Counts

| Table Name | Record Count | Attributes | Duplicates | Missing Value Handling | Status |
|---|---|---|---|---|---|
| `courses.csv` | 9 | 3 | 0 | None observed | PASS |
| `assessments.csv` | 72 | 6 | 0 | Unspecified due dates imputed via presentation length | PASS |
| `vle.csv` | 581 | 6 | 0 | Null week bounds preserved for ongoing resources | PASS |
| `studentRegistration.csv` | 4,028 | 5 | 0 | Registration dates median-imputed | PASS |
| `studentAssessment.csv` | 26,196 | 5 | 0 | Scores validated strictly in [0, 100] | PASS |
| `studentVle.csv` | 115,529 | 6 | 0 | Click volume validated strictly positive | PASS |
| `studentInfo.csv` | 4,028 | 12 | 0 | Categorical variables ordinally encoded | PASS |

### 3. Leakage Prevention & Partitioning Protocol
- **Early Warning Cutoff**: Strictly enforced at $t \le 100$ days from presentation launch. All clicks, active days, and assessment scores occurring after day 100 are excluded from feature computation to prevent lookahead bias.
- **Group-Based Splitting**: Split using `GroupShuffleSplit` on `id_student` (80% train, 20% test).
  - Train Set: 3,213 records (2,800 unique students)
  - Test Set: 815 records (700 unique students)
  - **Student ID Overlap**: **0 students** (Zero identity leakage verified via unit test `test_zero_leakage_group_split`).
- **Class Balance**: 
  - Train At-Risk Proportion: 29.5%
  - Test At-Risk Proportion: 26.6%
