# Data Dictionary
## Open University Learning Analytics Dataset (OULAD) Entities

### 1. Relational Tables

#### 1.1 `courses.csv`
Contains basic module presentation structures.
- `code_module` (VARCHAR): Identification code for module (e.g. AAA, BBB, CCC).
- `code_presentation` (VARCHAR): Presentation code indicating year and intake (e.g. 2026J, 2026B).
- `module_presentation_length` (INTEGER): Total module duration in calendar days.

#### 1.2 `assessments.csv`
Definitions of graded assessments.
- `id_assessment` (INTEGER PRIMARY KEY): Unique assessment identifier.
- `code_module` (VARCHAR): Associated module code.
- `code_presentation` (VARCHAR): Associated presentation code.
- `assessment_type` (VARCHAR): Format (Tutor-Marked 'TMA', Computer-Marked 'CMA', 'Exam').
- `date` (INTEGER): Target due day relative to course start.
- `weight` (FLOAT): Percentage weighting toward final course score (0-100%).

#### 1.3 `vle.csv`
Curriculum learning resources within the Virtual Learning Environment.
- `id_site` (INTEGER PRIMARY KEY): Unique site or material identifier.
- `code_module` (VARCHAR): Associated module code.
- `code_presentation` (VARCHAR): Associated presentation code.
- `activity_type` (VARCHAR): Resource type (forumng, oucontent, quiz, url, resource, subpage).
- `week_from` (INTEGER): Recommended start week.
- `week_to` (INTEGER): Recommended completion week.

#### 1.4 `studentInfo.csv`
Student demographic and final academic outcome records.
- `code_module` (VARCHAR): Module enrolled.
- `code_presentation` (VARCHAR): Presentation enrolled.
- `id_student` (INTEGER): Student identification number.
- `gender` (VARCHAR): Student gender (M/F).
- `region` (VARCHAR): Geographic home region.
- `highest_education` (VARCHAR): Prior education qualification.
- `imd_band` (VARCHAR): Index of Multiple Deprivation decile.
- `age_band` (VARCHAR): Age bracket (0-35, 35-55, 55<=).
- `num_of_prev_attempts` (INTEGER): Previous enrollments in this module.
- `studied_credits` (INTEGER): Cumulative credits currently attempted.
- `disability` (VARCHAR): Registered disability status (Y/N).
- `final_result` (VARCHAR): Final academic outcome (Distinction, Pass, Fail, Withdrawn).

#### 1.5 `studentRegistration.csv`
Student enrollment timeline events.
- `code_module` (VARCHAR): Module code.
- `code_presentation` (VARCHAR): Presentation code.
- `id_student` (INTEGER): Student identification number.
- `date_registration` (INTEGER): Days relative to module start when registration occurred.
- `date_unregistration` (INTEGER): Days relative to module start when unregistration occurred.

#### 1.6 `studentAssessment.csv`
Individual assessment submissions and earned scores.
- `id_assessment` (INTEGER): Assessment ID.
- `id_student` (INTEGER): Submitting student ID.
- `date_submitted` (INTEGER): Submission day.
- `is_banked` (INTEGER): 1 if grade carried from prior attempt, 0 otherwise.
- `score` (FLOAT): Score achieved (0-100).

#### 1.7 `studentVle.csv`
Granular click-stream interaction logs.
- `code_module` (VARCHAR): Module code.
- `code_presentation` (VARCHAR): Presentation code.
- `id_student` (INTEGER): Student ID.
- `id_site` (INTEGER): VLE site resource ID accessed.
- `date` (INTEGER): Day of interaction event.
- `sum_click` (INTEGER): Count of interaction clicks on that date.
