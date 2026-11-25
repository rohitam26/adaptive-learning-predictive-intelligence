"""
Dataset Ingestion and Benchmark Generator for Learning Analytics.
Produces or verifies the 7 relational tables matching the OULAD specification:
- courses.csv
- assessments.csv
- vle.csv
- studentInfo.csv
- studentRegistration.csv
- studentAssessment.csv
- studentVle.csv
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_benchmark_dataset(seed: int = 42, num_students: int = 3500):
    np.random.seed(seed)
    random.seed(seed)

    # 1. courses.csv
    courses_data = [
        {"code_module": "AAA", "code_presentation": "2026J", "module_presentation_length": 268},
        {"code_module": "AAA", "code_presentation": "2026B", "module_presentation_length": 240},
        {"code_module": "BBB", "code_presentation": "2026J", "module_presentation_length": 268},
        {"code_module": "BBB", "code_presentation": "2026B", "module_presentation_length": 240},
        {"code_module": "CCC", "code_presentation": "2026J", "module_presentation_length": 269},
        {"code_module": "DDD", "code_presentation": "2026J", "module_presentation_length": 262},
        {"code_module": "EEE", "code_presentation": "2026J", "module_presentation_length": 268},
        {"code_module": "FFF", "code_presentation": "2026J", "module_presentation_length": 268},
        {"code_module": "GGG", "code_presentation": "2026J", "module_presentation_length": 261},
    ]
    df_courses = pd.DataFrame(courses_data)
    df_courses.to_csv(DATA_DIR / "courses.csv", index=False)
    print(f"Created courses.csv with {len(df_courses)} records.")

    # 2. assessments.csv
    assessments_list = []
    assessment_id = 10001
    for c in courses_data:
        mod = c["code_module"]
        pres = c["code_presentation"]
        # Typically 5 Tutor Marked Assessments (TMA), 4 Computer Marked Assessments (CMA), 1 Exam
        num_tma = 4
        num_cma = 3
        weights_tma = [15, 20, 20, 25]
        days_tma = [40, 95, 150, 210]
        for i in range(num_tma):
            assessments_list.append({
                "code_module": mod,
                "code_presentation": pres,
                "id_assessment": assessment_id,
                "assessment_type": "TMA",
                "date": days_tma[i],
                "weight": weights_tma[i]
            })
            assessment_id += 1
        for i in range(num_cma):
            assessments_list.append({
                "code_module": mod,
                "code_presentation": pres,
                "id_assessment": assessment_id,
                "assessment_type": "CMA",
                "date": 30 + i * 50,
                "weight": 5
            })
            assessment_id += 1
        # Final Exam
        assessments_list.append({
            "code_module": mod,
            "code_presentation": pres,
            "id_assessment": assessment_id,
            "assessment_type": "Exam",
            "date": c["module_presentation_length"] - 10,
            "weight": 100
        })
        assessment_id += 1

    df_assessments = pd.DataFrame(assessments_list)
    df_assessments.to_csv(DATA_DIR / "assessments.csv", index=False)
    print(f"Created assessments.csv with {len(df_assessments)} records.")

    # 3. vle.csv (Virtual Learning Environment resources)
    activity_types = [
        "resource", "oucontent", "url", "forumng", "quiz", 
        "subpage", "glossary", "page", "collaborate", "externalquiz"
    ]
    vle_list = []
    site_id = 200000
    for c in courses_data:
        mod = c["code_module"]
        pres = c["code_presentation"]
        # Generate 40-70 resources per course
        n_res = random.randint(45, 75)
        for _ in range(n_res):
            act = random.choice(activity_types)
            week_from = random.choice([None, random.randint(1, 20)])
            week_to = (week_from + random.randint(1, 6)) if week_from else None
            vle_list.append({
                "id_site": site_id,
                "code_module": mod,
                "code_presentation": pres,
                "activity_type": act,
                "week_from": week_from,
                "week_to": week_to
            })
            site_id += 1

    df_vle = pd.DataFrame(vle_list)
    df_vle.to_csv(DATA_DIR / "vle.csv", index=False)
    print(f"Created vle.csv with {len(df_vle)} resources.")

    # 4. studentInfo.csv & studentRegistration.csv
    regions = [
        "East Anglian Region", "Scotland", "North Western Region", "South East Region",
        "Southern Region", "West Midlands Region", "Wales", "Yorkshire Region",
        "North Region", "East Midlands Region", "Ireland", "London Region", "South West Region"
    ]
    educations = [
        "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
        "HE Qualification", "Post Graduate Qualification"
    ]
    imd_bands = [
        "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
    ]
    age_bands = ["0-35", "35-55", "55<="]
    
    student_info_list = []
    student_reg_list = []
    
    student_ids = list(range(100001, 100001 + num_students))
    
    for s_id in student_ids:
        # Assign to 1 or 2 courses
        n_enrollments = 1 if random.random() < 0.85 else 2
        assigned_courses = random.sample(courses_data, n_enrollments)
        
        gender = "M" if random.random() < 0.52 else "F"
        region = random.choice(regions)
        edu = np.random.choice(educations, p=[0.05, 0.32, 0.44, 0.17, 0.02])
        imd = random.choice(imd_bands)
        age = np.random.choice(age_bands, p=[0.70, 0.27, 0.03])
        prev_attempts = int(np.random.choice([0, 1, 2, 3], p=[0.82, 0.13, 0.04, 0.01]))
        studied_credits = int(np.random.choice([30, 60, 90, 120, 150, 180], p=[0.10, 0.65, 0.10, 0.10, 0.03, 0.02]))
        disability = "Y" if random.random() < 0.09 else "N"
        
        # Latent academic ability and engagement factor
        ability = np.random.normal(loc=0.0, scale=1.0)
        
        for c in assigned_courses:
            mod = c["code_module"]
            pres = c["code_presentation"]
            
            # Registration date typically between -150 and -10 days before course
            reg_date = int(np.random.normal(loc=-40, scale=25))
            
            # Outcome correlates with ability, prev_attempts, edu
            latent_score = (
                ability * 1.4 
                + (0.4 if edu in ["HE Qualification", "Post Graduate Qualification"] else 0.0)
                - (0.5 * prev_attempts)
                - (0.3 if disability == "Y" else 0.0)
                + np.random.normal(0, 0.5)
            )
            
            if latent_score < -0.9:
                final_result = "Withdrawn"
                unreg_date = random.randint(15, 180)
            elif latent_score < -0.15:
                final_result = "Fail"
                unreg_date = None
            elif latent_score < 1.35:
                final_result = "Pass"
                unreg_date = None
            else:
                final_result = "Distinction"
                unreg_date = None
                
            student_info_list.append({
                "code_module": mod,
                "code_presentation": pres,
                "id_student": s_id,
                "gender": gender,
                "region": region,
                "highest_education": edu,
                "imd_band": imd,
                "age_band": age,
                "num_of_prev_attempts": prev_attempts,
                "studied_credits": studied_credits,
                "disability": disability,
                "final_result": final_result,
                "_latent_ability": ability # temporary for simulation
            })
            
            student_reg_list.append({
                "code_module": mod,
                "code_presentation": pres,
                "id_student": s_id,
                "date_registration": reg_date,
                "date_unregistration": unreg_date
            })

    df_student_info = pd.DataFrame(student_info_list)
    df_student_reg = pd.DataFrame(student_reg_list)
    
    # Save studentRegistration
    df_student_reg.to_csv(DATA_DIR / "studentRegistration.csv", index=False)
    print(f"Created studentRegistration.csv with {len(df_student_reg)} records.")
    
    # 5. studentAssessment.csv
    student_assess_list = []
    # Build lookup of assessments by module/presentation
    assessments_by_mod = {}
    for a in assessments_list:
        key = (a["code_module"], a["code_presentation"])
        assessments_by_mod.setdefault(key, []).append(a)
        
    for s in student_info_list:
        s_id = s["id_student"]
        mod = s["code_module"]
        pres = s["code_presentation"]
        ability = s["_latent_ability"]
        result = s["final_result"]
        
        mod_assessments = assessments_by_mod.get((mod, pres), [])
        for a in mod_assessments:
            # If student withdrew early, they skip later assessments
            if result == "Withdrawn" and random.random() < 0.65:
                continue
            
            # Score depends on ability, assessment type
            base_score = 65 + ability * 15 + np.random.normal(0, 10)
            if a["assessment_type"] == "Exam":
                base_score -= 8
            if result == "Fail":
                base_score = min(base_score, 45 + np.random.normal(0, 5))
            elif result == "Distinction":
                base_score = max(base_score, 82 + np.random.normal(0, 4))
            
            score = max(0, min(100, int(round(base_score))))
            
            sub_date = a["date"] + random.randint(-5, 4) if a["date"] else 50
            is_banked = 1 if random.random() < 0.02 else 0
            
            student_assess_list.append({
                "id_assessment": a["id_assessment"],
                "id_student": s_id,
                "date_submitted": sub_date,
                "is_banked": is_banked,
                "score": score
            })

    df_student_assess = pd.DataFrame(student_assess_list)
    df_student_assess.to_csv(DATA_DIR / "studentAssessment.csv", index=False)
    print(f"Created studentAssessment.csv with {len(df_student_assess)} assessment submissions.")

    # 6. studentVle.csv (VLE interactions: clicks per resource over time)
    vle_by_mod = {}
    for v in vle_list:
        key = (v["code_module"], v["code_presentation"])
        vle_by_mod.setdefault(key, []).append(v)
        
    student_vle_list = []
    for s in student_info_list:
        s_id = s["id_student"]
        mod = s["code_module"]
        pres = s["code_presentation"]
        ability = s["_latent_ability"]
        result = s["final_result"]
        
        mod_resources = vle_by_mod.get((mod, pres), [])
        if not mod_resources:
            continue
            
        # Engagement volume scales with result
        if result == "Distinction":
            num_interactions = random.randint(40, 80)
            avg_clicks = 8
        elif result == "Pass":
            num_interactions = random.randint(25, 55)
            avg_clicks = 5
        elif result == "Fail":
            num_interactions = random.randint(10, 30)
            avg_clicks = 3
        else: # Withdrawn
            num_interactions = random.randint(3, 15)
            avg_clicks = 2
            
        chosen_resources = random.choices(mod_resources, k=num_interactions)
        for res in chosen_resources:
            # Interaction date during the term (-10 to 240)
            max_day = 80 if result == "Withdrawn" else 240
            date = random.randint(-10, max_day)
            clicks = max(1, int(np.random.poisson(lam=avg_clicks)))
            
            student_vle_list.append({
                "code_module": mod,
                "code_presentation": pres,
                "id_student": s_id,
                "id_site": res["id_site"],
                "date": date,
                "sum_click": clicks
            })

    df_student_vle = pd.DataFrame(student_vle_list)
    df_student_vle.to_csv(DATA_DIR / "studentVle.csv", index=False)
    print(f"Created studentVle.csv with {len(df_student_vle)} interaction events.")

    # Remove temporary column and save studentInfo
    df_student_info_clean = df_student_info.drop(columns=["_latent_ability"])
    df_student_info_clean.to_csv(DATA_DIR / "studentInfo.csv", index=False)
    print(f"Created studentInfo.csv with {len(df_student_info_clean)} students.")

if __name__ == "__main__":
    generate_benchmark_dataset()
