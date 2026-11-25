export interface User {
  id: number;
  email: string;
  full_name: string;
  role: 'student' | 'educator' | 'admin';
  student_id?: number | null;
}

export interface StudentProfile {
  student_id: number;
  modules: string[];
  gender?: string;
  region?: string;
  highest_education?: string;
  imd_band?: string;
  age_band?: string;
  num_of_prev_attempts: number;
  studied_credits: number;
  disability: string;
  final_result?: string;
}

export interface AssessmentRecord {
  assessment_id: number;
  module: string;
  type: string;
  score: number;
  weight: number;
  date_submitted: number;
  delay_days: number;
  is_late: boolean;
}

export interface PerformanceMetrics {
  student_id: number;
  assessments_completed: number;
  mean_score: number;
  min_score: number;
  max_score: number;
  late_submissions: number;
  mean_delay_days: number;
  assessment_history: AssessmentRecord[];
}

export interface ShapFactor {
  feature: string;
  display_name: string;
  feature_value: number;
  shap_value: number;
  impact: string;
  magnitude: number;
}

export interface AcademicRisk {
  student_id: number;
  model_name: string;
  risk_probability: number;
  risk_tier: 'Low' | 'Elevated' | 'Critical';
  confidence: number;
  top_contributing_factors: ShapFactor[];
  limitations: string[];
  assessed_at: string;
}

export interface ResourceRecommendation {
  resource_id: number;
  module: string;
  activity_type: string;
  affinity_score: number;
  graph_affinity: number;
  popularity_score: number;
  has_interacted: boolean;
  rationale: string;
}

export interface StudentRecommendations {
  student_id: number;
  total_recommendations: number;
  recommendations: ResourceRecommendation[];
}

export interface WeeklyEngagement {
  week: number;
  clicks: number;
}

export interface EngagementAnalytics {
  student_id: number;
  total_clicks: number;
  unique_resources_accessed: number;
  active_days: number;
  clicks_per_active_day: number;
  clicks_by_activity_type: Record<string, number>;
  weekly_trend: WeeklyEngagement[];
}

export interface ModuleSummary {
  module: string;
  students: number;
  at_risk_rate: number;
  avg_clicks: number;
  avg_score: number;
}

export interface CohortAnalytics {
  total_students: number;
  at_risk_count: number;
  at_risk_percentage: number;
  risk_distribution: {
    low: number;
    elevated: number;
    critical: number;
  };
  average_engagement_clicks: number;
  average_assessment_score: number;
  module_breakdown: ModuleSummary[];
  global_feature_importance: Array<{
    feature: string;
    importance: number;
    display_name: string;
  }>;
}

export interface ResourceAnalytics {
  total_resources: number;
  activity_type_counts: Record<string, number>;
  most_popular_resources: Array<{
    resource_id: number;
    module: string;
    activity_type: string;
    interaction_count: number;
    total_clicks: number;
    popularity_score: number;
  }>;
}

export interface ModelsStatus {
  champion_risk_model: {
    name: string;
    f1_score: number;
    type: string;
    status: string;
  };
  ml_models: Record<string, any>;
  deep_learning_models: {
    mlp: any;
    gru: any;
  };
  graph_learning: {
    graph_stats: any;
    ranking_metrics: any;
  };
  active_version: string;
}
