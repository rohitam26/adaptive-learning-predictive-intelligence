import React, { useEffect, useState } from 'react';
import { ApiClient } from '../api/client';
import { StudentProfile, AcademicRisk, PerformanceMetrics, EngagementAnalytics, StudentRecommendations } from '../types';
import {
  AlertTriangle, CheckCircle, TrendingUp, Sparkles,
  BookOpen, ArrowRight, Activity, ShieldCheck
} from 'lucide-react';

interface StudentDashboardProps {
  studentId: number;
  onNavigate: (view: string) => void;
}

export const StudentDashboard: React.FC<StudentDashboardProps> = ({ studentId, onNavigate }) => {
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [risk, setRisk] = useState<AcademicRisk | null>(null);
  const [perf, setPerf] = useState<PerformanceMetrics | null>(null);
  const [eng, setEng] = useState<EngagementAnalytics | null>(null);
  const [recs, setRecs] = useState<StudentRecommendations | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const [profRes, riskRes, perfRes, engRes, recRes] = await Promise.allSettled([
          ApiClient.getStudentProfile(studentId),
          ApiClient.getStudentRisk(studentId),
          ApiClient.getStudentPerformance(studentId),
          ApiClient.getStudentEngagement(studentId),
          ApiClient.getStudentRecommendations(studentId)
        ]);
        if (profRes.status === 'fulfilled') setProfile(profRes.value);
        if (riskRes.status === 'fulfilled') setRisk(riskRes.value);
        if (perfRes.status === 'fulfilled') setPerf(perfRes.value);
        if (engRes.status === 'fulfilled') setEng(engRes.value);
        if (recRes.status === 'fulfilled') setRecs(recRes.value);
      } catch (err) {
        console.error('Failed to load student dashboard:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [studentId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400 text-xs">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-400 mr-3" />
        Synthesizing Learner Predictive Intelligence...
      </div>
    );
  }

  const isRiskHigh = risk?.risk_tier === 'Critical' || risk?.risk_tier === 'Elevated';

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-navy-900 to-navy-850 border border-slate-800 shadow-xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-teal-500/10 text-teal-300 border border-teal-500/30">
                Student ID #{studentId}
              </span>
              <span className="text-xs text-slate-400">
                Enrolled Modules: {profile?.modules?.join(', ') || 'N/A'}
              </span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Adaptive Learner Intelligence Overview
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Predictive risk trajectory evaluated up to current milestone with SHAP explainability.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className={`px-4 py-2.5 rounded-xl border flex items-center gap-3 ${
              isRiskHigh
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
                : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
            }`}>
              {isRiskHigh ? (
                <AlertTriangle className="w-5 h-5 text-rose-400" />
              ) : (
                <CheckCircle className="w-5 h-5 text-emerald-400" />
              )}
              <div>
                <p className="text-[10px] uppercase font-bold tracking-wider">Academic Risk Status</p>
                <p className="text-sm font-bold">
                  {risk?.risk_tier} Risk ({((risk?.risk_probability || 0) * 100).toFixed(1)}%)
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 4 Core Stat Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl bg-navy-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Assessment Score Mean</span>
            <TrendingUp className="w-4 h-4 text-teal-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {(perf?.mean_score ?? 0).toFixed(1)}%
          </p>
          <p className="text-[11px] text-slate-500 mt-1">
            Across {perf?.assessments_completed ?? 0} completed assessments
          </p>
        </div>

        <div className="p-5 rounded-xl bg-navy-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>VLE Click Telemetry</span>
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {eng?.total_clicks?.toLocaleString() ?? 0}
          </p>
          <p className="text-[11px] text-slate-500 mt-1">
            {eng?.clicks_per_active_day ?? 0} clicks / active day ({eng?.active_days ?? 0} days)
          </p>
        </div>

        <div className="p-5 rounded-xl bg-navy-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Unique Resources</span>
            <BookOpen className="w-4 h-4 text-teal-300" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {eng?.unique_resources_accessed ?? 0}
          </p>
          <p className="text-[11px] text-slate-500 mt-1">
            Distinct course materials accessed
          </p>
        </div>

        <div className="p-5 rounded-xl bg-navy-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Model Confidence</span>
            <ShieldCheck className="w-4 h-4 text-purple-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {((risk?.confidence || 0) * 100).toFixed(1)}%
          </p>
          <p className="text-[11px] text-slate-500 mt-1">
            Champion: {risk?.model_name || 'Gradient Boosting'}
          </p>
        </div>
      </div>

      {/* Risk Alert & Explainer Snippet */}
      {risk && risk.top_contributing_factors.length > 0 && (
        <div className="p-5 rounded-xl bg-navy-900/70 border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-teal-400" />
              <h3 className="font-semibold text-sm text-white">
                Primary Predictive Factors (SHAP Local Explanations)
              </h3>
            </div>
            <button
              onClick={() => onNavigate('risk-dashboard')}
              className="text-xs text-teal-400 hover:text-teal-300 flex items-center gap-1"
            >
              Full XAI Breakdown <ArrowRight className="w-3 h-3" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {risk.top_contributing_factors.slice(0, 3).map((factor, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg bg-navy-950/70 border border-slate-800/80 text-xs"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-slate-200">{factor.display_name}</span>
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                    factor.impact === 'Increases Risk'
                      ? 'bg-rose-500/10 text-rose-300'
                      : 'bg-emerald-500/10 text-emerald-300'
                  }`}>
                    {factor.impact}
                  </span>
                </div>
                <div className="mt-2 text-slate-400 text-[11px] flex justify-between">
                  <span>Current Metric:</span>
                  <span className="text-white font-mono">{factor.feature_value}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations Carousel */}
      <div className="p-5 rounded-xl bg-navy-900/70 border border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-semibold text-sm text-white">
              Targeted Learning Recommendations
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Ranked by student-resource bipartite graph affinity & personalized risk context
            </p>
          </div>
          <button
            onClick={() => onNavigate('recommendations')}
            className="text-xs text-teal-400 hover:text-teal-300 flex items-center gap-1"
          >
            Explore All Resources <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {recs?.recommendations.slice(0, 3).map((rec) => (
            <div
              key={rec.resource_id}
              className="p-4 rounded-xl bg-navy-950 border border-slate-800 flex flex-col justify-between hover:border-teal-500/40 transition-all"
            >
              <div>
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="px-2 py-0.5 rounded bg-teal-500/10 text-teal-300 font-mono text-[10px] uppercase">
                    {rec.activity_type}
                  </span>
                  <span className="text-slate-500 font-mono text-[10px]">
                    Site #{rec.resource_id}
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed mt-2">
                  {rec.rationale}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Graph Match Affinity:</span>
                <span className="text-teal-300 font-bold font-mono">
                  {(rec.affinity_score * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
