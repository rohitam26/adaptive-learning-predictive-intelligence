import React, { useEffect, useState } from 'react';
import { ApiClient } from '../api/client';
import { CohortAnalytics } from '../types';
import { Users, AlertTriangle, CheckCircle, TrendingUp, Cpu, PieChart as PieIcon } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';

export const CohortAnalyticsView: React.FC = () => {
  const [cohort, setCohort] = useState<CohortAnalytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await ApiClient.getCohortAnalytics();
        setCohort(data);
      } catch (err) {
        console.error('Failed to load cohort analytics:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return <div className="text-center py-12 text-slate-400 text-xs">Aggregating cohort-wide telemetry...</div>;
  }

  const riskDistData = [
    { name: 'Low Risk', count: cohort?.risk_distribution.low || 0, color: '#10b981' },
    { name: 'Elevated Risk', count: cohort?.risk_distribution.elevated || 0, color: '#f59e0b' },
    { name: 'Critical Risk', count: cohort?.risk_distribution.critical || 0, color: '#f43f5e' },
  ];

  const globalShapData = (cohort?.global_feature_importance || []).slice(0, 8).map(f => ({
    name: f.display_name,
    importance: f.importance
  }));

  return (
    <div className="space-y-6">
      {/* 4 Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <p className="text-xs text-slate-400">Total Learner Cohort</p>
          <p className="text-2xl font-bold text-white mt-1">
            {cohort?.total_students.toLocaleString()}
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">Enrolled across 9 course presentations</p>
        </div>

        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <p className="text-xs text-slate-400">At-Risk Prevalence</p>
          <p className="text-2xl font-bold text-rose-400 mt-1">
            {cohort?.at_risk_percentage}%
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">{cohort?.at_risk_count} total flagged students</p>
        </div>

        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <p className="text-xs text-slate-400">Cohort Score Mean</p>
          <p className="text-2xl font-bold text-teal-400 mt-1">
            {cohort?.average_assessment_score}%
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">Weighted assessment average</p>
        </div>

        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <p className="text-xs text-slate-400">Cohort Mean Clicks</p>
          <p className="text-2xl font-bold text-cyan-400 mt-1">
            {cohort?.average_engagement_clicks.toLocaleString()}
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">VLE interaction telemetry</p>
        </div>
      </div>

      {/* Cohort Risk Breakdown & Global SHAP Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Risk Distribution */}
        <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
          <h3 className="font-semibold text-sm text-white mb-1">
            Cohort Early Academic Risk Tiers
          </h3>
          <p className="text-xs text-slate-400 mb-6">
            Stratification across Low, Elevated and Critical vulnerability
          </p>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskDistData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    fontSize: '12px'
                  }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {riskDistData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Global SHAP Feature Importance */}
        <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
          <h3 className="font-semibold text-sm text-white mb-1">
            Global Feature Importance (Cohort-Wide Mean |SHAP|)
          </h3>
          <p className="text-xs text-slate-400 mb-6">
            Top variables driving predictive models across the student population
          </p>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={globalShapData}
                layout="vertical"
                margin={{ top: 0, right: 20, left: 60, bottom: 0 }}
              >
                <XAxis type="number" stroke="#64748b" fontSize={10} />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={10} width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    fontSize: '12px'
                  }}
                />
                <Bar dataKey="importance" fill="#14b8a6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Module Breakdown Table */}
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
        <h3 className="font-semibold text-sm text-white mb-4">
          Course Module Comparative Intelligence
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-medium">
                <th className="pb-3">Module Code</th>
                <th className="pb-3">Enrolled Learners</th>
                <th className="pb-3">At-Risk Rate</th>
                <th className="pb-3">Mean VLE Clicks</th>
                <th className="pb-3">Assessment Mean</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {cohort?.module_breakdown.map((m) => (
                <tr key={m.module} className="hover:bg-navy-950/40">
                  <td className="py-3 font-semibold text-teal-300 font-mono">Module {m.module}</td>
                  <td className="py-3 text-white">{m.students} students</td>
                  <td className="py-3">
                    <span className={`px-2 py-0.5 rounded font-bold text-[11px] ${
                      m.at_risk_rate > 35 ? 'bg-rose-500/10 text-rose-300' : 'bg-emerald-500/10 text-emerald-300'
                    }`}>
                      {m.at_risk_rate}%
                    </span>
                  </td>
                  <td className="py-3 text-slate-300 font-mono">{m.avg_clicks} clicks</td>
                  <td className="py-3 font-bold text-white font-mono">{m.avg_score}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
