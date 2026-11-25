import React, { useEffect, useState } from 'react';
import { ApiClient } from '../api/client';
import { AcademicRisk } from '../types';
import {
  AlertTriangle, CheckCircle, ShieldCheck, HelpCircle,
  TrendingDown, TrendingUp, Cpu, Info
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface RiskDashboardViewProps {
  studentId: number;
}

export const RiskDashboardView: React.FC<RiskDashboardViewProps> = ({ studentId }) => {
  const [risk, setRisk] = useState<AcademicRisk | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await ApiClient.getStudentRisk(studentId);
        setRisk(data);
      } catch (err) {
        console.error('Failed to load risk dashboard:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [studentId]);

  if (loading) {
    return <div className="text-center py-12 text-slate-400 text-xs">Computing SHAP attributions...</div>;
  }

  const chartData = (risk?.top_contributing_factors || []).map((f) => ({
    name: f.display_name,
    shap_value: f.shap_value,
    impact: f.impact,
    feature_value: f.feature_value
  }));

  const isCritical = risk?.risk_tier === 'Critical';
  const isElevated = risk?.risk_tier === 'Elevated';

  return (
    <div className="space-y-6">
      {/* Risk Header Status */}
      <div className={`p-6 rounded-2xl border ${
        isCritical ? 'bg-rose-950/30 border-rose-500/40 text-rose-200' :
        isElevated ? 'bg-amber-950/30 border-amber-500/40 text-amber-200' :
        'bg-teal-950/30 border-teal-500/40 text-teal-200'
      }`}>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-black/40">
                Early Warning Indicator
              </span>
              <span className="text-xs text-slate-400">Assessed: {new Date(risk?.assessed_at || '').toLocaleDateString()}</span>
            </div>
            <h2 className="text-2xl font-bold text-white tracking-tight">
              Academic Risk Tier: {risk?.risk_tier || 'N/A'} ({((risk?.risk_probability ?? 0) * 100).toFixed(1)}%)
            </h2>
            <p className="text-xs text-slate-300 mt-1">
              Evaluated using <span className="font-semibold text-white">{risk?.model_name || 'Champion Model'}</span> with SHAP TreeExplainer attributions.
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div className="px-4 py-2 rounded-xl bg-black/30 border border-white/10 text-center">
              <p className="text-[10px] text-slate-400 uppercase font-bold">Prediction Confidence</p>
              <p className="text-xl font-bold font-mono text-white">
                {((risk?.confidence ?? 0) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* SHAP Waterfall / Bar Explanation */}
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="font-semibold text-sm text-white">
              SHAP Factor Attribution (Local Feature Impact)
            </h3>
            <p className="text-xs text-slate-400">
              Positive values (red) push toward Academic Risk; negative values (teal) represent protective learning behaviors.
            </p>
          </div>
        </div>

        <div className="h-72 w-full mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <XAxis type="number" stroke="#64748b" fontSize={11} />
              <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={11} width={130} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-navy-950 border border-slate-700 p-2.5 rounded-lg text-xs shadow-xl">
                        <p className="font-semibold text-white">{data.name}</p>
                        <p className="text-slate-400 mt-0.5">Value: <span className="text-white font-mono">{data.feature_value}</span></p>
                        <p className={`mt-1 font-semibold ${data.shap_value > 0 ? 'text-rose-400' : 'text-teal-400'}`}>
                          Impact: {data.shap_value > 0 ? `+${data.shap_value.toFixed(4)} (Increases Risk)` : `${data.shap_value.toFixed(4)} (Protective)`}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.shap_value > 0 ? '#f43f5e' : '#14b8a6'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Factor Breakdown Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {risk?.top_contributing_factors.map((f, i) => (
          <div
            key={i}
            className="p-4 rounded-xl bg-navy-900 border border-slate-800 flex items-start gap-3"
          >
            <div className={`p-2 rounded-lg ${
              f.impact === 'Increases Risk' ? 'bg-rose-500/15 text-rose-400' : 'bg-teal-500/15 text-teal-400'
            }`}>
              {f.impact === 'Increases Risk' ? (
                <TrendingUp className="w-5 h-5" />
              ) : (
                <TrendingDown className="w-5 h-5" />
              )}
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-xs text-white">{f.display_name}</span>
                <span className="text-xs font-mono text-slate-400">Recorded: {f.feature_value}</span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {f.impact === 'Increases Risk'
                  ? `Identified as a critical vulnerability. Statistical contribution of +${f.shap_value.toFixed(4)} toward risk.`
                  : `Serves as an academic buffer. Protective attribution of ${f.shap_value.toFixed(4)} stabilizing outcome.`}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Pedagogical Limitations Note */}
      <div className="p-5 rounded-xl bg-navy-900/60 border border-slate-800 text-xs">
        <div className="flex items-center gap-2 text-slate-300 font-semibold mb-2">
          <Info className="w-4 h-4 text-teal-400" />
          Model Governance & Pedagogical Limitations
        </div>
        <ul className="list-disc list-inside space-y-1 text-slate-400 text-[11px] leading-relaxed">
          {risk?.limitations.map((lim, i) => (
            <li key={i}>{lim}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};
