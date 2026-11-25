import React, { useEffect, useState } from 'react';
import { ApiClient } from '../api/client';
import { PerformanceMetrics } from '../types';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Award, Clock, CheckCircle2, AlertCircle } from 'lucide-react';

interface PerformanceViewProps {
  studentId: number;
}

export const PerformanceView: React.FC<PerformanceViewProps> = ({ studentId }) => {
  const [perf, setPerf] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await ApiClient.getStudentPerformance(studentId);
        setPerf(data);
      } catch (err) {
        console.error('Failed to load performance metrics:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [studentId]);

  if (loading) {
    return <div className="text-center py-12 text-slate-400 text-xs">Loading performance data...</div>;
  }

  const chartData = (perf?.assessment_history || []).map((a, i) => ({
    name: `Assess ${i + 1} (${a.type})`,
    score: a.score,
    weight: a.weight,
    delay: a.delay_days
  }));

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <p className="text-xs text-slate-400">Mean Score</p>
          <p className="text-2xl font-bold text-teal-400 mt-1">{perf?.mean_score}%</p>
          <p className="text-[11px] text-slate-500 mt-0.5">Cohort Benchmark: ~65%</p>
        </div>
        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <p className="text-xs text-slate-400">Score Range</p>
          <p className="text-2xl font-bold text-white mt-1">{perf?.min_score}% - {perf?.max_score}%</p>
          <p className="text-[11px] text-slate-500 mt-0.5">Min to Max achieved</p>
        </div>
        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <p className="text-xs text-slate-400">Completed Assessments</p>
          <p className="text-2xl font-bold text-white mt-1">{perf?.assessments_completed}</p>
          <p className="text-[11px] text-slate-500 mt-0.5">Formal submissions</p>
        </div>
        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <p className="text-xs text-slate-400">Late Submissions</p>
          <p className={`text-2xl font-bold mt-1 ${
            (perf?.late_submissions || 0) > 0 ? 'text-amber-400' : 'text-emerald-400'
          }`}>
            {perf?.late_submissions}
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">Mean Delay: {perf?.mean_delay_days} days</p>
        </div>
      </div>

      {/* Trajectory Chart */}
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
        <h3 className="font-semibold text-sm text-white mb-1">
          Assessment Score Progression & Trajectory
        </h3>
        <p className="text-xs text-slate-400 mb-6">
          Sequential performance across Tutor-Marked (TMA), Computer-Marked (CMA) and Exams
        </p>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" domain={[0, 100]} fontSize={11} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '0.75rem',
                  fontSize: '12px'
                }}
              />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#14b8a6"
                strokeWidth={3}
                dot={{ r: 4, fill: '#14b8a6' }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Assessment Table */}
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
        <h3 className="font-semibold text-sm text-white mb-4">
          Detailed Assessment Audit Log
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-medium">
                <th className="pb-3">Assessment ID</th>
                <th className="pb-3">Module</th>
                <th className="pb-3">Type</th>
                <th className="pb-3">Weight</th>
                <th className="pb-3">Submission Day</th>
                <th className="pb-3">Punctuality</th>
                <th className="pb-3">Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {perf?.assessment_history.map((item) => (
                <tr key={item.assessment_id} className="hover:bg-navy-950/40">
                  <td className="py-3 font-mono text-slate-300">#{item.assessment_id}</td>
                  <td className="py-3 font-semibold text-white">{item.module}</td>
                  <td className="py-3">
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px]">
                      {item.type}
                    </span>
                  </td>
                  <td className="py-3 text-slate-400">{item.weight}%</td>
                  <td className="py-3 text-slate-300">Day {item.date_submitted}</td>
                  <td className="py-3">
                    {item.is_late ? (
                      <span className="inline-flex items-center gap-1 text-amber-400 text-[10px]">
                        <Clock className="w-3 h-3" /> {item.delay_days}d Late
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-emerald-400 text-[10px]">
                        <CheckCircle2 className="w-3 h-3" /> On Time
                      </span>
                    )}
                  </td>
                  <td className="py-3 font-bold font-mono text-teal-300 text-sm">
                    {item.score}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
