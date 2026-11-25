import React, { useEffect, useState } from 'react';
import { ApiClient } from '../api/client';
import { EngagementAnalytics } from '../types';
import { Activity, MousePointer, Calendar, Layers } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, BarChart, Bar } from 'recharts';

interface EngagementViewProps {
  studentId: number;
}

export const EngagementView: React.FC<EngagementViewProps> = ({ studentId }) => {
  const [eng, setEng] = useState<EngagementAnalytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await ApiClient.getStudentEngagement(studentId);
        setEng(data);
      } catch (err) {
        console.error('Failed to load engagement telemetry:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [studentId]);

  if (loading) {
    return <div className="text-center py-12 text-slate-400 text-xs">Aggregating VLE telemetry logs...</div>;
  }

  const typeData = Object.entries(eng?.clicks_by_activity_type || {}).map(([type, clicks]) => ({
    type,
    clicks
  })).sort((a, b) => b.clicks - a.clicks);

  return (
    <div className="space-y-6">
      {/* 4 Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Total Interaction Clicks</span>
            <MousePointer className="w-4 h-4 text-teal-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">{eng?.total_clicks.toLocaleString()}</p>
          <p className="text-[11px] text-slate-500 mt-1">Virtual Learning Environment (VLE)</p>
        </div>

        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Active Study Days</span>
            <Calendar className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">{eng?.active_days} days</p>
          <p className="text-[11px] text-slate-500 mt-1">Days with registered activity</p>
        </div>

        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Daily Velocity</span>
            <Activity className="w-4 h-4 text-teal-300" />
          </div>
          <p className="text-2xl font-bold text-teal-400 mt-2">{eng?.clicks_per_active_day}</p>
          <p className="text-[11px] text-slate-500 mt-1">Clicks per active session</p>
        </div>

        <div className="p-4 rounded-xl bg-navy-900 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Materials Covered</span>
            <Layers className="w-4 h-4 text-purple-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">{eng?.unique_resources_accessed}</p>
          <p className="text-[11px] text-slate-500 mt-1">Distinct course assets</p>
        </div>
      </div>

      {/* 12-Week Temporal Velocity Area Chart */}
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
        <h3 className="font-semibold text-sm text-white mb-1">
          12-Week Learner Engagement Velocity Curve
        </h3>
        <p className="text-xs text-slate-400 mb-6">
          Weekly click distribution modeled sequentially by PyTorch Behavior GRU
        </p>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={eng?.weekly_trend || []}>
              <defs>
                <linearGradient id="clickGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.35}/>
                  <stop offset="95%" stopColor="#14b8a6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="week" stroke="#64748b" fontSize={11} tickFormatter={(w) => `Week ${w}`} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '0.75rem',
                  fontSize: '12px'
                }}
              />
              <Area
                type="monotone"
                dataKey="clicks"
                stroke="#14b8a6"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#clickGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Clicks by Activity Type */}
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
        <h3 className="font-semibold text-sm text-white mb-1">
          Engagement by Curriculum Resource Category
        </h3>
        <p className="text-xs text-slate-400 mb-6">
          Relative concentration across quizzes, collaborative forums, and course text
        </p>

        <div className="h-60 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={typeData} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="type" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '0.75rem',
                  fontSize: '12px'
                }}
              />
              <Bar dataKey="clicks" fill="#0d9488" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
