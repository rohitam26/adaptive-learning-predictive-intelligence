import React, { useEffect, useState } from 'react';
import { ApiClient } from '../api/client';
import { StudentRecommendations } from '../types';
import { Sparkles, Network, BookOpen, Check, ExternalLink, HelpCircle } from 'lucide-react';

interface RecommendationsViewProps {
  studentId: number;
}

export const RecommendationsView: React.FC<RecommendationsViewProps> = ({ studentId }) => {
  const [recs, setRecs] = useState<StudentRecommendations | null>(null);
  const [loading, setLoading] = useState(true);
  const [completedSites, setCompletedSites] = useState<Set<number>>(new Set());

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await ApiClient.getStudentRecommendations(studentId);
        setRecs(data);
      } catch (err) {
        console.error('Failed to load recommendations:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [studentId]);

  if (loading) {
    return <div className="text-center py-12 text-slate-400 text-xs">Generating graph recommendations...</div>;
  }

  const toggleComplete = (id: number) => {
    setCompletedSites(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
        <div className="flex items-center gap-2 mb-1">
          <Sparkles className="w-4 h-4 text-teal-400" />
          <span className="text-xs uppercase font-bold text-teal-400 tracking-wider">
            Hybrid Personalization Engine
          </span>
        </div>
        <h2 className="text-xl font-bold text-white tracking-tight">
          Curated Learning Pathway & Resource Rationale
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl">
          Recommendations synthesize 2-layer LightGCN bipartite graph structural representations, collaborative peer trajectories, and academic risk context to personalize study priority.
        </p>
      </div>

      {/* Recommendations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {recs?.recommendations.map((item, idx) => {
          const isDone = completedSites.has(item.resource_id);
          return (
            <div
              key={item.resource_id}
              className={`p-5 rounded-xl border transition-all ${
                isDone
                  ? 'bg-navy-950/40 border-slate-800 opacity-60'
                  : 'bg-navy-900/90 border-slate-800 hover:border-teal-500/40'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-lg bg-teal-500/20 text-teal-300 font-bold text-xs flex items-center justify-center">
                    {idx + 1}
                  </span>
                  <div>
                    <span className="font-mono text-[10px] uppercase px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-semibold">
                      {item.activity_type}
                    </span>
                    <span className="text-[11px] text-slate-400 ml-2">Module: {item.module}</span>
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-[10px] text-slate-400 uppercase font-semibold">Affinity Score</span>
                  <p className="font-bold text-sm font-mono text-teal-400">
                    {(item.affinity_score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <p className="text-xs text-slate-300 mt-3 leading-relaxed">
                {item.rationale}
              </p>

              <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs">
                <div className="flex items-center gap-3 text-[11px] text-slate-400">
                  <span>Graph Link: <strong className="text-white font-mono">{(item.graph_affinity * 100).toFixed(0)}%</strong></span>
                  <span>Popularity: <strong className="text-white font-mono">{(item.popularity_score * 100).toFixed(0)}%</strong></span>
                </div>

                <button
                  onClick={() => toggleComplete(item.resource_id)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all ${
                    isDone
                      ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
                      : 'bg-teal-500/20 text-teal-300 hover:bg-teal-500/30 border border-teal-500/40'
                  }`}
                >
                  <Check className="w-3.5 h-3.5" />
                  {isDone ? 'Reviewed' : 'Mark Reviewed'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
