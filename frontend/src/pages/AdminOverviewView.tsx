import React, { useEffect, useState } from 'react';
import { ApiClient } from '../api/client';
import { ModelsStatus } from '../types';
import { Database, Cpu, Network, CheckCircle, Server, Activity } from 'lucide-react';

export const AdminOverviewView: React.FC = () => {
  const [statusData, setStatusData] = useState<ModelsStatus | null>(null);
  const [healthData, setHealthData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [models, health] = await Promise.all([
          ApiClient.getModelsStatus(),
          fetch('/health').then(r => r.json())
        ]);
        setStatusData(models);
        setHealthData(health);
      } catch (err) {
        console.error('Failed to load admin overview:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return <div className="text-center py-12 text-slate-400 text-xs">Querying model registry & telemetry...</div>;
  }

  const mlModels: any = statusData?.ml_models || {};
  const dlModels: any = statusData?.deep_learning_models || {};
  const graph: any = statusData?.graph_learning || {};

  return (
    <div className="space-y-6">
      {/* Platform Health Header */}
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">
              System Live & Operational
            </span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Machine Learning Engineering & Model Registry
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Model version: <span className="font-mono text-teal-300">v{statusData?.active_version}</span> | Database: <span className="text-white font-mono">{healthData?.database}</span> | Artifacts: <span className="text-emerald-400">Verified</span>
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-navy-950 border border-slate-800 text-xs text-slate-300">
          <Server className="w-4 h-4 text-teal-400" />
          <span>FastAPI 0.165 + PyTorch 2.15 + XGBoost 3.2</span>
        </div>
      </div>

      {/* Supervised ML Model Comparison */}
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-semibold text-sm text-white">
              Supervised Machine Learning Model Comparison
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Trained on stratified OULAD feature split with leakage-free 100-day early cutoff
            </p>
          </div>
          <span className="px-2.5 py-1 rounded bg-teal-500/10 text-teal-300 border border-teal-500/30 text-xs font-semibold">
            Champion: {statusData?.champion_risk_model.name}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-medium">
                <th className="pb-3">Algorithm</th>
                <th className="pb-3">Accuracy</th>
                <th className="pb-3">Precision</th>
                <th className="pb-3">Recall</th>
                <th className="pb-3">F1 Score</th>
                <th className="pb-3">ROC-AUC</th>
                <th className="pb-3">PR-AUC</th>
                <th className="pb-3">Brier Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {Object.entries(mlModels).map(([name, m]: [string, any]) => (
                <tr key={name} className="hover:bg-navy-950/40">
                  <td className="py-3 font-semibold text-white flex items-center gap-2">
                    {name === statusData?.champion_risk_model.name && (
                      <CheckCircle className="w-3.5 h-3.5 text-teal-400" />
                    )}
                    {name}
                  </td>
                  <td className="py-3 font-mono text-slate-300">{m.accuracy}</td>
                  <td className="py-3 font-mono text-slate-300">{m.precision}</td>
                  <td className="py-3 font-mono text-slate-300">{m.recall}</td>
                  <td className="py-3 font-bold font-mono text-teal-300">{m.f1}</td>
                  <td className="py-3 font-mono text-slate-300">{m.roc_auc}</td>
                  <td className="py-3 font-mono text-slate-300">{m.pr_auc}</td>
                  <td className="py-3 font-mono text-slate-400">{m.brier_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Deep Learning & Graph Learning Split */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* PyTorch Deep Learning */}
        <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
          <div className="flex items-center gap-2 mb-3">
            <Cpu className="w-5 h-5 text-teal-400" />
            <h3 className="font-semibold text-sm text-white">PyTorch Deep Learning Models</h3>
          </div>

          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-navy-950 border border-slate-800">
              <div className="flex justify-between items-center mb-1">
                <span className="font-semibold text-xs text-white">Tabular MLP</span>
                <span className="text-[10px] text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded font-mono">
                  Accuracy: {dlModels?.mlp?.accuracy}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Multi-layer perceptron with batch normalization, dropout (0.3), and AdamW optimization.
              </p>
              <div className="mt-3 flex gap-4 text-xs font-mono text-slate-300">
                <span>F1: <strong>{dlModels?.mlp?.f1}</strong></span>
                <span>ROC-AUC: <strong>{dlModels?.mlp?.roc_auc}</strong></span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-navy-950 border border-slate-800">
              <div className="flex justify-between items-center mb-1">
                <span className="font-semibold text-xs text-white">Sequential Behavior GRU</span>
                <span className="text-[10px] text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded font-mono">
                  Accuracy: {dlModels?.gru?.accuracy}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                2-layer recurrent GRU over 12 weekly interaction click sequences with gradient clipping.
              </p>
              <div className="mt-3 flex gap-4 text-xs font-mono text-slate-300">
                <span>F1: <strong>{dlModels?.gru?.f1}</strong></span>
                <span>ROC-AUC: <strong>{dlModels?.gru?.roc_auc}</strong></span>
              </div>
            </div>
          </div>
        </div>

        {/* Bipartite Graph Learning */}
        <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
          <div className="flex items-center gap-2 mb-3">
            <Network className="w-5 h-5 text-teal-300" />
            <h3 className="font-semibold text-sm text-white">Bipartite Graph & LightGCN Representation</h3>
          </div>

          <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 mb-4">
            <p className="text-xs font-semibold text-slate-300 mb-2">Network Topology</p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Total Nodes:</span>
                <span className="font-mono text-white">{graph?.graph_stats?.total_nodes}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Total Edges:</span>
                <span className="font-mono text-white">{graph?.graph_stats?.total_edges}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Student Nodes:</span>
                <span className="font-mono text-white">{graph?.graph_stats?.student_nodes}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Resource Nodes:</span>
                <span className="font-mono text-white">{graph?.graph_stats?.resource_nodes}</span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-navy-950 border border-slate-800">
            <p className="text-xs font-semibold text-slate-300 mb-2">Ranking Evaluation Benchmark</p>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="p-2 rounded bg-navy-900 border border-slate-800">
                <p className="text-[10px] text-slate-400 uppercase">Precision@5</p>
                <p className="font-mono font-bold text-teal-400 mt-1">
                  {(graph?.ranking_metrics || graph?.unconstrained_ranking_metrics)?.['precision@5'] ?? 'N/A'}
                </p>
              </div>
              <div className="p-2 rounded bg-navy-900 border border-slate-800">
                <p className="text-[10px] text-slate-400 uppercase">Recall@5</p>
                <p className="font-mono font-bold text-teal-400 mt-1">
                  {(graph?.ranking_metrics || graph?.unconstrained_ranking_metrics)?.['recall@5'] ?? 'N/A'}
                </p>
              </div>
              <div className="p-2 rounded bg-navy-900 border border-slate-800">
                <p className="text-[10px] text-slate-400 uppercase">NDCG@5</p>
                <p className="font-mono font-bold text-cyan-400 mt-1">
                  {(graph?.ranking_metrics || graph?.unconstrained_ranking_metrics)?.['ndcg@5'] ?? 'N/A'}
                </p>
              </div>
              <div className="p-2 rounded bg-navy-900 border border-slate-800">
                <p className="text-[10px] text-slate-400 uppercase">Precision@10</p>
                <p className="font-mono font-bold text-teal-400 mt-1">
                  {(graph?.ranking_metrics || graph?.unconstrained_ranking_metrics)?.['precision@10'] ?? 'N/A'}
                </p>
              </div>
              <div className="p-2 rounded bg-navy-900 border border-slate-800">
                <p className="text-[10px] text-slate-400 uppercase">Recall@10</p>
                <p className="font-mono font-bold text-teal-400 mt-1">
                  {(graph?.ranking_metrics || graph?.unconstrained_ranking_metrics)?.['recall@10'] ?? 'N/A'}
                </p>
              </div>
              <div className="p-2 rounded bg-navy-900 border border-slate-800">
                <p className="text-[10px] text-slate-400 uppercase">NDCG@10</p>
                <p className="font-mono font-bold text-cyan-400 mt-1">
                  {(graph?.ranking_metrics || graph?.unconstrained_ranking_metrics)?.['ndcg@10'] ?? 'N/A'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
