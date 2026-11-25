import React from 'react';
import {
  GraduationCap, Cpu, Network, ShieldCheck,
  Sparkles, ArrowRight, Activity, BarChart3, Database
} from 'lucide-react';

interface LandingPageProps {
  onGetStarted: () => void;
  onQuickLogin: (role: 'student' | 'educator' | 'admin') => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted, onQuickLogin }) => {
  return (
    <div className="min-h-screen bg-navy-950 text-slate-100 flex flex-col">
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-20 pb-24 border-b border-slate-800/80">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(20,184,166,0.15),rgba(255,255,255,0))]" />
        
        <div className="max-w-6xl mx-auto px-6 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-teal-500/10 border border-teal-500/25 text-teal-300 text-xs font-semibold uppercase tracking-wider mb-8">
            <Sparkles className="w-4 h-4 text-teal-400" />
            MintX Solutions • Educational Intelligence
          </div>

          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-white max-w-4xl mx-auto leading-tight">
            Adaptive Learning & <span className="bg-gradient-to-r from-teal-400 via-cyan-400 to-teal-200 bg-clip-text text-transparent">Predictive Intelligence</span> Platform
          </h1>

          <p className="mt-6 text-base md:text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed">
            Production-grade machine learning and graph neural architecture to diagnose academic risk, model sequential learner behavior, and deliver personalized study recommendations.
          </p>

          <div className="mt-10 flex flex-wrap justify-center gap-4">
            <button
              onClick={onGetStarted}
              className="px-6 py-3 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600 text-white font-semibold rounded-xl shadow-lg shadow-teal-500/25 flex items-center gap-2 transition-all"
            >
              Launch Live Workspace
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* Quick Demo Launchers */}
          <div className="mt-12 pt-8 border-t border-slate-800/60 max-w-3xl mx-auto">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
              Explore One-Click Demo Personas
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <button
                onClick={() => onQuickLogin('student')}
                className="p-3.5 rounded-xl bg-navy-900/80 hover:bg-navy-850 border border-teal-500/30 text-left transition-all group"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-teal-300">Student View</span>
                  <span className="text-[10px] text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded">ID: 100001</span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">SHAP risk factors & graph recommendations</p>
              </button>

              <button
                onClick={() => onQuickLogin('educator')}
                className="p-3.5 rounded-xl bg-navy-900/80 hover:bg-navy-850 border border-amber-500/30 text-left transition-all group"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-amber-300">Educator View</span>
                  <span className="text-[10px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded">Dr. Sarah Adams</span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Cohort risk matrix & module performance</p>
              </button>

              <button
                onClick={() => onQuickLogin('admin')}
                className="p-3.5 rounded-xl bg-navy-900/80 hover:bg-navy-850 border border-purple-500/30 text-left transition-all group"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-purple-300">ML Administrator</span>
                  <span className="text-[10px] text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded">System Health</span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Model comparison, PyTorch & Graph stats</p>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture Highlights */}
      <section className="py-20 max-w-6xl mx-auto px-6">
        <div className="text-center mb-14">
          <h2 className="text-2xl md:text-3xl font-bold text-white tracking-tight">
            Integrated Machine Learning Architecture
          </h2>
          <p className="text-sm text-slate-400 mt-2">
            Built from scratch on the Open University Learning Analytics benchmark dataset
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="p-6 rounded-2xl bg-navy-900/70 border border-slate-800">
            <Cpu className="w-8 h-8 text-teal-400 mb-4" />
            <h3 className="font-semibold text-base text-white">Ensemble Outcome ML</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Supervised comparison across XGBoost, Random Forest, and Gradient Boosting with early cutoff leakage prevention.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-navy-900/70 border border-slate-800">
            <Activity className="w-8 h-8 text-cyan-400 mb-4" />
            <h3 className="font-semibold text-base text-white">PyTorch Deep Learning</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Sequential Behavior GRU and Tabular MLP capturing 12-week click traces, temporal dynamics, and risk transitions.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-navy-900/70 border border-slate-800">
            <Network className="w-8 h-8 text-teal-300 mb-4" />
            <h3 className="font-semibold text-base text-white">Bipartite Graph LightGCN</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              4,000+ node bipartite graph learning student-resource affinities evaluated on Precision@K and NDCG@K.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-navy-900/70 border border-slate-800">
            <ShieldCheck className="w-8 h-8 text-amber-400 mb-4" />
            <h3 className="font-semibold text-base text-white">Explainable AI (SHAP)</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              TreeExplainer attributions offering transparent risk and protective factor rationales for student empowerment.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
