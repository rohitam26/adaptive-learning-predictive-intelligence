import React from 'react';
import {
  LayoutDashboard, UserCircle, LineChart, AlertTriangle,
  Sparkles, Activity, Users, Database, ShieldAlert
} from 'lucide-react';

interface SidebarProps {
  currentView: string;
  onNavigate: (view: string) => void;
  userRole?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentView, onNavigate, userRole = 'student' }) => {
  const studentNavItems = [
    { id: 'student-dashboard', label: 'Student Overview', icon: LayoutDashboard },
    { id: 'student-profile', label: 'Academic Profile', icon: UserCircle },
    { id: 'performance', label: 'Performance Analytics', icon: LineChart },
    { id: 'risk-dashboard', label: 'Risk & Explainable AI', icon: AlertTriangle },
    { id: 'recommendations', label: 'Personalized Resources', icon: Sparkles },
    { id: 'engagement', label: 'Engagement Telemetry', icon: Activity },
  ];

  const educatorNavItems = [
    { id: 'cohort-analytics', label: 'Cohort Intelligence', icon: Users },
  ];

  const adminNavItems = [
    { id: 'admin-overview', label: 'ML Models & Status', icon: Database },
  ];

  return (
    <aside className="w-64 border-r border-slate-800 bg-navy-900/60 backdrop-blur-md flex flex-col p-4 shrink-0">
      <div className="space-y-6">
        {/* Student Section */}
        <div>
          <p className="px-3 text-[10px] font-bold uppercase tracking-wider text-teal-400/80 mb-2">
            Learner Workspace
          </p>
          <nav className="space-y-1">
            {studentNavItems.map((item) => {
              const Icon = item.icon;
              const active = currentView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                    active
                      ? 'bg-teal-500/15 text-teal-300 border border-teal-500/30 shadow-sm shadow-teal-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${active ? 'text-teal-400' : 'text-slate-400'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Educator Section */}
        <div>
          <p className="px-3 text-[10px] font-bold uppercase tracking-wider text-amber-400/80 mb-2">
            Educator & Cohort
          </p>
          <nav className="space-y-1">
            {educatorNavItems.map((item) => {
              const Icon = item.icon;
              const active = currentView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                    active
                      ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${active ? 'text-amber-400' : 'text-slate-400'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Admin Section */}
        <div>
          <p className="px-3 text-[10px] font-bold uppercase tracking-wider text-purple-400/80 mb-2">
            System & MLOps
          </p>
          <nav className="space-y-1">
            {adminNavItems.map((item) => {
              const Icon = item.icon;
              const active = currentView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                    active
                      ? 'bg-purple-500/15 text-purple-300 border border-purple-500/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${active ? 'text-purple-400' : 'text-slate-400'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      <div className="mt-auto pt-4 border-t border-slate-800/80">
        <div className="p-3 rounded-lg bg-navy-950/70 border border-slate-800 text-[11px] text-slate-400">
          <p className="font-semibold text-slate-300">MintX Solutions Platform</p>
          <p className="text-[10px] text-slate-500 mt-0.5">OULAD Machine Learning Benchmark</p>
        </div>
      </div>
    </aside>
  );
};
