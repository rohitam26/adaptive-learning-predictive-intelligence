import React from 'react';
import { User } from '../types';
import { GraduationCap, LogOut, ShieldCheck, UserCheck, BookOpen } from 'lucide-react';

interface NavbarProps {
  currentUser: User | null;
  onLogout: () => void;
  onOpenAuth: () => void;
  activeStudentId: number;
  onStudentChange: (id: number) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentUser,
  onLogout,
  onOpenAuth,
  activeStudentId,
  onStudentChange
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-navy-900/90 backdrop-blur-md sticky top-0 z-40 px-6 flex items-center justify-between">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-600 to-cyan-500 flex items-center justify-center text-white shadow-lg shadow-teal-500/20">
          <GraduationCap className="w-6 h-6" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-lg text-white tracking-tight">MintX</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-400 border border-teal-500/20 font-medium">
              Intelligence
            </span>
          </div>
          <p className="text-xs text-slate-400">Adaptive Learning & Predictive Analytics</p>
        </div>
      </div>

      {/* Center Student Switcher */}
      {currentUser && (
        <div className="hidden md:flex items-center gap-3 bg-navy-950/60 border border-slate-800/80 px-3 py-1.5 rounded-lg text-xs">
          <BookOpen className="w-4 h-4 text-teal-400" />
          <span className="text-slate-400">Active Student ID:</span>
          <select
            value={activeStudentId}
            onChange={(e) => onStudentChange(Number(e.target.value))}
            className="bg-navy-900 border border-slate-700 text-teal-300 rounded px-2 py-1 text-xs focus:outline-none focus:border-teal-500"
          >
            <option value={100001}>100001 (Pass/Distinction Cohort)</option>
            <option value={100002}>100002 (At-Risk Early Alert)</option>
            <option value={100003}>100003 (Moderate Engagement)</option>
            <option value={100004}>100004 (High Performer TMA)</option>
          </select>
        </div>
      )}

      {/* User Actions */}
      <div className="flex items-center gap-4">
        {currentUser ? (
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg bg-navy-800 border border-slate-700/60 text-xs">
              <span className={`w-2 h-2 rounded-full ${
                currentUser.role === 'admin' ? 'bg-purple-400' :
                currentUser.role === 'educator' ? 'bg-amber-400' : 'bg-teal-400'
              }`} />
              <div className="text-left">
                <p className="font-semibold text-slate-200">{currentUser.full_name}</p>
                <p className="text-[10px] text-slate-400 capitalize">{currentUser.role}</p>
              </div>
            </div>
            <button
              onClick={onLogout}
              className="p-2 text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors"
              title="Sign Out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <button
            onClick={onOpenAuth}
            className="px-4 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600 text-white font-medium text-xs rounded-lg shadow-md shadow-teal-500/20 transition-all"
          >
            Sign In / Demo Access
          </button>
        )}
      </div>
    </header>
  );
};
