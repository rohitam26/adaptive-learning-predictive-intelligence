import React, { useState } from 'react';
import { X, Lock, Mail, User, Sparkles } from 'lucide-react';
import { ApiClient } from '../api/client';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (userData: any) => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('student@mintx.edu');
  const [password, setPassword] = useState('Pass123!');
  const [fullName, setFullName] = useState('Demo Student');
  const [role, setRole] = useState<'student' | 'educator' | 'admin'>('student');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      if (isLogin) {
        const res = await ApiClient.login(email, password);
        onSuccess(res);
      } else {
        // Register then login
        await fetch('/api/v1/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email,
            password,
            full_name: fullName,
            role,
            student_id: 100001
          })
        });
        const res = await ApiClient.login(email, password);
        onSuccess(res);
      }
      onClose();
    } catch (err: any) {
      setError(err.message || 'Authentication error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickPreset = (presetRole: 'student' | 'educator' | 'admin') => {
    if (presetRole === 'student') {
      setEmail('student@mintx.edu');
      setPassword('Pass123!');
    } else if (presetRole === 'educator') {
      setEmail('educator@mintx.edu');
      setPassword('Pass123!');
    } else {
      setEmail('admin@mintx.edu');
      setPassword('Pass123!');
    }
    setIsLogin(true);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-navy-950/80 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-md bg-navy-900 border border-slate-700/80 rounded-2xl p-6 shadow-2xl relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800"
        >
          <X className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2 mb-2">
          <div className="w-8 h-8 rounded-lg bg-teal-500/20 text-teal-400 flex items-center justify-center">
            <Sparkles className="w-4 h-4" />
          </div>
          <h3 className="text-lg font-bold text-white">
            {isLogin ? 'Sign In to Platform' : 'Create Student Account'}
          </h3>
        </div>
        <p className="text-xs text-slate-400 mb-6">
          Access individualized predictive risk factors and hybrid recommendations
        </p>

        {/* Quick Demo Fill Buttons */}
        <div className="mb-6 p-3 bg-navy-950/80 border border-slate-800 rounded-xl">
          <p className="text-[10px] uppercase font-bold text-teal-400/90 tracking-wider mb-2">
            Instant Demo Credentials
          </p>
          <div className="grid grid-cols-3 gap-2">
            <button
              type="button"
              onClick={() => handleQuickPreset('student')}
              className="py-1 px-2 text-[11px] rounded bg-teal-500/10 text-teal-300 border border-teal-500/30 hover:bg-teal-500/20"
            >
              Student
            </button>
            <button
              type="button"
              onClick={() => handleQuickPreset('educator')}
              className="py-1 px-2 text-[11px] rounded bg-amber-500/10 text-amber-300 border border-amber-500/30 hover:bg-amber-500/20"
            >
              Educator
            </button>
            <button
              type="button"
              onClick={() => handleQuickPreset('admin')}
              className="py-1 px-2 text-[11px] rounded bg-purple-500/10 text-purple-300 border border-purple-500/30 hover:bg-purple-500/20"
            >
              Admin
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-rose-500/15 border border-rose-500/30 text-rose-300 text-xs">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {!isLogin && (
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Full Name</label>
              <div className="relative">
                <User className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full bg-navy-950 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500"
                  required
                />
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">Email Address</label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-navy-950 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">Password</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-navy-950 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600 text-white font-semibold rounded-lg text-xs shadow-md shadow-teal-500/20 disabled:opacity-50 transition-all"
          >
            {loading ? 'Authenticating...' : isLogin ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <div className="mt-4 text-center">
          <button
            type="button"
            onClick={() => setIsLogin(!isLogin)}
            className="text-xs text-teal-400 hover:underline"
          >
            {isLogin ? "Need a new account? Register" : "Already registered? Sign In"}
          </button>
        </div>
      </div>
    </div>
  );
};
