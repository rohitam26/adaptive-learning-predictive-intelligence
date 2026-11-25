import React, { useEffect, useState } from 'react';
import { ApiClient } from '../api/client';
import { StudentProfile } from '../types';
import { User, MapPin, Award, BookOpen, Layers, Shield } from 'lucide-react';

interface StudentProfileViewProps {
  studentId: number;
}

export const StudentProfileView: React.FC<StudentProfileViewProps> = ({ studentId }) => {
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await ApiClient.getStudentProfile(studentId);
        setProfile(data);
      } catch (err) {
        console.error('Failed to load profile:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [studentId]);

  if (loading) {
    return <div className="text-center py-12 text-slate-400 text-xs">Loading learner profile...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="p-6 rounded-2xl bg-navy-900 border border-slate-800">
        <h2 className="text-xl font-bold text-white tracking-tight mb-1">
          Learner Academic Profile & Demographics
        </h2>
        <p className="text-xs text-slate-400">
          Relational demographic records sourced from OULAD <span className="font-mono text-teal-400">studentInfo.csv</span>
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-3">
            <div className="flex items-center gap-2 text-teal-400 text-xs font-semibold uppercase">
              <User className="w-4 h-4" /> Personal & Demographic
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Student ID:</span>
                <span className="font-mono text-white font-medium">{profile?.student_id}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Gender:</span>
                <span className="text-white">{profile?.gender === 'M' ? 'Male' : 'Female'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Age Band:</span>
                <span className="text-white">{profile?.age_band}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Disability Registered:</span>
                <span className="text-white">{profile?.disability === 'Y' ? 'Yes' : 'No'}</span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-3">
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase">
              <MapPin className="w-4 h-4" /> Regional & Socio-Economic
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Geographic Region:</span>
                <span className="text-white text-right">{profile?.region}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">IMD Deprivation Decile:</span>
                <span className="text-white font-mono">{profile?.imd_band}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Highest Prior Education:</span>
                <span className="text-white text-right">{profile?.highest_education}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Previous Course Attempts:</span>
                <span className="text-white font-mono">{profile?.num_of_prev_attempts}</span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-3">
            <div className="flex items-center gap-2 text-teal-300 text-xs font-semibold uppercase">
              <BookOpen className="w-4 h-4" /> Curriculum & Enrollment
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Enrolled Modules:</span>
                <span className="font-semibold text-teal-300">{profile?.modules?.join(', ') || 'N/A'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Total Credits Studied:</span>
                <span className="font-mono text-white">{profile?.studied_credits} credits</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Academic Standing:</span>
                <span className="px-2 py-0.5 rounded bg-teal-500/10 text-teal-300 border border-teal-500/30 text-[10px] font-bold">
                  {profile?.final_result || 'In Progress'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
