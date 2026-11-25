import React, { useState, useEffect } from 'react';
import { User } from './types';
import { ApiClient } from './api/client';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { LandingPage } from './pages/LandingPage';
import { AuthModal } from './pages/AuthModal';
import { StudentDashboard } from './pages/StudentDashboard';
import { StudentProfileView } from './pages/StudentProfileView';
import { PerformanceView } from './pages/PerformanceView';
import { RiskDashboardView } from './pages/RiskDashboardView';
import { RecommendationsView } from './pages/RecommendationsView';
import { EngagementView } from './pages/EngagementView';
import { CohortAnalyticsView } from './pages/CohortAnalyticsView';
import { AdminOverviewView } from './pages/AdminOverviewView';

export function App() {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [activeStudentId, setActiveStudentId] = useState<number>(100001);
  const [currentView, setCurrentView] = useState<string>('student-dashboard');
  const [showLanding, setShowLanding] = useState<boolean>(true);

  useEffect(() => {
    // Check if token exists
    const token = ApiClient.getToken();
    if (token) {
      ApiClient.getMe()
        .then((user) => {
          setCurrentUser(user);
          if (user.student_id) {
            setActiveStudentId(user.student_id);
          }
          setShowLanding(false);
        })
        .catch(() => {
          ApiClient.setToken(null);
        });
    }
  }, []);

  const handleLogout = () => {
    ApiClient.setToken(null);
    setCurrentUser(null);
    setShowLanding(true);
  };

  const handleAuthSuccess = (userData: any) => {
    setCurrentUser({
      id: userData.student_id || 1,
      email: userData.email || 'user@mintx.edu',
      full_name: userData.full_name || 'Active User',
      role: userData.role,
      student_id: userData.student_id || 100001
    });
    if (userData.student_id) {
      setActiveStudentId(userData.student_id);
    }
    if (userData.role === 'educator') {
      setCurrentView('cohort-analytics');
    } else if (userData.role === 'admin') {
      setCurrentView('admin-overview');
    } else {
      setCurrentView('student-dashboard');
    }
    setShowLanding(false);
  };

  const handleQuickLogin = async (role: 'student' | 'educator' | 'admin') => {
    const email = `${role}@mintx.edu`;
    const password = 'Pass123!';
    try {
      const res = await ApiClient.login(email, password);
      handleAuthSuccess(res);
    } catch (e) {
      setIsAuthOpen(true);
    }
  };

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100 flex flex-col font-sans">
      <Navbar
        currentUser={currentUser}
        onLogout={handleLogout}
        onOpenAuth={() => setIsAuthOpen(true)}
        activeStudentId={activeStudentId}
        onStudentChange={setActiveStudentId}
      />

      {showLanding && !currentUser ? (
        <LandingPage
          onGetStarted={() => setIsAuthOpen(true)}
          onQuickLogin={handleQuickLogin}
        />
      ) : (
        <div className="flex-1 flex overflow-hidden">
          <Sidebar
            currentView={currentView}
            onNavigate={setCurrentView}
            userRole={currentUser?.role}
          />

          <main className="flex-1 overflow-y-auto p-8">
            <div className="max-w-6xl mx-auto">
              {currentView === 'student-dashboard' && (
                <StudentDashboard studentId={activeStudentId} onNavigate={setCurrentView} />
              )}
              {currentView === 'student-profile' && (
                <StudentProfileView studentId={activeStudentId} />
              )}
              {currentView === 'performance' && (
                <PerformanceView studentId={activeStudentId} />
              )}
              {currentView === 'risk-dashboard' && (
                <RiskDashboardView studentId={activeStudentId} />
              )}
              {currentView === 'recommendations' && (
                <RecommendationsView studentId={activeStudentId} />
              )}
              {currentView === 'engagement' && (
                <EngagementView studentId={activeStudentId} />
              )}
              {currentView === 'cohort-analytics' && (
                <CohortAnalyticsView />
              )}
              {currentView === 'admin-overview' && (
                <AdminOverviewView />
              )}
            </div>
          </main>
        </div>
      )}

      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onSuccess={handleAuthSuccess}
      />
    </div>
  );
}

export default App;
