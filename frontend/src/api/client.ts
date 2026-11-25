/**
 * API client communicating with FastAPI backend.
 */

const API_BASE = '/api/v1';

export class ApiClient {
  private static token: string | null = localStorage.getItem('token');

  static setToken(token: string | null) {
    this.token = token;
    if (token) {
      localStorage.setItem('token', token);
    } else {
      localStorage.removeItem('token');
    }
  }

  static getToken(): string | null {
    return this.token;
  }

  private static async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP Error ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  // Auth
  static async login(email: string, password: string) {
    const data = await this.request<{
      access_token: string;
      role: string;
      student_id?: number;
      full_name: string;
    }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    this.setToken(data.access_token);
    return data;
  }

  static async getMe() {
    return this.request<any>('/students/me');
  }

  // Students
  static async getStudentProfile(studentId: number) {
    return this.request<any>(`/students/${studentId}/profile`);
  }

  static async getStudentPerformance(studentId: number) {
    return this.request<any>(`/students/${studentId}/performance`);
  }

  static async getStudentRisk(studentId: number) {
    return this.request<any>(`/students/${studentId}/risk`);
  }

  static async getStudentRecommendations(studentId: number) {
    return this.request<any>(`/students/${studentId}/recommendations`);
  }

  static async getStudentEngagement(studentId: number) {
    return this.request<any>(`/students/${studentId}/engagement`);
  }

  // Analytics
  static async getCohortAnalytics() {
    return this.request<any>('/analytics/cohort');
  }

  static async getResourceAnalytics() {
    return this.request<any>('/analytics/resources');
  }

  static async getModelsStatus() {
    return this.request<any>('/models/status');
  }
}
