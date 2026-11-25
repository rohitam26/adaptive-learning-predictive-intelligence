/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#070b14',
          900: '#0b1120',
          850: '#0f172a',
          800: '#1e293b',
          700: '#334155',
          600: '#475569'
        },
        teal: {
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e'
        }
      }
    },
  },
  plugins: [],
}
