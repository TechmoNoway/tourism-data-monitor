/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f8f9fa',
          100: '#e9ecef',
          200: '#dee2e6',
          300: '#adb5bd',
          400: '#6c757d',
          500: '#495057',
          600: '#343a40',
          700: '#212529',
          800: '#1a1d20',
          900: '#0d0f10',
        },
      },
      borderRadius: {
        'sm': '0.25rem',
        DEFAULT: '0.375rem',
        'md': '0.5rem',
        'lg': '0.625rem',
        'xl': '0.75rem',
        '2xl': '0.875rem',
        '3xl': '1rem',
      },
    },
  },
  plugins: [],
}
