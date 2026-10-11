/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#172033',
        mist: '#f7f8fc',
        brand: '#5b5ce2',
      },
      boxShadow: {
        panel: '0 18px 50px rgba(28, 39, 71, 0.08)',
      },
    },
  },
  plugins: [],
}
