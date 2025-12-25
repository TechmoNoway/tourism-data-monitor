# Tourism Data Monitor - Frontend

Modern React frontend built with Vite, TypeScript, and TailwindCSS for the Tourism Data Monitor backend.

## Features

- 🔍 **Unified Search**: Search both provinces and tourist attractions from a single search bar
- 🗺️ **Province Pages**: Browse attractions by province with advanced filtering
- 📊 **Real-time Analytics**: View comment trends, sentiment analysis, and statistics
- 🎨 **Modern UI**: Beautiful, responsive design with Tailwind CSS
- 📱 **Mobile-Friendly**: Fully responsive across all devices
- 🎯 **Aspect Analysis**: Filter comments by specific aspects
- 📈 **Trend Charts**: Visualize comment trends over time with Recharts

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **TailwindCSS** - Styling
- **React Router** - Routing
- **Axios** - API requests
- **Recharts** - Data visualization
- **Lucide React** - Icons

## Getting Started

### Prerequisites

- Node.js 18+ 
- Yarn package manager

### Installation

1. Install dependencies:
```bash
yarn install
```

2. Start the development server:
```bash
yarn dev
```

The app will be available at `http://localhost:3000`

### Building for Production

```bash
yarn build
```

The built files will be in the `dist` directory.

### Preview Production Build

```bash
yarn preview
```

## Project Structure

```
web/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── Header.tsx
│   │   ├── Layout.tsx
│   │   ├── SearchBar.tsx
│   │   └── AttractionCard.tsx
│   ├── pages/              # Page components
│   │   ├── HomePage.tsx
│   │   ├── ProvincePage.tsx
│   │   └── AttractionPage.tsx
│   ├── services/           # API services
│   │   └── api.ts
│   ├── App.tsx             # Main app component with routing
│   ├── main.tsx            # App entry point
│   └── index.css           # Global styles
├── public/                 # Static assets
├── index.html
├── vite.config.ts
├── tailwind.config.js
└── package.json
```

## Features Overview

### Home Page
- Hero section with search bar
- Statistics cards
- Province grid
- Featured attractions

### Province Page
- Province header with details
- Search bar for attractions within province
- Filter buttons (All, Trending, Positive, Negative)
- Attraction cards grid

### Attraction Page
- Attraction details with image
- Statistics cards (Total Comments, Positive %, Negative %, Neutral %)
- Weekly comment trend chart
- Aspect-based filtering tabs
- Sentiment filtering
- Comments list with sentiment indicators

## API Integration

The frontend connects to the backend API running on `http://localhost:8000`. The proxy is configured in `vite.config.ts`:

```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  },
}
```

Make sure your backend is running before starting the frontend.

## Development

### Available Scripts

- `yarn dev` - Start development server
- `yarn build` - Build for production
- `yarn preview` - Preview production build
- `yarn lint` - Run ESLint

### Environment Variables

Create a `.env` file if you need to customize the API URL:

```env
VITE_API_URL=http://localhost:8000
```

## Customization

### Colors

The primary color scheme can be customized in [tailwind.config.js](tailwind.config.js):

```javascript
colors: {
  primary: {
    // Customize these values
    500: '#0ea5e9',
    600: '#0284c7',
    // ...
  },
}
```

### API Endpoints

All API calls are centralized in [src/services/api.ts](src/services/api.ts). Modify this file to adjust endpoints or add new API methods.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

This project is part of the Tourism Data Monitor system.
