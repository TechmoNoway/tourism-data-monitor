# Application Architecture

## Frontend Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (Port 3000)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP Requests
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Vite Dev Server                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Proxy: /api → http://localhost:8000                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Proxied Requests
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Backend API (Port 8000)                        │
│  • /api/provinces/                                              │
│  • /api/attractions/                                            │
│  • /api/comments/                                               │
│  • /api/comments/stats                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Component Hierarchy

```
App.tsx (Router)
│
├─ Layout
│  ├─ Header
│  │  └─ Navigation & Logo
│  │
│  ├─ Main Content Area
│  │  │
│  │  ├─ Route: / → HomePage
│  │  │  ├─ SearchBar (unified search)
│  │  │  ├─ Stats Cards
│  │  │  ├─ Province Grid
│  │  │  └─ Featured Attractions
│  │  │
│  │  ├─ Route: /province/:id → ProvincePage
│  │  │  ├─ Province Header
│  │  │  ├─ Search + Filter Bar
│  │  │  │  └─ Filter Buttons (All/Trending/Positive/Negative)
│  │  │  └─ Attractions Grid
│  │  │     └─ AttractionCard (reusable)
│  │  │
│  │  └─ Route: /attraction/:id → AttractionPage
│  │     ├─ Attraction Header
│  │     ├─ Stats Cards (4 cards)
│  │     ├─ Trend Chart (Recharts)
│  │     ├─ Sentiment Filters
│  │     ├─ Aspect Tabs
│  │     └─ Comments List
│  │
│  └─ Footer
│
└─ Services Layer (api.ts)
   └─ Axios HTTP Client
```

## Data Flow

```
User Action
    ↓
Component Event Handler
    ↓
Call API Service Function
    ↓
Axios HTTP Request
    ↓
Vite Proxy
    ↓
Backend API
    ↓
Database Query
    ↓
JSON Response
    ↓
Update Component State
    ↓
Re-render UI
```

## Page Flow Diagram

```
┌─────────────┐
│  HomePage   │
│  (Search)   │
└──────┬──────┘
       │
       ├─────────┐
       │         │
       ↓         ↓
┌──────────┐  ┌─────────────┐
│ Province │  │ Attraction  │
│   Page   │→ │    Page     │
└──────────┘  └─────────────┘
       ↑              ↑
       │              │
       └──────┬───────┘
              │
        Back Navigation
```

## State Management Flow

```
Component State (useState)
├─ attractions: TouristAttraction[]
├─ filteredAttractions: TouristAttraction[]
├─ searchQuery: string
├─ activeFilter: FilterType
├─ activeAspect: string
├─ activeSentiment: string
└─ isLoading: boolean

Effects (useEffect)
├─ Fetch data on mount
├─ Filter when query/filter changes
└─ Debounce search input
```

## API Integration Points

```
HomePage
├─ getProvinces()
└─ getAttractions({ limit: 8 })

ProvincePage
├─ getProvinceById(id)
└─ getAttractions({ province_id: id })

AttractionPage
├─ getAttractionById(id)
├─ getComments({ attraction_id: id })
└─ getCommentStats(id)

SearchBar
└─ search(query)
    ├─ getProvinces(query)
    └─ getAttractions({ search: query })
```

## Filter Logic Flow

### Province Page Filters

```
All Attractions
    ↓
Search Filter (if query exists)
    ↓
Sort by Active Filter:
├─ All: No sorting
├─ Trending: Sort by total_comments DESC
├─ Positive: Sort by positive_count/total_comments DESC
└─ Negative: Sort by negative_count/total_comments DESC
    ↓
Display Filtered Results
```

### Attraction Page Filters

```
All Comments
    ↓
Filter by Active Aspect (if not 'all')
    ↓
Filter by Active Sentiment (if not 'all')
    ↓
Display Filtered Comments
```

## Technology Stack Layers

```
┌──────────────────────────────────────┐
│         User Interface Layer         │
│  React Components + Tailwind CSS     │
└──────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────┐
│        State Management Layer        │
│     React Hooks (useState/Effect)    │
└──────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────┐
│         Routing Layer                │
│         React Router v6              │
└──────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────┐
│         API Service Layer            │
│        Axios HTTP Client             │
└──────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────┐
│         Backend API Layer            │
│    FastAPI + PostgreSQL Database     │
└──────────────────────────────────────┘
```

## Build & Development Workflow

```
Development
    npm run dev
        ↓
    Vite Dev Server
        ↓
    Hot Module Replacement (HMR)
        ↓
    Instant Browser Updates

Production
    npm run build
        ↓
    TypeScript Compilation
        ↓
    Vite Build (Rollup)
        ↓
    Optimized Bundle in dist/
        ↓
    Deploy to Server
```

## File Responsibility Matrix

| File | Responsibility |
|------|---------------|
| `App.tsx` | Router configuration, route definitions |
| `Layout.tsx` | Page layout wrapper, header, footer |
| `Header.tsx` | Top navigation, branding |
| `SearchBar.tsx` | Unified search, autocomplete, navigation |
| `AttractionCard.tsx` | Reusable attraction display component |
| `HomePage.tsx` | Landing page, search interface, overview |
| `ProvincePage.tsx` | Province details, attraction list, filters |
| `AttractionPage.tsx` | Attraction details, charts, comments, aspects |
| `api.ts` | All backend API calls, data fetching |
| `index.css` | Global styles, Tailwind imports |
| `vite.config.ts` | Build configuration, dev server, proxy |
| `tailwind.config.js` | Theme, colors, styling configuration |

## Key Features Implementation

### Unified Search
- Component: `SearchBar.tsx`
- Debounced input (300ms)
- Parallel API calls
- Dropdown results with categories
- Click-to-navigate

### Aspect Filtering
- Component: `AttractionPage.tsx`
- Extract unique aspects from comments
- Create dynamic tabs
- Filter comments by selected aspect
- Combine with sentiment filter

### Trend Visualization
- Component: `AttractionPage.tsx`
- Library: Recharts
- Data: Weekly comment counts
- Type: Line chart
- Interactive tooltips

### Sentiment Analysis
- Colors:
  - Positive: Green (#10b981)
  - Negative: Red (#ef4444)
  - Neutral: Gray (#6b7280)
- Display: Badges, cards, filters
- Calculation: Percentage of total

---

This architecture provides:
- ✅ Separation of concerns
- ✅ Reusable components
- ✅ Type safety (TypeScript)
- ✅ Efficient state management
- ✅ Clean API integration
- ✅ Responsive design
- ✅ Performance optimization
