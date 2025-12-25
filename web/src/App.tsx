import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import ProvincePage from './pages/ProvincePage';
import AttractionPage from './pages/AttractionPage';

function App() {
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/province/:provinceId" element={<ProvincePage />} />
          <Route path="/attraction/:attractionId" element={<AttractionPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
