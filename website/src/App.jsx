import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';

import Hero from './components/Hero';
import WLKernelExplanation from './components/WLKernelExplanation';
import PipelineShowcase from './components/PipelineShowcase';
import ResultsVisualization from './components/ResultsVisualization';

function Navigation() {
  const location = useLocation();
  
  return (
    <nav>
      <Link to="/" className="logo">WL<span className="text-gradient">Graph</span></Link>
      <div className="links">
        <Link to="/" className={location.pathname === '/' ? 'active' : ''}>Home</Link>
        <Link to="/kernel" className={location.pathname === '/kernel' ? 'active' : ''}>WL Kernel</Link>
        <Link to="/pipeline" className={location.pathname === '/pipeline' ? 'active' : ''}>Pipeline</Link>
        <Link to="/results" className={location.pathname === '/results' ? 'active' : ''}>Results</Link>
      </div>
    </nav>
  );
}

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<Hero />} />
        <Route path="/kernel" element={<WLKernelExplanation />} />
        <Route path="/pipeline" element={<PipelineShowcase />} />
        <Route path="/results" element={<ResultsVisualization />} />
      </Routes>
    </AnimatePresence>
  );
}

function App() {
  return (
    <Router>
      <Navigation />
      <AnimatedRoutes />
    </Router>
  );
}

export default App;
