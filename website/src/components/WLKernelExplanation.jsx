import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const WLKernelExplanation = () => {
  const [step, setStep] = useState(0);
  
  const steps = [
    { title: "1. Initial Labeling", desc: "Every node in the graph is assigned an initial label representing its feature." },
    { title: "2. Neighborhood Aggregation", desc: "Nodes collect labels from their immediate neighbors to form a multiset." },
    { title: "3. Label Sorting", desc: "The multiset of neighbor labels is sorted to ensure permutation invariance." },
    { title: "4. Hash / Re-labeling", desc: "The node's current label and its sorted multiset are hashed into a new, unique label." },
    { title: "5. Iteration", desc: "This process repeats for 'h' iterations, capturing structural information up to 'h' hops away." }
  ];

  // Graph topology
  const nodes = [
    { id: 1, cx: 100, cy: 100, initial: 'A' },
    { id: 2, cx: 250, cy: 60, initial: 'B' },
    { id: 3, cx: 180, cy: 220, initial: 'B' },
    { id: 4, cx: 380, cy: 140, initial: 'A' },
  ];

  const edges = [
    { source: 1, target: 2 },
    { source: 1, target: 3 },
    { source: 2, target: 4 },
  ];

  const getLabel = (nodeId, currentStep) => {
    if (currentStep === 0) return nodes.find(n => n.id === nodeId).initial;
    if (currentStep === 1) {
      if (nodeId === 1) return 'A, {B, B}';
      if (nodeId === 2) return 'B, {A, A}';
      if (nodeId === 3) return 'B, {A}';
      if (nodeId === 4) return 'A, {B}';
    }
    if (currentStep === 2) {
      // sorted
      if (nodeId === 1) return 'A, {B, B}';
      if (nodeId === 2) return 'B, {A, A}';
      if (nodeId === 3) return 'B, {A}';
      if (nodeId === 4) return 'A, {B}';
    }
    // Hashed
    if (nodeId === 1) return 'C1';
    if (nodeId === 2) return 'C2';
    if (nodeId === 3) return 'C3';
    if (nodeId === 4) return 'C4';
  };

  const getColor = (currentStep, nodeId) => {
    if (currentStep === 0) return 'var(--accent-purple)';
    if (currentStep < 3) return 'var(--accent-magenta)';
    // In hashed step, color nodes with same hash identically
    if (currentStep >= 3) {
       if (nodeId === 1) return 'var(--accent-cyan)';
       if (nodeId === 2) return '#ffaa00';
       if (nodeId === 3) return '#00ffaa';
       if (nodeId === 4) return '#ff00aa';
    }
    return 'var(--accent-cyan)';
  };

  // Helper to get coordinates for message passing animation
  const getEdgeMidpoint = (sourceId, targetId) => {
    const s = nodes.find(n => n.id === sourceId);
    const t = nodes.find(n => n.id === targetId);
    return { x: (s.cx + t.cx) / 2, y: (s.cy + t.cy) / 2 };
  };

  return (
    <motion.section 
      initial={{ opacity: 0, x: 100 }} 
      animate={{ opacity: 1, x: 0 }} 
      exit={{ opacity: 0, x: -100 }}
      className="container"
    >
      <h2 style={{ fontSize: '3rem', marginBottom: '40px' }}>
        How <span className="text-gradient">WL Kernels</span> Work
      </h2>
      
      <div style={{ display: 'flex', gap: '40px', minHeight: '500px', flexDirection: 'row', flexWrap: 'wrap' }}>
        
        {/* Step Navigation */}
        <div className="glass-panel" style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {steps.map((s, idx) => (
            <div 
              key={idx} 
              onClick={() => setStep(idx)}
              style={{
                padding: '20px',
                borderRadius: '12px',
                cursor: 'pointer',
                background: step === idx ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                border: `1px solid ${step === idx ? 'var(--accent-cyan)' : 'var(--glass-border)'}`,
                transition: 'all 0.3s'
              }}
            >
              <h3 style={{ color: step === idx ? 'var(--accent-cyan)' : 'var(--text-primary)', fontSize: '1.2rem' }}>{s.title}</h3>
              <AnimatePresence>
                {step === idx && (
                  <motion.p 
                    initial={{ opacity: 0, height: 0 }} 
                    animate={{ opacity: 1, height: 'auto' }} 
                    exit={{ opacity: 0, height: 0 }}
                    style={{ marginTop: '10px', color: 'var(--text-secondary)', fontSize: '0.95rem' }}
                  >
                    {s.desc}
                  </motion.p>
                )}
              </AnimatePresence>
            </div>
          ))}
          <div style={{ marginTop: 'auto', display: 'flex', justifyContent: 'space-between', paddingTop: '20px' }}>
            <button className="btn-secondary" onClick={() => setStep(Math.max(0, step - 1))}>Previous</button>
            <button className="btn-primary" onClick={() => setStep(Math.min(steps.length - 1, step + 1))}>Next</button>
          </div>
        </div>

        {/* Visualizer */}
        <div className="glass-panel" style={{ flex: '2 1 500px', position: 'relative', display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
          <svg width="100%" height="100%" viewBox="0 0 500 300" style={{ overflow: 'visible' }}>
            {/* Edges */}
            {edges.map((edge, idx) => {
              const source = nodes.find(n => n.id === edge.source);
              const target = nodes.find(n => n.id === edge.target);
              return (
                <g key={`edge-${idx}`}>
                  <motion.line 
                    x1={source.cx} 
                    y1={source.cy} 
                    x2={target.cx} 
                    y2={target.cy} 
                    stroke="var(--glass-border)" 
                    strokeWidth="4" 
                    animate={{
                      stroke: step === 1 ? 'rgba(255, 255, 255, 0.5)' : 'rgba(255, 255, 255, 0.1)'
                    }}
                  />
                  {/* Message Passing Animation (Step 1) */}
                  {step === 1 && (
                    <>
                      <motion.circle
                        r="6"
                        fill="var(--accent-cyan)"
                        initial={{ cx: source.cx, cy: source.cy, opacity: 0 }}
                        animate={{ cx: target.cx, cy: target.cy, opacity: [0, 1, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                      />
                      <motion.circle
                        r="6"
                        fill="var(--accent-magenta)"
                        initial={{ cx: target.cx, cy: target.cy, opacity: 0 }}
                        animate={{ cx: source.cx, cy: source.cy, opacity: [0, 1, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut", delay: 0.75 }}
                      />
                    </>
                  )}
                </g>
              );
            })}
            
            {/* Nodes */}
            {nodes.map(node => (
              <motion.g 
                key={`node-${node.id}`}
                animate={{
                  scale: step === 1 || step === 3 ? 1.1 : 1
                }}
                transition={{ type: 'spring', stiffness: 300 }}
              >
                <motion.circle
                  cx={node.cx}
                  cy={node.cy}
                  r="35"
                  fill={getColor(step, node.id)}
                  stroke="#ffffff"
                  strokeWidth="3"
                  style={{ filter: 'drop-shadow(0px 4px 10px rgba(0,0,0,0.5))' }}
                  layout
                />
                <motion.text
                  x={node.cx}
                  y={node.cy}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill="#ffffff"
                  fontSize={step === 1 || step === 2 ? "14px" : "22px"}
                  fontWeight="bold"
                  initial={false}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  {getLabel(node.id, step)}
                </motion.text>
              </motion.g>
            ))}
          </svg>
        </div>
      </div>
    </motion.section>
  );
};

export default WLKernelExplanation;
