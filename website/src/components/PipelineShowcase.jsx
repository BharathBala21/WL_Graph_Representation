import React from 'react';
import { motion } from 'framer-motion';

const PipelineShowcase = () => {
  const pipelineSteps = [
    {
      title: "08. Feature Extraction",
      description: "Extracting linguistic and structural features from the OLID dataset to form the initial node attributes for our graphs.",
      icon: "📊",
      color: "var(--accent-purple)"
    },
    {
      title: "09. Feature Validation",
      description: "Validating the extracted features to ensure robustness and consistency before graph construction.",
      icon: "✅",
      color: "var(--accent-magenta)"
    },
    {
      title: "10. Segmentation Bias Audit",
      description: "Auditing the dataset for segmentation biases to prevent skewed graph representations and model prejudice.",
      icon: "🔍",
      color: "var(--accent-cyan)"
    },
    {
      title: "11. Final Graph Construction",
      description: "Constructing the final graphs using the validated features and applying WL Kernels to capture deep structural topologies.",
      icon: "🕸️",
      color: "var(--text-primary)"
    }
  ];

  return (
    <motion.section 
      initial={{ opacity: 0, y: 50 }} 
      animate={{ opacity: 1, y: 0 }} 
      exit={{ opacity: 0, y: -50 }}
      className="container"
    >
      <h2 style={{ fontSize: '3rem', marginBottom: '60px', textAlign: 'center' }}>
        The <span className="text-gradient">Full Pipeline</span>
      </h2>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '40px', position: 'relative' }}>
        {/* Connecting Line */}
        <div style={{ position: 'absolute', left: '50px', top: '50px', bottom: '50px', width: '2px', background: 'var(--glass-border)', zIndex: -1 }} />

        {pipelineSteps.map((step, idx) => (
          <motion.div 
            key={idx}
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.5, delay: idx * 0.2 }}
            className="glass-panel"
            style={{ display: 'flex', gap: '30px', alignItems: 'center', marginLeft: '20px' }}
          >
            <div style={{ 
              width: '60px', 
              height: '60px', 
              borderRadius: '50%', 
              background: step.color,
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              fontSize: '2rem',
              boxShadow: `0 0 20px ${step.color}80`
            }}>
              {step.icon}
            </div>
            <div>
              <h3 style={{ fontSize: '1.8rem', marginBottom: '10px' }}>{step.title}</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', lineHeight: '1.5' }}>
                {step.description}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.section>
  );
};

export default PipelineShowcase;
