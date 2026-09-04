import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const Hero = () => {
  return (
    <motion.section 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      exit={{ opacity: 0 }}
      className="container"
      style={{ alignItems: 'center', textAlign: 'center' }}
    >
      <motion.div
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        className="glass-panel"
        style={{ maxWidth: '800px', margin: '0 auto', padding: '60px' }}
      >
        <motion.h1 
          style={{ fontSize: '4rem', marginBottom: '20px', lineHeight: '1.2' }}
        >
          Unveiling Structure with <br/>
          <span className="text-gradient">WL Graph Kernels</span>
        </motion.h1>
        
        <motion.p 
          style={{ fontSize: '1.2rem', color: 'var(--text-secondary)', marginBottom: '40px', lineHeight: '1.6' }}
        >
          A comprehensive exploration of the Weisfeiler-Lehman Graph Representation methodology. 
          From feature extraction and segmentation bias audit to final graph construction, 
          discover how complex topologies are transformed into actionable embeddings.
        </motion.p>
        
        <motion.div 
          style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}
        >
          <Link to="/kernel" className="btn-primary">Explore WL Kernels</Link>
          <Link to="/pipeline" className="btn-secondary">View Pipeline</Link>
        </motion.div>
      </motion.div>
      
      {/* Abstract Background Animation */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1, overflow: 'hidden' }}>
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            style={{
              position: 'absolute',
              background: i % 2 === 0 ? 'var(--accent-purple)' : 'var(--accent-cyan)',
              borderRadius: '50%',
              filter: 'blur(80px)',
              opacity: 0.15,
            }}
            animate={{
              x: [Math.random() * window.innerWidth, Math.random() * window.innerWidth],
              y: [Math.random() * window.innerHeight, Math.random() * window.innerHeight],
              width: [100 + Math.random() * 200, 100 + Math.random() * 200],
              height: [100 + Math.random() * 200, 100 + Math.random() * 200],
            }}
            transition={{
              duration: 10 + Math.random() * 20,
              repeat: Infinity,
              repeatType: 'reverse',
              ease: 'linear'
            }}
          />
        ))}
      </div>
    </motion.section>
  );
};

export default Hero;
