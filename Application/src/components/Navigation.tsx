import React from 'react';
import { PageView } from '../types';

interface NavigationProps {
  currentPage: PageView;
  onNavigate: (page: PageView) => void;
}

export const Navigation: React.FC<NavigationProps> = ({ currentPage, onNavigate }) => {
  const navItems: { id: PageView; label: string; icon: string; badge?: string }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
    { id: 'analyze-plant', label: 'Analyze Plant', icon: 'photo_camera', badge: 'AI SCAN' },
    { id: 'results-diagnostics', label: 'Diagnostics', icon: 'biotech' },
    { id: 'plant-stress-library', label: 'Stress Library', icon: 'menu_book' },
    { id: 'my-plants-health-tracking', label: 'My Plants', icon: 'psychiatry' },
    { id: 'history-logs', label: 'History & Logs', icon: 'history' },
    { id: 'model-pipeline', label: 'ML Pipeline', icon: 'neurology' },
    { id: 'settings', label: 'Settings', icon: 'settings' },
  ];

  const mobileNavItems: { id: PageView; label: string; icon: string }[] = [
    { id: 'dashboard', label: 'Home', icon: 'dashboard' },
    { id: 'analyze-plant', label: 'Analyze', icon: 'photo_camera' },
    { id: 'my-plants-health-tracking', label: 'Plants', icon: 'psychiatry' },
    { id: 'history-logs', label: 'History', icon: 'history' },
    { id: 'plant-stress-library', label: 'Learn', icon: 'menu_book' },
  ];

  return (
    <>
      {/* Desktop Sidebar */}
      <aside
        id="desktop-sidebar"
        className="fixed left-0 top-0 h-full w-64 bg-[#f2f3ff] z-50 hidden md:flex flex-col justify-between py-4 px-3 shadow-[0_1px_8px_rgba(0,0,0,0.04)] border-r border-[#dae2fd]/40"
      >
        <div className="flex flex-col gap-4">
          {/* Logo & Brand */}
          <button
            id="brand-logo-btn"
            onClick={() => onNavigate('dashboard')}
            className="flex items-center gap-2.5 px-2 py-1 text-left rounded-lg hover:bg-[#e2e7ff]/60 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 40" fill="none" className="h-10 w-auto shrink-0">
              <rect x="2" y="4" width="32" height="32" rx="8" fill="#15803d"/>
              <path d="M18 10C18 10 26 13 26 21C26 26.5 21.5 28 18 28C14.5 28 10 26.5 10 21C10 13 18 10 18 10Z" fill="#86efac"/>
              <path d="M18 14V27" stroke="#15803d" strokeWidth="2" strokeLinecap="round"/>
              <path d="M18 19L22 16" stroke="#15803d" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M18 22L14 19.5" stroke="#15803d" strokeWidth="1.5" strokeLinecap="round"/>
              <circle cx="27" cy="9" r="3" fill="#22c55e"/>
              <path d="M26 6L28 12M24 8L30 10" stroke="#ffffff" strokeWidth="1" strokeLinecap="round"/>
              <text x="42" y="24" fill="#0f172a" fontFamily="Plus Jakarta Sans, sans-serif" fontSize="17" fontWeight="800" letterSpacing="-0.5px">PlantCare<tspan fill="#16a34a">.AI</tspan></text>
              <text x="43" y="33" fill="#64748b" fontFamily="Plus Jakarta Sans, sans-serif" fontSize="8" fontWeight="600" letterSpacing="0.8px">PRECISION AGRONOMY</text>
            </svg>
          </button>

          {/* Navigation Links */}
          <nav id="desktop-nav-menu" className="flex flex-col gap-1">
            {navItems.map((item) => {
              const isActive = currentPage === item.id;
              return (
                <button
                  key={item.id}
                  id={`nav-item-${item.id}`}
                  onClick={() => onNavigate(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-all text-left font-title-sm text-title-sm cursor-pointer ${
                    isActive
                      ? 'bg-[#14532d] text-white font-semibold shadow-sm'
                      : 'text-[#404941] hover:bg-[#e2e7ff] hover:text-[#131b2e]'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`material-symbols-outlined text-[20px] ${
                        isActive ? 'text-white' : item.id === 'analyze-plant' ? 'text-[#006e2d]' : 'text-[#717970]'
                      }`}
                    >
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </div>
                  {item.badge && (
                    <span
                      className={`font-label-sm text-[10px] px-1.5 py-0.5 rounded-full font-bold uppercase tracking-wider ${
                        isActive
                          ? 'bg-[#7cf994] text-[#007230]'
                          : 'bg-[#7cf994] text-[#007230]'
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* System Notice Box */}
        <div id="sidebar-system-notice" className="flex flex-col gap-1.5 p-3 bg-[#eaedff] rounded-xl border border-[#dae2fd]/60">
          <div className="flex items-center gap-1.5 text-[#717970]">
            <span className="material-symbols-outlined text-[16px] text-[#006e2d]">info</span>
            <span className="font-label-sm text-[10px] font-bold uppercase tracking-wider text-[#003b1b]">
              System Notice
            </span>
          </div>
          <p className="font-body-sm text-[12px] text-[#404941] leading-relaxed">
            AI Visual Assessment System - Experimental Agronomy Tool.
          </p>
        </div>
      </aside>

      {/* Mobile Bottom Navigation */}
      <nav
        id="mobile-bottom-nav"
        className="fixed bottom-0 left-0 right-0 h-16 bg-[#faf8ff]/95 backdrop-blur-xl md:hidden z-50 flex items-center justify-around px-2 shadow-[0_-1px_8px_rgba(0,0,0,0.06)] border-t border-[#dae2fd]/40"
      >
        {mobileNavItems.map((item) => {
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              id={`mobile-nav-${item.id}`}
              onClick={() => onNavigate(item.id)}
              className={`flex flex-col items-center gap-1 font-label-sm text-[11px] transition-colors py-1 ${
                isActive ? 'text-[#003b1b] font-bold' : 'text-[#404941] hover:text-[#131b2e]'
              }`}
            >
              <span
                className={`material-symbols-outlined text-[20px] ${
                  isActive ? 'text-[#003b1b]' : 'text-[#717970]'
                }`}
              >
                {item.icon}
              </span>
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </>
  );
};
