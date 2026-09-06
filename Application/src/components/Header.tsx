import React, { useState, useRef, useEffect } from 'react';
import { PageView } from '../types';

interface HeaderProps {
  onNavigate: (page: PageView) => void;
  onSearchSelect?: (term: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ onNavigate, onSearchSelect }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [showNotifications, setShowNotifications] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showSearchResults, setShowSearchResults] = useState(false);

  const notifRef = useRef<HTMLDivElement>(null);
  const profileRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setShowNotifications(false);
      }
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) {
        setShowProfileMenu(false);
      }
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowSearchResults(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const searchSuggestions = [
    { title: 'Tomato Water Stress', type: 'Abiotic Stress', target: 'results-diagnostics' as PageView },
    { title: 'Maize Common Rust (Puccinia)', type: 'Pathogen', target: 'plant-stress-library' as PageView },
    { title: 'Soybean Iron Chlorosis (IDC)', type: 'Deficiency', target: 'plant-stress-library' as PageView },
    { title: 'Hydroponic Basil B2', type: 'Monitored Crop', target: 'my-plants-health-tracking' as PageView },
    { title: 'Bell Pepper Plot C', type: 'Monitored Crop', target: 'my-plants-health-tracking' as PageView },
    { title: 'NDVI & VARI Spectral Band Calibration', type: 'Model Pipeline', target: 'model-pipeline' as PageView },
  ];

  const filteredSuggestions = searchTerm.trim()
    ? searchSuggestions.filter(s =>
        s.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.type.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : [];

  const handleSelectSuggestion = (item: typeof searchSuggestions[0]) => {
    setShowSearchResults(false);
    setSearchTerm('');
    if (onSearchSelect) onSearchSelect(item.title);
    onNavigate(item.target);
  };

  return (
    <header
      id="main-app-header"
      className="fixed top-0 left-0 md:left-64 right-0 h-16 bg-[#faf8ff]/90 backdrop-blur-xl z-40 flex items-center justify-between px-4 lg:px-6 shadow-[0_1px_8px_rgba(0,0,0,0.04)] border-b border-[#dae2fd]/40"
    >
      {/* Global Search Bar */}
      <div className="flex items-center gap-3 flex-1 max-w-xl" ref={searchRef}>
        <div className="relative w-full flex items-center">
          <span className="material-symbols-outlined absolute left-3 text-[#717970] text-[18px]">
            search
          </span>
          <input
            id="global-search-input"
            type="text"
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setShowSearchResults(true);
            }}
            onFocus={() => setShowSearchResults(true)}
            placeholder="Search symptoms, plant species (e.g. Tomato, Maize), or stresses..."
            className="w-full h-10 pl-9 pr-4 bg-white text-[#131b2e] placeholder-[#717970] font-body-sm text-[13px] rounded-lg border border-[#c0c9be]/50 focus:outline-none focus:ring-2 focus:ring-[#006e2d]/30 focus:border-[#006e2d] transition-all shadow-xs"
          />

          {/* Search Autocomplete Popup */}
          {showSearchResults && filteredSuggestions.length > 0 && (
            <div className="absolute top-12 left-0 right-0 bg-white rounded-xl shadow-xl border border-[#dae2fd] overflow-hidden z-50 p-1.5 animate-in fade-in slide-in-from-top-1">
              <div className="px-2.5 py-1 text-[11px] font-bold uppercase tracking-wider text-[#717970]">
                Quick Agronomy Search
              </div>
              {filteredSuggestions.map((item, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSelectSuggestion(item)}
                  className="w-full px-3 py-2 text-left rounded-lg hover:bg-[#f2f3ff] transition-colors flex items-center justify-between cursor-pointer group"
                >
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-[18px] text-[#006e2d] group-hover:scale-110 transition-transform">
                      psychiatry
                    </span>
                    <span className="font-title-sm text-[13px] text-[#131b2e] group-hover:text-[#003b1b]">
                      {item.title}
                    </span>
                  </div>
                  <span className="font-label-sm text-[10px] px-2 py-0.5 rounded bg-[#eaedff] text-[#404941]">
                    {item.type}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-2 lg:gap-4 ml-2 sm:ml-4">
        {/* ML Engine Status Badge */}
        <button
          id="ml-engine-badge"
          onClick={() => onNavigate('model-pipeline')}
          title="Click to view ML Pipeline architecture"
          className="hidden xl:flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#e2e7ff] text-[#131b2e] hover:bg-[#dae2fd] transition-colors cursor-pointer border border-[#dae2fd]"
        >
          <span className="w-2 h-2 rounded-full bg-[#006e2d] animate-pulse"></span>
          <span className="font-code-sm text-[11px] text-[#404941] font-medium">
            ML Engine v2.4 Active (ResNet50 + ViT)
          </span>
        </button>

        {/* Notifications Button */}
        <div className="relative" ref={notifRef}>
          <button
            id="notifications-btn"
            aria-label="Notifications"
            onClick={() => setShowNotifications(!showNotifications)}
            className="relative p-2 rounded-full text-[#404941] hover:bg-[#e2e7ff] hover:text-[#131b2e] transition-colors cursor-pointer"
          >
            <span className="material-symbols-outlined text-[22px]">notifications</span>
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-[#ba1a1a]"></span>
          </button>

          {/* Notifications Dropdown */}
          {showNotifications && (
            <div className="absolute right-0 top-12 w-80 sm:w-96 bg-white rounded-xl shadow-2xl border border-[#dae2fd] p-3 z-50 animate-in fade-in">
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-[#eaedff]">
                <span className="font-title-sm text-title-sm text-[#131b2e]">Agronomy Telemetry Alerts</span>
                <span className="font-label-sm text-[10px] px-2 py-0.5 rounded-full bg-[#ffdad6] text-[#93000a] font-bold">
                  2 New
                </span>
              </div>
              <div className="flex flex-col gap-2">
                <div
                  onClick={() => {
                    setShowNotifications(false);
                    onNavigate('results-diagnostics');
                  }}
                  className="p-2.5 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] cursor-pointer transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-title-sm text-[12px] text-[#703a00] font-bold">
                      Water Deficit Alert (Field Tomato #04)
                    </span>
                    <span className="text-[10px] text-[#717970]">2h ago</span>
                  </div>
                  <p className="font-body-sm text-[11px] text-[#404941] mt-0.5">
                    Model inference detected 82% confidence moderate moisture stress in Quadrant B-02.
                  </p>
                </div>

                <div
                  onClick={() => {
                    setShowNotifications(false);
                    onNavigate('my-plants-health-tracking');
                  }}
                  className="p-2.5 rounded-lg bg-[#faf8ff] hover:bg-[#eaedff] cursor-pointer transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-title-sm text-[12px] text-[#006e2d] font-bold">
                      Canopy Telemetry Sync Complete
                    </span>
                    <span className="text-[10px] text-[#717970]">5h ago</span>
                  </div>
                  <p className="font-body-sm text-[11px] text-[#404941] mt-0.5">
                    108 of 142 scans returned normal NDVI indices (&gt;0.75).
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Profile Card & Dropdown */}
        <div className="relative" ref={profileRef}>
          <button
            id="profile-dropdown-btn"
            onClick={() => setShowProfileMenu(!showProfileMenu)}
            className="flex items-center gap-2 pl-2 border-l border-[#c0c9be]/40 hover:opacity-90 transition-opacity cursor-pointer text-left"
          >
            <img
              alt="Profile"
              className="w-8 h-8 rounded-full object-cover ring-2 ring-[#006e2d]/30"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuC33vFPq6yB1Rdt3h4Suk9pEW3aLGBYmMRx_nTV1PsM0l2SpvYsL3TFDdSDNythBMCNc7LouTeYZW0d41NHZW1FJBs2NCmeIOa5P231X1kthH_aeAZTRjUpPIPp_ti6pBkS1vBW6JMw_bTPat3Y-Wh8TfjwWyBUKGjpVvyoNbzXoxQ-OeC82wDpLeKsrRkp9XF4TFyV-HD_zsIaCGEBNy06yY3iOJfmsqUl9560p-5yiJ249pOi7woj"
            />
            <div className="hidden sm:flex flex-col text-left">
              <span className="font-title-sm text-[13px] text-[#131b2e] leading-none font-semibold">
                Dr. Sarah Jenkins, PhD
              </span>
              <span className="font-label-sm text-[10px] text-[#717970] leading-tight mt-0.5">
                Senior Agronomist
              </span>
            </div>
            <span className="material-symbols-outlined text-[#717970] text-[18px]">
              expand_more
            </span>
          </button>

          {/* Profile Dropdown Menu */}
          {showProfileMenu && (
            <div className="absolute right-0 top-12 w-64 bg-white rounded-xl shadow-2xl border border-[#dae2fd] p-2 z-50 animate-in fade-in">
              <div className="px-3 py-2 border-b border-[#eaedff]">
                <p className="font-title-sm text-[13px] text-[#131b2e] font-bold">Dr. Sarah Jenkins</p>
                <p className="font-body-sm text-[11px] text-[#717970]">vamsimuppagowni@gmail.com</p>
                <div className="inline-flex items-center gap-1 mt-1 px-2 py-0.5 rounded bg-[#b1f2be] text-[#003b1b] text-[10px] font-bold">
                  <span className="material-symbols-outlined text-[12px]">verified</span>
                  Lab Director Access
                </div>
              </div>
              <div className="py-1">
                <button
                  onClick={() => {
                    setShowProfileMenu(false);
                    onNavigate('settings');
                  }}
                  className="w-full px-3 py-2 text-left rounded-lg text-[12px] font-title-sm text-[#131b2e] hover:bg-[#f2f3ff] flex items-center gap-2 cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[16px] text-[#717970]">settings</span>
                  Agronomy Preferences
                </button>
                <button
                  onClick={() => {
                    setShowProfileMenu(false);
                    onNavigate('history-logs');
                  }}
                  className="w-full px-3 py-2 text-left rounded-lg text-[12px] font-title-sm text-[#131b2e] hover:bg-[#f2f3ff] flex items-center gap-2 cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[16px] text-[#717970]">history</span>
                  My Scan Audit Logs
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
