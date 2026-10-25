import React, { useEffect, useRef, useState } from 'react';
import { PageView } from '../types';
import { RESEARCH } from '../data/research';
import { conditionName, cropName, num } from '../lib';

interface HeaderProps {
  onNavigate: (page: PageView) => void;
  onOpenSample: (id: string) => void;
}

interface Suggestion {
  title: string;
  type: string;
  icon: string;
  go: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onNavigate, onOpenSample }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowSearchResults(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const suggestions: Suggestion[] = [
    ...RESEARCH.crops.map((c) => ({ title: c.name, type: 'Crop', icon: 'grass', go: () => onNavigate('crops') })),
    ...RESEARCH.conditions.map((c) => ({
      title: `${c.name} (${c.code})`,
      type: c.type,
      icon: 'menu_book',
      go: () => onNavigate('plant-stress-library'),
    })),
    ...RESEARCH.samples.map((s) => ({
      title: `${cropName(s.crop)} · ${conditionName(s.condition)} — ${s.file}`,
      type: 'Image',
      icon: 'image',
      go: () => onOpenSample(s.id),
    })),
  ];

  const q = searchTerm.trim().toLowerCase();
  const filteredSuggestions = q
    ? suggestions.filter((s) => s.title.toLowerCase().includes(q) || s.type.toLowerCase().includes(q)).slice(0, 8)
    : [];

  const handleSelectSuggestion = (item: Suggestion) => {
    setShowSearchResults(false);
    setSearchTerm('');
    item.go();
  };

  return (
    <header
      id="main-app-header"
      className="sticky top-16 h-16 bg-pc-bg/90 backdrop-blur-xl z-30 print:hidden flex items-center justify-between px-4 lg:px-6 shadow-[0_1px_8px_rgba(0,0,0,0.04)] border-b border-pc-line/40"
    >
      <div className="flex items-center gap-3 flex-1 max-w-xl" ref={searchRef}>
        <div className="relative w-full flex items-center">
          <span className="material-symbols-outlined absolute left-3 text-pc-muted text-[18px]">search</span>
          <input
            id="global-search-input"
            type="text"
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setShowSearchResults(true);
            }}
            onFocus={() => setShowSearchResults(true)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && filteredSuggestions[0]) handleSelectSuggestion(filteredSuggestions[0]);
              if (e.key === 'Escape') setShowSearchResults(false);
            }}
            placeholder="Search crops, stress conditions or sample images…"
            className="w-full h-10 pl-9 pr-4 bg-pc-card text-pc-ink placeholder-pc-muted font-body-sm text-[13px] rounded-lg border border-pc-outline/50 focus:outline-none focus:ring-2 focus:ring-pc-green/30 focus:border-pc-green transition-all shadow-xs"
          />

          {showSearchResults && filteredSuggestions.length > 0 && (
            <div className="absolute top-12 left-0 right-0 bg-pc-card rounded-xl shadow-xl border border-pc-line overflow-hidden z-50 p-1.5 animate-in">
              {filteredSuggestions.map((item) => (
                <button
                  key={item.type + item.title}
                  onClick={() => handleSelectSuggestion(item)}
                  className="w-full px-3 py-2 text-left rounded-lg hover:bg-pc-low transition-colors flex items-center justify-between gap-2 group"
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <span className="material-symbols-outlined text-[18px] text-pc-green">{item.icon}</span>
                    <span className="font-title-sm text-[13px] text-pc-ink truncate">{item.title}</span>
                  </span>
                  <span className="font-label-sm text-[10px] px-2 py-0.5 rounded bg-pc-cont text-pc-ink2 shrink-0">{item.type}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <button
        onClick={() => onNavigate('protocol')}
        title="Experiment protocol"
        className="hidden md:flex items-center gap-2 ml-4 px-3 py-1.5 rounded-full bg-pc-high text-pc-ink hover:bg-pc-line transition-colors border border-pc-line"
      >
        <span className="w-2 h-2 rounded-full bg-pc-green-bg" />
        <span className="font-code-sm text-[11px] text-pc-ink2">
          {RESEARCH.dataset.name} · {num(RESEARCH.dataset.accepted)} images · {RESEARCH.dataset.folds}-fold OOF
        </span>
      </button>
    </header>
  );
};
