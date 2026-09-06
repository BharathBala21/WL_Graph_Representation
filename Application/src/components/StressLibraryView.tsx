import React, { useState } from 'react';
import { PageView, StressGuideItem } from '../types';
import { STRESS_GUIDES } from '../data/mockData';

interface StressLibraryViewProps {
  onNavigate: (page: PageView) => void;
  onScanStress: (guide: StressGuideItem) => void;
}

export const StressLibraryView: React.FC<StressLibraryViewProps> = ({ onNavigate, onScanStress }) => {
  const [activeCategory, setActiveCategory] = useState<string>('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedGuide, setSelectedGuide] = useState<StressGuideItem | null>(null);

  const categories = ['All', 'Water Stress', 'Nutrient Deficiency', 'Pathogen / Disease', 'Thermal Extremes'];

  const filteredGuides = STRESS_GUIDES.filter((guide) => {
    const matchesCategory = activeCategory === 'All' || guide.category === activeCategory;
    const matchesSearch =
      guide.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      guide.scientificTaxa.toLowerCase().includes(searchQuery.toLowerCase()) ||
      guide.visualIndicators.toLowerCase().includes(searchQuery.toLowerCase()) ||
      guide.affectedCrops.some((c) => c.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesSearch;
  });

  return (
    <div id="stress-library-view" className="max-w-[1440px] w-full mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Header Banner */}
      <div className="bg-white rounded-xl p-6 md:p-8 shadow-sm border border-[#dae2fd]/40 relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 relative z-10">
          <div className="space-y-2 max-w-2xl">
            <div className="flex items-center gap-2">
              <span className="font-label-sm text-[11px] px-2.5 py-0.5 rounded-full bg-[#7cf994] text-[#007230] font-bold uppercase tracking-wider">
                Agronomic Field Atlas
              </span>
              <span className="font-code-sm text-[12px] text-[#717970]">80+ Stresses Documented</span>
            </div>
            <h1 className="font-headline-lg text-[28px] md:text-[32px] text-[#003b1b] font-extrabold tracking-tight">
              Plant Stress & Pathology Library
            </h1>
            <p className="font-body-md text-[14px] text-[#404941]">
              Browse validated photographic profiles, physiological etiology, and agronomic management plans for abiotic stresses, nutrient deficiencies, and biotic fungal pathogens.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <button
              onClick={() => onNavigate('analyze-plant')}
              className="px-5 py-2.5 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[13px] font-bold shadow-sm flex items-center justify-center gap-2 cursor-pointer transition-all"
            >
              <span className="material-symbols-outlined text-[18px]">photo_camera</span>
              <span>Upload New Scan</span>
            </button>
          </div>
        </div>

        {/* Search & Category Filter Bar */}
        <div className="mt-6 pt-6 border-t border-[#dae2fd]/40 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
          {/* Category Chips */}
          <div className="flex flex-wrap items-center gap-2">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                className={`px-3.5 py-1.5 rounded-full font-label-sm text-[12px] font-bold transition-all cursor-pointer ${
                  activeCategory === cat
                    ? 'bg-[#14532d] text-white shadow-xs'
                    : 'bg-[#f2f3ff] text-[#404941] hover:bg-[#eaedff] border border-[#dae2fd]/60'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Local Search */}
          <div className="relative w-full md:w-72">
            <span className="material-symbols-outlined absolute left-3 top-2.5 text-[#717970] text-[18px]">
              search
            </span>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search library..."
              className="w-full h-9 pl-9 pr-4 bg-[#f2f3ff] rounded-lg text-[#131b2e] font-body-sm text-[12px] border border-[#dae2fd] focus:outline-none focus:bg-white focus:ring-2 focus:ring-[#006e2d]/30"
            />
          </div>
        </div>
      </div>

      {/* Guide Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredGuides.map((guide) => (
          <div
            key={guide.id}
            id={`guide-${guide.id}`}
            className="bg-white rounded-xl overflow-hidden shadow-sm border border-[#dae2fd]/40 flex flex-col justify-between hover:shadow-md hover:border-[#006e2d]/40 transition-all group"
          >
            <div>
              {/* Image Banner */}
              <div className="relative aspect-[16/10] bg-[#dae2fd] overflow-hidden">
                <img
                  src={guide.imageUrl}
                  alt={guide.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute top-2 left-2 flex gap-1.5">
                  <span
                    className={`font-label-sm text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${
                      guide.type === 'Abiotic'
                        ? 'bg-[#ffdcc3] text-[#2f1500]'
                        : 'bg-[#ffdad6] text-[#93000a]'
                    }`}
                  >
                    {guide.type}
                  </span>
                  <span className="font-label-sm text-[10px] px-2 py-0.5 rounded-full bg-black/60 text-white font-bold">
                    {guide.category}
                  </span>
                </div>
              </div>

              {/* Body */}
              <div className="p-5 space-y-3">
                <div>
                  <h3 className="font-headline-sm text-[18px] text-[#131b2e] font-bold group-hover:text-[#003b1b] transition-colors leading-snug">
                    {guide.title}
                  </h3>
                  <p className="font-body-sm text-[11px] text-[#717970] italic">
                    {guide.scientificTaxa}
                  </p>
                </div>

                <p className="font-body-sm text-[12px] text-[#404941] leading-relaxed">
                  {guide.visualIndicators}
                </p>

                {/* Key Symptoms */}
                <div className="space-y-1 pt-1">
                  <span className="font-label-sm text-[10px] font-bold uppercase tracking-wider text-[#717970]">
                    Primary Symptoms
                  </span>
                  <ul className="space-y-1">
                    {guide.symptoms.slice(0, 3).map((sym, idx) => (
                      <li key={idx} className="flex items-start gap-1.5 font-body-sm text-[11px] text-[#131b2e]">
                        <span className="material-symbols-outlined text-[14px] text-[#006e2d] mt-0.5">
                          check_circle
                        </span>
                        <span>{sym}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            {/* Card Footer Actions */}
            <div className="p-4 pt-0 flex items-center justify-between border-t border-[#dae2fd]/40 mt-3 pt-3">
              <button
                onClick={() => setSelectedGuide(guide)}
                className="font-title-sm text-[12px] text-[#006e2d] font-bold hover:underline cursor-pointer flex items-center gap-1"
              >
                <span>Full Protocol</span>
                <span className="material-symbols-outlined text-[16px]">visibility</span>
              </button>

              <button
                onClick={() => onScanStress(guide)}
                className="px-3 py-1.5 rounded-lg bg-[#eaedff] hover:bg-[#dae2fd] text-[#003b1b] font-title-sm text-[11px] font-bold transition-colors cursor-pointer flex items-center gap-1"
              >
                <span className="material-symbols-outlined text-[14px]">photo_camera</span>
                <span>Scan Specimen</span>
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Detailed Modal for Selected Guide */}
      {selectedGuide && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full p-6 space-y-5 shadow-2xl max-h-[90vh] overflow-y-auto animate-in fade-in">
            <div className="flex items-start justify-between">
              <div>
                <span className="font-label-sm text-[11px] px-2.5 py-0.5 rounded-full bg-[#7cf994] text-[#007230] font-bold">
                  {selectedGuide.category}
                </span>
                <h2 className="font-headline-md text-[22px] text-[#131b2e] font-extrabold mt-1">
                  {selectedGuide.title}
                </h2>
                <span className="font-body-sm text-[12px] text-[#717970] italic">
                  {selectedGuide.scientificTaxa}
                </span>
              </div>
              <button
                onClick={() => setSelectedGuide(null)}
                className="p-1 rounded-full text-[#717970] hover:text-[#131b2e] hover:bg-[#eaedff] cursor-pointer"
              >
                <span className="material-symbols-outlined text-[24px]">close</span>
              </button>
            </div>

            <div className="w-full aspect-[16/9] rounded-xl overflow-hidden bg-[#dae2fd]">
              <img
                src={selectedGuide.imageUrl}
                alt={selectedGuide.title}
                className="w-full h-full object-cover"
              />
            </div>

            <div className="space-y-4">
              <div>
                <h3 className="font-title-sm text-[13px] font-bold text-[#131b2e] uppercase tracking-wider">
                  Visual Diagnostics & Indicators
                </h3>
                <p className="font-body-sm text-[13px] text-[#404941] mt-1 leading-relaxed">
                  {selectedGuide.visualIndicators}
                </p>
              </div>

              <div>
                <h3 className="font-title-sm text-[13px] font-bold text-[#131b2e] uppercase tracking-wider">
                  Underlying Agronomic Causes
                </h3>
                <ul className="mt-1 space-y-1">
                  {selectedGuide.causes.map((c, i) => (
                    <li key={i} className="flex items-center gap-2 font-body-sm text-[12px] text-[#404941]">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#ba1a1a]"></span>
                      <span>{c}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="font-title-sm text-[13px] font-bold text-[#131b2e] uppercase tracking-wider">
                  Recommended Treatment & Mitigation Protocols
                </h3>
                <ul className="mt-1 space-y-1.5">
                  {selectedGuide.remediation.map((r, i) => (
                    <li key={i} className="flex items-start gap-2 font-body-sm text-[12px] text-[#131b2e] bg-[#f2f3ff] p-2.5 rounded-lg border border-[#dae2fd]/40">
                      <span className="material-symbols-outlined text-[16px] text-[#006e2d] mt-0.5">
                        task_alt
                      </span>
                      <span>{r}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="font-title-sm text-[13px] font-bold text-[#131b2e] uppercase tracking-wider">
                  Susceptible Host Crops
                </h3>
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {selectedGuide.affectedCrops.map((crop, i) => (
                    <span
                      key={i}
                      className="font-code-sm text-[11px] px-2.5 py-1 rounded bg-[#eaedff] text-[#003b1b] font-medium"
                    >
                      {crop}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-[#dae2fd] flex justify-end gap-3">
              <button
                onClick={() => setSelectedGuide(null)}
                className="px-4 py-2 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] text-[#131b2e] font-title-sm text-[12px] font-bold cursor-pointer"
              >
                Close
              </button>
              <button
                onClick={() => {
                  const guide = selectedGuide;
                  setSelectedGuide(null);
                  onScanStress(guide);
                }}
                className="px-5 py-2 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[12px] font-bold cursor-pointer"
              >
                Launch Diagnostic Scan
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
