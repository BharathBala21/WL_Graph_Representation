import React from 'react';
import { PageView, InspectionRecord } from '../types';
import { RECENT_INSPECTIONS, MONITORED_PLANTS } from '../data/mockData';

interface DashboardViewProps {
  onNavigate: (page: PageView) => void;
  onSelectInspection: (record: InspectionRecord) => void;
  onOpenBatchScan: () => void;
  onOpenLiveCamera: () => void;
  onExportCSV: () => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  onNavigate,
  onSelectInspection,
  onOpenBatchScan,
  onOpenLiveCamera,
  onExportCSV,
}) => {
  return (
    <div id="dashboard-view-container" className="flex flex-col gap-y-6 w-full max-w-[1440px] mx-auto animate-in fade-in duration-300">
      {/* Top Analytical Hero Card with Micro-Textures */}
      <div id="hero-analytical-card" className="relative overflow-hidden rounded-xl bg-white p-6 md:p-8 shadow-sm border border-[#dae2fd]/40">
        <div className="absolute -right-16 -top-16 w-80 h-80 rounded-full bg-[#006e2d]/5 blur-3xl pointer-events-none"></div>
        <div className="absolute right-1/3 bottom-0 w-64 h-48 rounded-full bg-[#7ffc97]/20 blur-2xl pointer-events-none"></div>

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="flex flex-col max-w-2xl">
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-[#7cf994] text-[#007230] font-label-sm text-[11px] uppercase tracking-wider font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-[#006e2d] animate-pulse"></span>
                Optical Bio-Screening Engine
              </span>
              <span className="font-code-sm text-[12px] text-[#717970]">v2.4.8-PROD</span>
            </div>

            <h1 className="font-headline-lg text-[28px] md:text-[34px] text-[#003b1b] tracking-tight leading-tight font-extrabold">
              Check Your Plant Health
            </h1>

            <p className="font-body-md text-[14px] text-[#404941] mt-2 leading-relaxed max-w-2xl">
              Upload or capture high-resolution imagery of plant foliar tissue to detect early morphological stresses, chlorosis patterns, and physiological water anomalies.
            </p>

            {/* Action Buttons */}
            <div className="flex flex-wrap items-center gap-3 mt-5">
              <button
                id="hero-analyze-btn"
                onClick={() => onNavigate('analyze-plant')}
                className="inline-flex items-center gap-2 bg-[#003b1b] text-white px-5 py-2.5 rounded-lg shadow-sm hover:bg-[#14532d] transition-all cursor-pointer font-title-sm text-title-sm group"
              >
                <span className="material-symbols-outlined text-[20px] transition-transform group-hover:scale-110">
                  photo_camera
                </span>
                <span>Analyze Plant</span>
              </button>

              <button
                id="hero-batch-scan-btn"
                onClick={onOpenBatchScan}
                className="inline-flex items-center gap-2 bg-[#f2f3ff] text-[#131b2e] hover:bg-[#e2e7ff] px-4 py-2.5 rounded-lg transition-all cursor-pointer font-title-sm text-title-sm border border-[#dae2fd]/60"
              >
                <span className="material-symbols-outlined text-[20px] text-[#717970]">upload_file</span>
                <span>Batch Scan</span>
              </button>

              <button
                id="hero-live-camera-btn"
                onClick={onOpenLiveCamera}
                className="inline-flex items-center gap-2 bg-[#f2f3ff] text-[#131b2e] hover:bg-[#e2e7ff] px-4 py-2.5 rounded-lg transition-all cursor-pointer font-title-sm text-title-sm border border-[#dae2fd]/60"
              >
                <span className="material-symbols-outlined text-[20px] text-[#006e2d]">videocam</span>
                <span>Take Live Photo</span>
              </button>
            </div>
          </div>

          {/* Quick Summary Spark Card */}
          <div className="flex flex-col gap-3 bg-[#f2f3ff] p-4 rounded-xl max-w-sm w-full border border-[#dae2fd]/60 shrink-0">
            <div className="flex items-center justify-between">
              <span className="font-label-sm text-[11px] uppercase text-[#717970] tracking-wider font-bold">
                Active Sensor Fleet
              </span>
              <span className="font-code-sm text-[11px] text-[#006e2d] font-bold">
                99.4% Latency: 320ms
              </span>
            </div>

            <div className="flex items-center gap-4">
              {/* Health Ratio Gauge */}
              <div className="relative w-14 h-14 flex items-center justify-center shrink-0">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                  <path
                    className="text-[#dae2fd] stroke-current"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    strokeWidth="3.5"
                  />
                  <path
                    className="text-[#006e2d] stroke-current"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    strokeDasharray="76, 100"
                    strokeLinecap="round"
                    strokeWidth="3.5"
                  />
                </svg>
                <span className="absolute font-code-sm text-[13px] font-bold text-[#003b1b]">76%</span>
              </div>

              <div className="flex flex-col">
                <span className="font-title-sm text-[13px] text-[#131b2e] font-bold">Canopy Health Ratio</span>
                <span className="font-body-sm text-[11px] text-[#717970] leading-snug">
                  108 of 142 scans normal. High hydration stability index across greenhouse blocks A-D.
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Key Stats Ribbon */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6 pt-5 border-t border-[#dae2fd]/40">
          <div className="flex flex-col bg-[#f2f3ff] p-3.5 rounded-lg border border-[#dae2fd]/40">
            <div className="flex items-center justify-between text-[#717970]">
              <span className="font-label-sm text-[10px] uppercase tracking-wider font-bold">
                Scans This Month
              </span>
              <span className="material-symbols-outlined text-[18px]">document_scanner</span>
            </div>
            <div className="flex items-baseline gap-1.5 mt-1.5">
              <span className="font-headline-md text-[24px] text-[#003b1b] font-extrabold">142</span>
              <span className="font-code-sm text-[12px] text-[#006e2d] font-bold">↑ +18%</span>
            </div>
            <span className="font-body-sm text-[11px] text-[#717970] mt-0.5">Across 4 facility quadrants</span>
          </div>

          <div className="flex flex-col bg-[#f2f3ff] p-3.5 rounded-lg border border-[#dae2fd]/40">
            <div className="flex items-center justify-between text-[#717970]">
              <span className="font-label-sm text-[10px] uppercase tracking-wider font-bold">
                Stress Incidents
              </span>
              <span className="material-symbols-outlined text-[18px] text-[#703a00]">warning</span>
            </div>
            <div className="flex items-baseline gap-1.5 mt-1.5">
              <span className="font-headline-md text-[24px] text-[#703a00] font-extrabold">24%</span>
              <span className="font-code-sm text-[12px] text-[#717970] font-medium">34 cases total</span>
            </div>
            <span className="font-body-sm text-[11px] text-[#717970] mt-0.5">Drought & micro-nutrient gaps</span>
          </div>

          <div className="flex flex-col bg-[#f2f3ff] p-3.5 rounded-lg border border-[#dae2fd]/40">
            <div className="flex items-center justify-between text-[#717970]">
              <span className="font-label-sm text-[10px] uppercase tracking-wider font-bold">
                Healthy Crops
              </span>
              <span className="material-symbols-outlined text-[18px] text-[#006e2d]">eco</span>
            </div>
            <div className="flex items-baseline gap-1.5 mt-1.5">
              <span className="font-headline-md text-[24px] text-[#006e2d] font-extrabold">76%</span>
              <span className="font-code-sm text-[12px] text-[#006e2d] font-semibold">Stable</span>
            </div>
            <span className="font-body-sm text-[11px] text-[#717970] mt-0.5">Chlorosis below threshold</span>
          </div>

          <div className="flex flex-col bg-[#f2f3ff] p-3.5 rounded-lg border border-[#dae2fd]/40">
            <div className="flex items-center justify-between text-[#717970]">
              <span className="font-label-sm text-[10px] uppercase tracking-wider font-bold">
                Active Profiles
              </span>
              <span className="material-symbols-outlined text-[18px]">psychiatry</span>
            </div>
            <div className="flex items-baseline gap-1.5 mt-1.5">
              <span className="font-headline-md text-[24px] text-[#003b1b] font-extrabold">18</span>
              <span className="font-code-sm text-[12px] text-[#717970] font-medium">Species mapped</span>
            </div>
            <span className="font-body-sm text-[11px] text-[#717970] mt-0.5">Automated telemetry synced</span>
          </div>
        </div>
      </div>

      {/* Diagnostic Grid (2 Columns Layout) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* COLUMN 1 (7 Cols): Feed & Academic Notice */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* Recent Visual Inspections */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-[#006e2d]"></span>
                  <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
                    Recent Visual Inspections
                  </h2>
                </div>
                <span className="font-body-sm text-[12px] text-[#717970]">
                  Real-time inference queue evaluated by dual CNN-ViT architecture
                </span>
              </div>

              <button
                id="export-csv-btn"
                onClick={onExportCSV}
                className="font-label-sm text-[11px] font-bold uppercase tracking-wider text-[#006e2d] hover:text-[#003b1b] transition-colors cursor-pointer flex items-center gap-1 px-2.5 py-1 rounded bg-[#eaedff] hover:bg-[#dae2fd]"
              >
                <span>Export CSV</span>
                <span className="material-symbols-outlined text-[16px]">file_download</span>
              </button>
            </div>

            {/* Analysis Cards Stack */}
            <div className="flex flex-col gap-3">
              {RECENT_INSPECTIONS.slice(0, 3).map((record) => (
                <div
                  key={record.id}
                  id={`inspection-card-${record.id}`}
                  onClick={() => onSelectInspection(record)}
                  className="bg-[#f2f3ff] hover:bg-[#eaedff] transition-all duration-200 rounded-xl p-3 sm:p-4 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between cursor-pointer group border border-[#dae2fd]/40 hover:border-[#006e2d]/30 hover:shadow-sm"
                >
                  <div className="flex items-center gap-3.5">
                    <div className="relative w-20 h-20 rounded-lg overflow-hidden shrink-0 bg-[#dae2fd]">
                      <img
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        src={record.imageUrl}
                        alt={record.specimenName}
                      />
                      <span className="absolute bottom-1 right-1 px-1 rounded bg-black/60 text-white font-code-sm text-[10px]">
                        {record.tagId}
                      </span>
                    </div>

                    <div className="flex flex-col">
                      <div className="flex items-center gap-2">
                        <span className="font-title-md text-[15px] text-[#131b2e] font-bold">
                          {record.specimenName}
                        </span>
                        <span className="italic font-body-sm text-[12px] text-[#717970]">
                          ({record.scientificName})
                        </span>
                      </div>

                      <div className="flex items-center gap-2 mt-1">
                        <span
                          className={`px-2 py-0.5 rounded-full font-label-sm text-[11px] font-semibold flex items-center gap-1 ${
                            record.statusType === 'water-stress'
                              ? 'bg-[#ffdcc3] text-[#2f1500]'
                              : record.statusType === 'healthy'
                              ? 'bg-[#7cf994] text-[#007230]'
                              : 'bg-[#dae2fd] text-[#404941]'
                          }`}
                        >
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${
                              record.statusType === 'water-stress'
                                ? 'bg-[#4f2700]'
                                : record.statusType === 'healthy'
                                ? 'bg-[#006e2d]'
                                : 'bg-[#717970]'
                            }`}
                          ></span>
                          {record.statusText}
                        </span>
                      </div>

                      <span className="font-body-sm text-[12px] text-[#717970] mt-1 line-clamp-1">
                        {record.symptomSummary}
                      </span>
                    </div>
                  </div>

                  <div className="flex sm:flex-col items-end justify-between w-full sm:w-auto shrink-0 gap-1 border-t sm:border-t-0 pt-2 sm:pt-0 border-[#dae2fd]">
                    <div className="flex items-center gap-1.5">
                      <span
                        className={`font-code-sm text-[12px] font-bold ${
                          record.statusType === 'water-stress'
                            ? 'text-[#703a00]'
                            : record.statusType === 'healthy'
                            ? 'text-[#006e2d]'
                            : 'text-[#717970]'
                        }`}
                      >
                        {record.confidence}% Conf.
                      </span>
                      <div className="w-16 h-1.5 bg-[#dae2fd] rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            record.statusType === 'water-stress'
                              ? 'bg-[#703a00]'
                              : record.statusType === 'healthy'
                              ? 'bg-[#006e2d]'
                              : 'bg-[#717970]'
                          }`}
                          style={{ width: `${record.confidence}%` }}
                        ></div>
                      </div>
                    </div>
                    <span className="font-code-sm text-[11px] text-[#717970]">{record.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Footer Button */}
            <button
              id="view-all-history-btn"
              onClick={() => onNavigate('history-logs')}
              className="w-full py-2.5 rounded-lg bg-[#eaedff] hover:bg-[#dae2fd] transition-colors font-title-sm text-title-sm text-[#003b1b] font-bold flex items-center justify-center gap-2 cursor-pointer"
            >
              <span>View All Analysis History</span>
              <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
            </button>
          </div>

          {/* Academic ML Diagnostic Notice (ISO/IEC 2382 Screening) */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 relative overflow-hidden">
            <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-[#006e2d]"></div>
            <div className="flex items-start gap-4 pl-1">
              <div className="p-2.5 rounded-lg bg-[#7cf994] text-[#007230] shrink-0 mt-0.5">
                <span className="material-symbols-outlined text-[22px]">biotech</span>
              </div>
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-2">
                  <span className="font-title-sm text-title-sm text-[#131b2e] font-bold">
                    Academic ML Diagnostic Notice
                  </span>
                  <span className="font-label-sm text-[10px] px-2 py-0.5 rounded bg-[#eaedff] font-code-sm text-[#717970]">
                    ISO/IEC 2382 Screening
                  </span>
                </div>
                <p className="font-body-sm text-[12px] text-[#404941] leading-relaxed">
                  PlantCare AI is an AI-assisted visual assessment platform designed for agronomic screening. Predictions represent probabilistic indicators based on visual symptomatology, not certified laboratory diagnostics. Always cross-reference high-stress predictions with leaf tissue laboratory sap tests, soil moisture tensiometers, or professional agronomist evaluations.
                </p>
                <div className="flex flex-wrap items-center gap-4 mt-1 text-[#717970] font-label-sm text-[11px]">
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[14px] text-[#006e2d]">verified</span>
                    Model Card ResNet-ViT v2.4
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[14px]">lock</span>
                    Data Encrypted AES-256
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Real-time Spectral Band Distribution (NDVI / VARI) */}
          <div className="bg-white rounded-xl p-5 shadow-sm border border-[#dae2fd]/40 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="font-title-sm text-title-sm text-[#131b2e] font-bold">
                Canopy Spectral Band Indices (NDVI / VARI)
              </span>
              <span className="font-label-sm text-[10px] text-[#717970] font-code-sm tracking-wider font-bold">
                LIVE SENSOR FEED
              </span>
            </div>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-[#f2f3ff] p-3 rounded-lg flex flex-col border border-[#dae2fd]/40">
                <span className="font-code-sm text-[11px] text-[#717970]">NDVI MEAN</span>
                <span className="font-headline-sm text-[22px] text-[#006e2d] font-extrabold mt-0.5">0.782</span>
                <span className="font-label-sm text-[11px] text-[#006e2d] font-semibold">High Vigour</span>
              </div>

              <div className="bg-[#f2f3ff] p-3 rounded-lg flex flex-col border border-[#dae2fd]/40">
                <span className="font-code-sm text-[11px] text-[#717970]">VARI INDEX</span>
                <span className="font-headline-sm text-[22px] text-[#003b1b] font-extrabold mt-0.5">0.614</span>
                <span className="font-label-sm text-[11px] text-[#003b1b] font-semibold">Normal Chlorophyll</span>
              </div>

              <div className="bg-[#f2f3ff] p-3 rounded-lg flex flex-col border border-[#dae2fd]/40">
                <span className="font-code-sm text-[11px] text-[#717970]">THERMAL STRESS</span>
                <span className="font-headline-sm text-[22px] text-[#703a00] font-extrabold mt-0.5">26.8°C</span>
                <span className="font-label-sm text-[11px] text-[#703a00] font-semibold">Target +/- 2.1°C</span>
              </div>
            </div>
          </div>
        </div>

        {/* COLUMN 2 (5 Cols): Monitored Plants & Education & AI Workflow */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* My Monitored Plants Preview */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[#003b1b] text-[20px]">potted_plant</span>
                <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
                  My Monitored Plants
                </h2>
              </div>
              <button
                onClick={() => onNavigate('my-plants-health-tracking')}
                className="font-label-sm text-[11px] px-2.5 py-1 rounded-full bg-[#eaedff] text-[#404941] hover:bg-[#dae2fd] font-semibold cursor-pointer"
              >
                3 of 18 showing
              </button>
            </div>

            <div className="flex flex-col gap-2.5">
              {MONITORED_PLANTS.slice(0, 3).map((plant) => (
                <div
                  key={plant.id}
                  onClick={() => onNavigate('my-plants-health-tracking')}
                  className="bg-[#f2f3ff] hover:bg-[#eaedff] transition-colors p-3 rounded-lg flex items-center justify-between cursor-pointer border border-[#dae2fd]/40"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-10 h-10 rounded-full flex items-center justify-center font-bold shrink-0 ${
                        plant.status === 'Optimal'
                          ? 'bg-[#7cf994] text-[#007230]'
                          : plant.status.includes('Water') || plant.status.includes('Moderate')
                          ? 'bg-[#ffdcc3] text-[#4f2700]'
                          : 'bg-[#dae2fd] text-[#404941]'
                      }`}
                    >
                      <span className="material-symbols-outlined text-[20px]">{plant.icon}</span>
                    </div>

                    <div className="flex flex-col">
                      <span className="font-title-sm text-[13px] text-[#131b2e] font-bold leading-tight">
                        {plant.name}
                      </span>
                      <span className="font-body-sm text-[11px] text-[#717970] italic">
                        {plant.scientificName}
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-col items-end">
                    <span
                      className={`px-2 py-0.5 rounded-full font-label-sm text-[10px] font-bold ${
                        plant.status === 'Optimal'
                          ? 'bg-[#7cf994] text-[#007230]'
                          : plant.status.includes('Moderate')
                          ? 'bg-[#ffdcc3] text-[#4f2700]'
                          : 'bg-[#dae2fd] text-[#404941]'
                      }`}
                    >
                      {plant.status}
                    </span>
                    <span
                      className={`font-code-sm text-[11px] font-semibold mt-1 flex items-center gap-0.5 ${
                        plant.statusTrend === 'Improving'
                          ? 'text-[#006e2d]'
                          : plant.statusTrend === 'Peak'
                          ? 'text-[#703a00]'
                          : 'text-[#717970]'
                      }`}
                    >
                      <span className="material-symbols-outlined text-[14px]">{plant.trendIcon}</span>
                      {plant.statusTrend}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Stress Library Quick Guides */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[#006e2d] text-[20px]">menu_book</span>
                <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
                  Stress Library Quick Guides
                </h2>
              </div>
              <button
                onClick={() => onNavigate('plant-stress-library')}
                className="font-label-sm text-[11px] text-[#006e2d] font-bold hover:underline cursor-pointer"
              >
                Explore 80+
              </button>
            </div>

            <div className="flex flex-col gap-2">
              <button
                onClick={() => onNavigate('plant-stress-library')}
                className="p-3 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] flex items-center justify-between transition-colors group cursor-pointer border border-[#dae2fd]/40 text-left"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-[#eaedff] flex items-center justify-center text-[#003b1b] shrink-0">
                    <span className="material-symbols-outlined text-[18px]">water_drop</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-title-sm text-[13px] text-[#131b2e] group-hover:text-[#003b1b] transition-colors font-bold">
                      Water Stress Diagnosis
                    </span>
                    <span className="font-body-sm text-[11px] text-[#717970]">
                      Drought wilting vs. Root waterlogging hypoxia
                    </span>
                  </div>
                </div>
                <span className="material-symbols-outlined text-[#717970] text-[18px] group-hover:translate-x-0.5 transition-transform">
                  chevron_right
                </span>
              </button>

              <button
                onClick={() => onNavigate('plant-stress-library')}
                className="p-3 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] flex items-center justify-between transition-colors group cursor-pointer border border-[#dae2fd]/40 text-left"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-[#eaedff] flex items-center justify-center text-[#703a00] shrink-0">
                    <span className="material-symbols-outlined text-[18px]">grain</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-title-sm text-[13px] text-[#131b2e] group-hover:text-[#003b1b] transition-colors font-bold">
                      Chlorosis & Deficiencies
                    </span>
                    <span className="font-body-sm text-[11px] text-[#717970]">
                      Symptom maps for N, P, K, Fe, and Mg shortfalls
                    </span>
                  </div>
                </div>
                <span className="material-symbols-outlined text-[#717970] text-[18px] group-hover:translate-x-0.5 transition-transform">
                  chevron_right
                </span>
              </button>

              <button
                onClick={() => onNavigate('plant-stress-library')}
                className="p-3 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] flex items-center justify-between transition-colors group cursor-pointer border border-[#dae2fd]/40 text-left"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-[#eaedff] flex items-center justify-center text-[#ba1a1a] shrink-0">
                    <span className="material-symbols-outlined text-[18px]">device_thermostat</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-title-sm text-[13px] text-[#131b2e] group-hover:text-[#003b1b] transition-colors font-bold">
                      Temperature Extremes
                    </span>
                    <span className="font-body-sm text-[11px] text-[#717970]">
                      Sunscald, scorching, and frost tissue necrosis
                    </span>
                  </div>
                </div>
                <span className="material-symbols-outlined text-[#717970] text-[18px] group-hover:translate-x-0.5 transition-transform">
                  chevron_right
                </span>
              </button>
            </div>
          </div>

          {/* How It Works: 5-Step AI Workflow */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[#003b1b] text-[20px]">account_tree</span>
                <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
                  How It Works (5-Step AI Flow)
                </h2>
              </div>
              <span className="font-code-sm text-[11px] text-[#006e2d] font-bold px-2 py-0.5 rounded bg-[#7cf994]/50">
                Autonomous
              </span>
            </div>

            <div className="relative flex flex-col gap-3 pl-4">
              {/* Vertical connecting line */}
              <div className="absolute left-[26px] top-3 bottom-3 w-0.5 bg-[#dae2fd]"></div>

              {/* Step 1 */}
              <div className="relative flex items-start gap-3 z-10">
                <div className="w-5 h-5 rounded-full bg-[#003b1b] text-white flex items-center justify-center text-[10px] font-bold shrink-0 ring-4 ring-white">
                  1
                </div>
                <div className="flex flex-col">
                  <span className="font-title-sm text-[13px] text-[#131b2e] font-bold">Upload Photo</span>
                  <span className="font-body-sm text-[11px] text-[#717970]">
                    RGB foliar capture under daylight or balanced white LED.
                  </span>
                </div>
              </div>

              {/* Step 2 */}
              <div className="relative flex items-start gap-3 z-10">
                <div className="w-5 h-5 rounded-full bg-[#003b1b] text-white flex items-center justify-center text-[10px] font-bold shrink-0 ring-4 ring-white">
                  2
                </div>
                <div className="flex flex-col">
                  <span className="font-title-sm text-[13px] text-[#131b2e] font-bold">
                    Preprocessing & Segmentation
                  </span>
                  <span className="font-body-sm text-[11px] text-[#717970]">
                    Background soil/tray removal via Otsu & U-Net mask.
                  </span>
                </div>
              </div>

              {/* Step 3 */}
              <div className="relative flex items-start gap-3 z-10">
                <div className="w-5 h-5 rounded-full bg-[#003b1b] text-white flex items-center justify-center text-[10px] font-bold shrink-0 ring-4 ring-white">
                  3
                </div>
                <div className="flex flex-col">
                  <span className="font-title-sm text-[13px] text-[#131b2e] font-bold">
                    CNN / ViT Feature Extraction
                  </span>
                  <span className="font-body-sm text-[11px] text-[#717970]">
                    Self-attention maps highlighting necrotic vein patterns.
                  </span>
                </div>
              </div>

              {/* Step 4 */}
              <div className="relative flex items-start gap-3 z-10">
                <div className="w-5 h-5 rounded-full bg-[#006e2d] text-white flex items-center justify-center text-[10px] font-bold shrink-0 ring-4 ring-white">
                  4
                </div>
                <div className="flex flex-col">
                  <span className="font-title-sm text-[13px] text-[#131b2e] font-bold">
                    Multi-Label Stress Classification
                  </span>
                  <span className="font-body-sm text-[11px] text-[#717970]">
                    Biotic pathogens vs abiotic nutrient & water deficiency.
                  </span>
                </div>
              </div>

              {/* Step 5 */}
              <div className="relative flex items-start gap-3 z-10">
                <div className="w-5 h-5 rounded-full bg-[#dae2fd] text-[#131b2e] flex items-center justify-center text-[10px] font-bold shrink-0 ring-4 ring-white">
                  5
                </div>
                <div className="flex flex-col">
                  <span className="font-title-sm text-[13px] text-[#131b2e] font-bold">
                    Evidence-Based Remediation Advice
                  </span>
                  <span className="font-body-sm text-[11px] text-[#717970]">
                    Actionable irrigation, fertilizer dosing, or quarantine guidelines.
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
