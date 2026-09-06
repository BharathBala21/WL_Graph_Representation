import React, { useState } from 'react';
import { PageView, InspectionRecord } from '../types';
import {
  DIAGNOSTIC_PINS,
  STRESS_PREDICTIONS,
  OBSERVED_SYMPTOMS,
  ROOT_CAUSES,
  REMEDIATION_STEPS,
  ML_PIPELINE_STAGES,
} from '../data/mockData';

interface DiagnosticsViewProps {
  onNavigate: (page: PageView) => void;
  activeInspection?: InspectionRecord | null;
  onSaveToMyPlants: () => void;
  onOpenCompareScans: () => void;
  onOpenShareModal: () => void;
  onExportPDF: () => void;
}

export const DiagnosticsView: React.FC<DiagnosticsViewProps> = ({
  onNavigate,
  activeInspection,
  onSaveToMyPlants,
  onOpenCompareScans,
  onOpenShareModal,
  onExportPDF,
}) => {
  const [selectedLayer, setSelectedLayer] = useState<'raw' | 'heatmap' | 'pathology'>('heatmap');
  const [activePinId, setActivePinId] = useState<number | null>(1);
  const [savedSuccess, setSavedSuccess] = useState(false);

  const specimenId = activeInspection?.id || 'PCAI-98421';
  const specimenName = activeInspection?.specimenName || 'Field Tomato #04';
  const scientificName = activeInspection?.scientificName || 'Solanum lycopersicum';
  const timestamp = activeInspection?.dateStr || 'Oct 24, 2024 (14:32 PST)';
  const imageSrc =
    activeInspection?.imageUrl ||
    'https://lh3.googleusercontent.com/aida-public/AB6AXuBMiDgfnnLYcUUsdxm7Exw-SA6swXGrlE_dD_I073_e0JhtESnSIAixpEEScG0-R5_EZV1qDKXOg8zEC4LGYOoPYSqgY8waadfiUKD3PYFT_3Ydwua8Nlir_htUuEgJt-RDyYdeV-n3eFa4SefXDprGJ_tpO858GG_YnHGUyxNX_ltacFHKMNmdim3SaL9B_YWO4gO40hxtKKvoxv1LYSYvZqtV2n4YBAfDSLsIBGlHxXBc0A9B56cR';

  const handleSave = () => {
    onSaveToMyPlants();
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 3000);
  };

  return (
    <div id="diagnostics-results-view" className="max-w-[1440px] w-full mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Top Metadata Header Bar */}
      <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-code-sm text-[12px] px-2.5 py-0.5 rounded-full bg-[#f2f3ff] text-[#003b1b] font-bold border border-[#dae2fd]">
              SPECIMEN ID: {specimenId}
            </span>
            <span className="font-label-sm text-[11px] px-2 py-0.5 rounded bg-[#ffdcc3] text-[#2f1500] font-bold">
              MODERATE WATER STRESS
            </span>
            <span className="font-code-sm text-[11px] text-[#717970]">
              Confidence: {activeInspection?.confidence || 82}%
            </span>
          </div>

          <h1 className="font-headline-md text-[24px] md:text-[28px] text-[#131b2e] font-extrabold tracking-tight">
            Diagnostic Assessment Report: {specimenName}
          </h1>

          <div className="flex flex-wrap items-center gap-4 text-[#717970] font-body-sm text-[12px]">
            <span className="flex items-center gap-1">
              <span className="material-symbols-outlined text-[15px] text-[#006e2d]">psychiatry</span>
              <span className="italic">{scientificName}</span>
            </span>
            <span>•</span>
            <span className="flex items-center gap-1">
              <span className="material-symbols-outlined text-[15px]">schedule</span>
              <span>{timestamp}</span>
            </span>
            <span>•</span>
            <span className="flex items-center gap-1">
              <span className="material-symbols-outlined text-[15px]">neurology</span>
              <span>ResNet-ViT v2.4 (Ensemble)</span>
            </span>
          </div>
        </div>

        {/* Action Buttons Toolbar */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            id="save-to-my-plants-btn"
            onClick={handleSave}
            className={`px-3.5 py-2 rounded-lg font-title-sm text-[12px] font-bold transition-all flex items-center gap-1.5 cursor-pointer shadow-xs ${
              savedSuccess
                ? 'bg-[#7cf994] text-[#007230]'
                : 'bg-[#f2f3ff] hover:bg-[#eaedff] text-[#131b2e] border border-[#dae2fd]'
            }`}
          >
            <span className="material-symbols-outlined text-[16px]">
              {savedSuccess ? 'check' : 'bookmark_add'}
            </span>
            <span>{savedSuccess ? 'Saved to Profile' : 'Save to My Plants'}</span>
          </button>

          <button
            id="compare-scans-btn"
            onClick={onOpenCompareScans}
            className="px-3.5 py-2 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] text-[#131b2e] font-title-sm text-[12px] font-bold transition-colors flex items-center gap-1.5 border border-[#dae2fd] cursor-pointer"
          >
            <span className="material-symbols-outlined text-[16px]">compare</span>
            <span>Compare Scans</span>
          </button>

          <button
            id="share-agronomist-btn"
            onClick={onOpenShareModal}
            className="px-3.5 py-2 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] text-[#131b2e] font-title-sm text-[12px] font-bold transition-colors flex items-center gap-1.5 border border-[#dae2fd] cursor-pointer"
          >
            <span className="material-symbols-outlined text-[16px]">share</span>
            <span>Share with Agronomist</span>
          </button>

          <button
            id="export-pdf-btn"
            onClick={onExportPDF}
            className="px-4 py-2 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[12px] font-bold transition-all flex items-center gap-1.5 shadow-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[16px]">picture_as_pdf</span>
            <span>Export PDF</span>
          </button>
        </div>
      </div>

      {/* Main Diagnostic Workspace (2 Column Grid) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column (7 Cols): Visual Map & Ranked Diagnoses */}
        <div className="lg:col-span-7 space-y-6">
          {/* Leaf Visual Map Card */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
                  Interactive Diagnostic Map
                </h2>
                <span className="font-body-sm text-[12px] text-[#717970]">
                  Grad-CAM Attention Heatmap & Pathology Anomaly Markers
                </span>
              </div>

              {/* Layer Switcher Buttons */}
              <div className="inline-flex p-1 bg-[#f2f3ff] rounded-lg border border-[#dae2fd]">
                <button
                  onClick={() => setSelectedLayer('raw')}
                  className={`px-3 py-1 rounded-md font-label-sm text-[11px] font-bold transition-all cursor-pointer ${
                    selectedLayer === 'raw'
                      ? 'bg-white text-[#131b2e] shadow-xs'
                      : 'text-[#404941] hover:text-[#131b2e]'
                  }`}
                >
                  Raw Image
                </button>
                <button
                  onClick={() => setSelectedLayer('heatmap')}
                  className={`px-3 py-1 rounded-md font-label-sm text-[11px] font-bold transition-all cursor-pointer ${
                    selectedLayer === 'heatmap'
                      ? 'bg-[#14532d] text-white shadow-xs'
                      : 'text-[#404941] hover:text-[#131b2e]'
                  }`}
                >
                  CAM Heatmap
                </button>
                <button
                  onClick={() => setSelectedLayer('pathology')}
                  className={`px-3 py-1 rounded-md font-label-sm text-[11px] font-bold transition-all cursor-pointer ${
                    selectedLayer === 'pathology'
                      ? 'bg-[#003b1b] text-white shadow-xs'
                      : 'text-[#404941] hover:text-[#131b2e]'
                  }`}
                >
                  Pathology Mask
                </button>
              </div>
            </div>

            {/* Interactive Leaf Stage Viewport */}
            <div className="relative w-full aspect-[4/3] rounded-xl overflow-hidden bg-[#283044] select-none border border-[#dae2fd]">
              {/* Base Image */}
              <img
                src={imageSrc}
                alt="Leaf Diagnostic"
                className="w-full h-full object-cover"
              />

              {/* Layer 2: CAM Heatmap Overlay (simulated via glowing color overlay) */}
              {selectedLayer === 'heatmap' && (
                <div className="absolute inset-0 pointer-events-none mix-blend-color-dodge opacity-75">
                  <div className="absolute top-[20%] right-[25%] w-48 h-48 rounded-full bg-gradient-to-br from-amber-400 via-rose-500 to-transparent blur-2xl"></div>
                  <div className="absolute bottom-[20%] right-[30%] w-40 h-40 rounded-full bg-gradient-to-tr from-emerald-400 via-amber-300 to-transparent blur-xl"></div>
                </div>
              )}

              {/* Layer 3: Pathology Boundary Mask */}
              {selectedLayer === 'pathology' && (
                <div className="absolute inset-0 pointer-events-none border-4 border-dashed border-[#7cf994]/60 m-4 rounded-lg">
                  <div className="absolute top-2 left-2 bg-black/70 text-white font-code-sm text-[10px] px-2 py-1 rounded">
                    Segmentation: Masked ROI [Otsu Threshold = 0.42]
                  </div>
                </div>
              )}

              {/* Interactive Diagnostic Pins (1, 2, 3) */}
              {DIAGNOSTIC_PINS.map((pin) => {
                const isSelected = activePinId === pin.id;
                return (
                  <div
                    key={pin.id}
                    style={{ top: pin.topPct, right: pin.rightPct }}
                    className="absolute z-20 group"
                  >
                    <button
                      onClick={() => setActivePinId(isSelected ? null : pin.id)}
                      className={`w-7 h-7 rounded-full font-bold text-[12px] flex items-center justify-center shadow-lg transition-transform hover:scale-110 cursor-pointer ring-2 ring-white ${
                        isSelected ? 'bg-[#007230] text-white scale-110' : pin.colorClass
                      }`}
                    >
                      {pin.id}
                    </button>

                    {/* Popover Callout */}
                    {isSelected && (
                      <div className="absolute -top-24 -left-28 w-56 p-2.5 rounded-lg bg-white/95 backdrop-blur-md shadow-2xl text-[#131b2e] border border-[#dae2fd] z-30 animate-in fade-in">
                        <div className="flex items-center justify-between">
                          <span className="font-code-sm text-[10px] font-bold text-[#006e2d]">
                            {pin.label}
                          </span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setActivePinId(null);
                            }}
                            className="text-[#717970] hover:text-[#131b2e]"
                          >
                            ×
                          </button>
                        </div>
                        <p className="font-title-sm text-[12px] font-bold mt-0.5">{pin.title}</p>
                        <p className="font-body-sm text-[11px] text-[#404941] leading-snug mt-0.5">
                          {pin.description}
                        </p>
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Viewport Meta Footer */}
              <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg text-white font-code-sm text-[11px]">
                <div className="flex items-center gap-3">
                  <span>RES: 3024 × 3024</span>
                  <span>BAND: RGB + NIR PROXY</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-[#7cf994]"></span>
                  <span>PIN ACTIVE: {activePinId ? `ZONE 0${activePinId}` : 'NONE'}</span>
                </div>
              </div>
            </div>

            {/* Diagnostic Pins Interactive Legend */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
              {DIAGNOSTIC_PINS.map((pin) => (
                <button
                  key={pin.id}
                  onClick={() => setActivePinId(pin.id)}
                  className={`p-2.5 rounded-lg text-left transition-all border cursor-pointer ${
                    activePinId === pin.id
                      ? 'bg-[#eaedff] border-[#006e2d]'
                      : 'bg-[#f2f3ff] border-[#dae2fd]/60 hover:bg-[#eaedff]'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-5 h-5 rounded-full flex items-center justify-center font-bold text-[10px] shrink-0 text-white ${
                        pin.id === 1 ? 'bg-[#003b1b]' : pin.id === 2 ? 'bg-[#703a00]' : 'bg-[#ba1a1a]'
                      }`}
                    >
                      {pin.id}
                    </span>
                    <span className="font-title-sm text-[12px] font-bold text-[#131b2e] truncate">
                      {pin.title}
                    </span>
                  </div>
                  <p className="font-body-sm text-[11px] text-[#404941] mt-1 line-clamp-2">
                    {pin.description}
                  </p>
                </button>
              ))}
            </div>
          </div>

          {/* Ranked Stress Predictions Card */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
                  Ranked Stress Predictions
                </h2>
                <span className="font-body-sm text-[12px] text-[#717970]">
                  Bayesian Softmax Calibrated Multi-Class Probability
                </span>
              </div>
              <span className="font-label-sm text-[10px] px-2.5 py-1 rounded-full bg-[#7cf994] text-[#007230] font-bold uppercase tracking-wider">
                Top 3 Candidate Vectors
              </span>
            </div>

            {/* Primary Stress Vector Highlight */}
            <div className="p-4 rounded-xl bg-[#f2f3ff] border-2 border-[#006e2d]/30 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-2.5">
                  <div className="w-10 h-10 rounded-full bg-[#7cf994] text-[#007230] flex items-center justify-center font-bold shrink-0">
                    <span className="material-symbols-outlined text-[22px]">water_drop</span>
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-title-md text-[16px] text-[#003b1b] font-bold">
                        {STRESS_PREDICTIONS[0].title}
                      </span>
                      <span className="font-label-sm text-[10px] px-2 py-0.5 rounded-full bg-[#ffdcc3] text-[#2f1500] font-bold">
                        PRIMARY DIAGNOSIS
                      </span>
                    </div>
                    <span className="font-body-sm text-[12px] text-[#717970]">
                      {STRESS_PREDICTIONS[0].category}
                    </span>
                  </div>
                </div>

                <div className="flex items-baseline gap-1.5 self-end sm:self-auto">
                  <span className="font-headline-md text-[26px] text-[#006e2d] font-extrabold">82%</span>
                  <span className="font-code-sm text-[11px] text-[#717970]">CONFIDENCE</span>
                </div>
              </div>

              <p className="font-body-sm text-[13px] text-[#404941] leading-relaxed">
                {STRESS_PREDICTIONS[0].description}
              </p>

              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-[#dae2fd]">
                <span className="font-label-sm text-[11px] text-[#703a00] font-bold flex items-center gap-1">
                  <span className="material-symbols-outlined text-[15px]">warning</span>
                  Severity: Moderate (Tissue reversible with 48h hydration)
                </span>

                <button
                  onClick={() => onNavigate('plant-stress-library')}
                  className="font-title-sm text-[12px] text-[#006e2d] font-bold hover:underline flex items-center gap-1 cursor-pointer"
                >
                  <span>View in Stress Library</span>
                  <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                </button>
              </div>
            </div>

            {/* Secondary Candidate Rows */}
            <div className="space-y-2.5">
              {STRESS_PREDICTIONS.slice(1).map((prediction) => (
                <div
                  key={prediction.id}
                  className="p-3 rounded-lg bg-[#faf8ff] border border-[#dae2fd]/60 flex flex-col sm:flex-row sm:items-center justify-between gap-2 hover:bg-[#f2f3ff] transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-[#eaedff] text-[#404941] flex items-center justify-center font-bold shrink-0">
                      <span className="material-symbols-outlined text-[18px]">biotech</span>
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-title-sm text-[13px] text-[#131b2e] font-bold">
                          {prediction.title}
                        </span>
                        <span className="font-body-sm text-[11px] text-[#717970]">
                          ({prediction.category})
                        </span>
                      </div>
                      <p className="font-body-sm text-[11px] text-[#717970] mt-0.5 line-clamp-1">
                        {prediction.description}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 self-end sm:self-auto shrink-0">
                    <div className="w-20 h-2 bg-[#eaedff] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#717970] rounded-full"
                        style={{ width: `${prediction.confidence}%` }}
                      ></div>
                    </div>
                    <span className="font-code-sm text-[12px] font-bold text-[#404941] w-10 text-right">
                      {prediction.confidence}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Observed Physical Features Checklist */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 space-y-3">
            <h3 className="font-title-md text-title-md text-[#131b2e] font-bold">
              Observed Symptom Features (Vision Feature Extractor)
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {OBSERVED_SYMPTOMS.map((symptom) => (
                <div
                  key={symptom.id}
                  className="p-2.5 rounded-lg bg-[#f2f3ff] flex items-center justify-between border border-[#dae2fd]/40"
                >
                  <span className="font-body-sm text-[12px] text-[#131b2e] font-medium">
                    {symptom.feature}
                  </span>
                  <span
                    className={`font-label-sm text-[10px] px-2 py-0.5 rounded font-bold ${
                      symptom.color === 'yellow'
                        ? 'bg-[#ffdcc3] text-[#703a00]'
                        : symptom.color === 'green'
                        ? 'bg-[#7cf994] text-[#007230]'
                        : 'bg-[#ffdad6] text-[#ba1a1a]'
                    }`}
                  >
                    {symptom.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column (5 Cols): Root Cause & Prioritized Remediation */}
        <div className="lg:col-span-5 space-y-6">
          {/* Agronomic Root Cause Analysis */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
            <div>
              <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
                Root Cause Probabilities
              </h2>
              <span className="font-body-sm text-[12px] text-[#717970]">
                Correlated from physiological canopy parameters
              </span>
            </div>

            <div className="space-y-3">
              {ROOT_CAUSES.map((rc, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-[12px] font-title-sm">
                    <span className="text-[#131b2e] font-semibold">{rc.cause}</span>
                    <span className="font-code-sm text-[#006e2d] font-bold">{rc.probability}%</span>
                  </div>
                  <div className="w-full h-2 bg-[#f2f3ff] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        rc.probability > 70
                          ? 'bg-[#006e2d]'
                          : rc.probability > 40
                          ? 'bg-[#62df7d]'
                          : 'bg-[#dae2fd]'
                      }`}
                      style={{ width: `${rc.probability}%` }}
                    ></div>
                  </div>
                  <span className="font-body-sm text-[11px] text-[#717970]">{rc.description}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Evidence-Based Prioritized Remediation Protocol */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
                  Remediation Protocol
                </h2>
                <span className="font-body-sm text-[12px] text-[#717970]">
                  Prioritized Agronomic Action Plan
                </span>
              </div>
              <span className="font-code-sm text-[11px] px-2 py-0.5 rounded bg-[#eaedff] text-[#003b1b] font-bold">
                4-STAGE PLAN
              </span>
            </div>

            <div className="space-y-3">
              {REMEDIATION_STEPS.map((step) => (
                <div
                  key={step.priority}
                  className="p-3.5 rounded-xl bg-[#f2f3ff] border border-[#dae2fd]/60 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`font-label-sm text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${
                        step.priority === 1
                          ? 'bg-[#ffdad6] text-[#ba1a1a]'
                          : step.priority === 2
                          ? 'bg-[#ffdcc3] text-[#703a00]'
                          : 'bg-[#eaedff] text-[#003b1b]'
                      }`}
                    >
                      {step.priorityLabel}
                    </span>
                    <span className="font-code-sm text-[11px] text-[#717970] font-bold">
                      {step.metricNote}
                    </span>
                  </div>

                  <h3 className="font-title-sm text-[14px] text-[#131b2e] font-bold flex items-center gap-1.5">
                    <span className="material-symbols-outlined text-[16px] text-[#006e2d]">
                      {step.icon}
                    </span>
                    <span>{step.title}</span>
                  </h3>

                  <p className="font-body-sm text-[12px] text-[#404941] leading-relaxed">
                    {step.actionText}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* ML Pipeline Methodology Card */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-title-md text-title-md text-[#131b2e] font-bold">
                6-Stage Inference Pipeline
              </h3>
              <button
                onClick={() => onNavigate('model-pipeline')}
                className="font-label-sm text-[11px] text-[#006e2d] font-bold hover:underline cursor-pointer"
              >
                Architecture Specs
              </button>
            </div>

            <div className="space-y-2">
              {ML_PIPELINE_STAGES.map((st) => (
                <div
                  key={st.stageNumber}
                  className="p-2.5 rounded-lg bg-[#f2f3ff] flex items-center justify-between text-[12px] border border-[#dae2fd]/40"
                >
                  <div className="flex items-center gap-2.5">
                    <span className="material-symbols-outlined text-[16px] text-[#006e2d]">
                      {st.icon}
                    </span>
                    <div>
                      <span className="font-title-sm text-[12px] text-[#131b2e] font-bold">
                        {st.stageNumber}: {st.name}
                      </span>
                      <p className="font-body-sm text-[11px] text-[#717970] line-clamp-1">
                        {st.description}
                      </p>
                    </div>
                  </div>
                  <span className="material-symbols-outlined text-[16px] text-[#006e2d]">
                    verified
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
