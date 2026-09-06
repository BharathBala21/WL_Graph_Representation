/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { PageView, InspectionRecord, MonitoredPlant, StressGuideItem } from './types';
import { RECENT_INSPECTIONS } from './data/mockData';
import { Navigation } from './components/Navigation';
import { Header } from './components/Header';
import { DashboardView } from './components/DashboardView';
import { AnalyzePlantView } from './components/AnalyzePlantView';
import { DiagnosticsView } from './components/DiagnosticsView';
import { StressLibraryView } from './components/StressLibraryView';
import { MyPlantsView } from './components/MyPlantsView';
import { HistoryLogsView } from './components/HistoryLogsView';
import { MLPipelineView } from './components/MLPipelineView';
import { SettingsView } from './components/SettingsView';
import {
  BatchScanModal,
  CompareScansModal,
  ShareWithAgronomistModal,
  ModelConfigModal,
} from './components/Modals';

export default function App() {
  const [currentPage, setCurrentPage] = useState<PageView>('dashboard');
  const [activeInspection, setActiveInspection] = useState<InspectionRecord>(RECENT_INSPECTIONS[0]);
  const [scanInitialSpecimen, setScanInitialSpecimen] = useState<{
    name: string;
    imageUrl: string;
    fileName: string;
  } | undefined>(undefined);

  // Modals
  const [isBatchScanOpen, setIsBatchScanOpen] = useState(false);
  const [isCompareScansOpen, setIsCompareScansOpen] = useState(false);
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);
  const [isModelConfigOpen, setIsModelConfigOpen] = useState(false);

  // Toast Notification
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const triggerToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleSelectInspection = (record: InspectionRecord) => {
    setActiveInspection(record);
    setCurrentPage('results-diagnostics');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleRunAnalysis = (specimenData: {
    name: string;
    imageUrl: string;
    taxon: string;
    fileName: string;
  }) => {
    const newRecord: InspectionRecord = {
      id: `PCAI-${Math.floor(10000 + Math.random() * 90000)}`,
      specimenName: specimenData.taxon.split('(')[0].trim() || 'Scanned Specimen',
      scientificName: specimenData.taxon.includes('(')
        ? specimenData.taxon.split('(')[1].split(')')[0]
        : 'Solanum lycopersicum',
      cultivar: 'Roma Hybrid',
      tagId: `#${Math.floor(10 + Math.random() * 90)}`,
      imageUrl: specimenData.imageUrl,
      timestamp: 'Just now',
      dateStr: new Date().toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }),
      statusText: 'Water Stress Detected (Moderate)',
      statusType: 'water-stress',
      confidence: 84,
      symptomSummary: 'Symptom: Interveinal chlorosis & curled leaf margins',
      severity: 'Moderate',
      quadrant: 'Quadrant B-02',
      ndvi: 0.682,
      vari: 0.510,
      temperature: 28.1,
    };
    setActiveInspection(newRecord);
    triggerToast('Neural inference complete! Diagnostic report generated.');
  };

  const handleScanPlant = (plant: MonitoredPlant) => {
    setScanInitialSpecimen({
      name: `${plant.name.toLowerCase().replace(/\s+/g, '_')}_scan.jpg`,
      imageUrl: RECENT_INSPECTIONS[0].imageUrl,
      fileName: `${plant.name.toLowerCase().replace(/\s+/g, '_')}_scan.jpg`,
    });
    setCurrentPage('analyze-plant');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleScanStress = (guide: StressGuideItem) => {
    setScanInitialSpecimen({
      name: guide.title.toLowerCase().replace(/\s+/g, '_') + '.jpg',
      imageUrl: guide.imageUrl,
      fileName: guide.title.toLowerCase().replace(/\s+/g, '_') + '.jpg',
    });
    setCurrentPage('analyze-plant');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleExportCSV = () => {
    const headers = [
      'Specimen_ID',
      'Crop_Name',
      'Scientific_Name',
      'Status',
      'Confidence_Pct',
      'Severity',
      'NDVI_Mean',
      'VARI_Index',
      'Timestamp',
    ];
    const rows = RECENT_INSPECTIONS.map((rec) => [
      rec.id,
      `"${rec.specimenName}"`,
      `"${rec.scientificName}"`,
      `"${rec.statusText}"`,
      rec.confidence,
      rec.severity,
      rec.ndvi || 0.75,
      rec.vari || 0.55,
      `"${rec.dateStr}"`,
    ]);
    const csvContent = [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `plantcare_ai_inspections_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    triggerToast('Inspection audit CSV exported successfully.');
  };

  const handleExportPDF = () => {
    triggerToast('Preparing diagnostic PDF for print / download...');
    setTimeout(() => {
      window.print();
    }, 400);
  };

  const handleStartBatch = (count: number) => {
    triggerToast(`Batch complete: Ingested & analyzed ${count} specimens.`);
  };

  return (
    <div className="min-h-screen bg-[#faf8ff] text-[#131b2e] flex flex-col font-sans selection:bg-[#7cf994] selection:text-[#003b1b]">
      {/* Navigation (Sidebar on Desktop, Bottom Bar on Mobile) */}
      <Navigation
        currentPage={currentPage}
        onNavigate={(page) => {
          setCurrentPage(page);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }}
      />

      {/* Main Content Area */}
      <div className="flex-1 md:pl-64 flex flex-col min-w-0">
        {/* Fixed Header */}
        <Header
          onNavigate={(page) => {
            setCurrentPage(page);
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
          onSearchSelect={(term) => {
            triggerToast(`Jumped to: ${term}`);
          }}
        />

        {/* Dynamic Page Router Body */}
        <main className="flex-1 p-4 md:p-6 lg:p-8 mt-16 mb-20 md:mb-8">
          {currentPage === 'dashboard' && (
            <DashboardView
              onNavigate={(page) => {
                setCurrentPage(page);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              onSelectInspection={handleSelectInspection}
              onOpenBatchScan={() => setIsBatchScanOpen(true)}
              onOpenLiveCamera={() => {
                setCurrentPage('analyze-plant');
              }}
              onExportCSV={handleExportCSV}
            />
          )}

          {currentPage === 'analyze-plant' && (
            <AnalyzePlantView
              onNavigate={(page) => {
                setCurrentPage(page);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              onRunAnalysis={handleRunAnalysis}
              onOpenModelConfig={() => setIsModelConfigOpen(true)}
              initialSpecimen={scanInitialSpecimen}
            />
          )}

          {currentPage === 'results-diagnostics' && (
            <DiagnosticsView
              onNavigate={(page) => {
                setCurrentPage(page);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              activeInspection={activeInspection}
              onSaveToMyPlants={() => {
                triggerToast(`Saved ${activeInspection.specimenName} to My Monitored Plants`);
              }}
              onOpenCompareScans={() => setIsCompareScansOpen(true)}
              onOpenShareModal={() => setIsShareModalOpen(true)}
              onExportPDF={handleExportPDF}
            />
          )}

          {currentPage === 'plant-stress-library' && (
            <StressLibraryView
              onNavigate={(page) => {
                setCurrentPage(page);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              onScanStress={handleScanStress}
            />
          )}

          {currentPage === 'my-plants-health-tracking' && (
            <MyPlantsView
              onNavigate={(page) => {
                setCurrentPage(page);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              onScanPlant={handleScanPlant}
            />
          )}

          {currentPage === 'history-logs' && (
            <HistoryLogsView
              onNavigate={(page) => {
                setCurrentPage(page);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              onSelectInspection={handleSelectInspection}
              onExportCSV={handleExportCSV}
            />
          )}

          {currentPage === 'model-pipeline' && (
            <MLPipelineView
              onNavigate={(page) => {
                setCurrentPage(page);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              onExportPDF={handleExportPDF}
            />
          )}

          {currentPage === 'settings' && (
            <SettingsView
              onNavigate={(page) => {
                setCurrentPage(page);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
            />
          )}
        </main>
      </div>

      {/* Modals Container */}
      <BatchScanModal
        isOpen={isBatchScanOpen}
        onClose={() => setIsBatchScanOpen(false)}
        onStartBatch={handleStartBatch}
      />

      <CompareScansModal
        isOpen={isCompareScansOpen}
        onClose={() => setIsCompareScansOpen(false)}
        activeInspection={activeInspection}
      />

      <ShareWithAgronomistModal
        isOpen={isShareModalOpen}
        onClose={() => setIsShareModalOpen(false)}
        specimenId={activeInspection?.id}
      />

      <ModelConfigModal
        isOpen={isModelConfigOpen}
        onClose={() => setIsModelConfigOpen(false)}
      />

      {/* Floating Global Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-20 md:bottom-6 right-6 z-50 px-4 py-2.5 rounded-xl bg-[#003b1b] text-white font-title-sm text-[13px] font-bold shadow-2xl flex items-center gap-2 border border-[#7cf994]/40 animate-in slide-in-from-bottom-2 duration-200">
          <span className="material-symbols-outlined text-[#7cf994] text-[18px]">verified</span>
          <span>{toastMessage}</span>
        </div>
      )}
    </div>
  );
}
