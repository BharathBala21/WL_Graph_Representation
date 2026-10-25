import React from 'react';
import { Navigate, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { PageView } from './types';
import { DEFAULT_SAMPLE, SAMPLE_BY_ID } from './lib';
import { Header } from './components/Header';
import { DashboardView } from './components/DashboardView';
import { AnalyzePlantView } from './components/AnalyzePlantView';
import { DiagnosticsView } from './components/DiagnosticsView';
import { StressLibraryView } from './components/StressLibraryView';
import { CropsView } from './components/CropsView';
import { HistoryLogsView } from './components/HistoryLogsView';
import { MLPipelineView } from './components/MLPipelineView';
import { ProtocolView } from './components/ProtocolView';
import './plantcare.css';

/** PlantCare AI feature pages, in sidebar order. `route` is the URL segment under /plantcare. */
export const PLANTCARE_PAGES: { id: PageView; route: string; label: string; badge?: string }[] = [
  { id: 'dashboard', route: 'dashboard', label: 'Dashboard' },
  { id: 'analyze-plant', route: 'analyze', label: 'Analyze Leaf', badge: 'WL' },
  { id: 'results-diagnostics', route: 'diagnostics', label: 'Diagnostics' },
  { id: 'plant-stress-library', route: 'stress-library', label: 'Stress Library' },
  { id: 'crops', route: 'crops', label: 'Crops' },
  { id: 'history-logs', route: 'history', label: 'History & Logs' },
  { id: 'model-pipeline', route: 'ml-pipeline', label: 'ML Pipeline' },
  { id: 'protocol', route: 'protocol', label: 'Protocol' },
];

export const plantcarePath = (page: PageView) =>
  `/plantcare/${PLANTCARE_PAGES.find((p) => p.id === page)?.route ?? 'dashboard'}`;

export default function PlantCareApp() {
  const { page: route } = useParams();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const currentPage = PLANTCARE_PAGES.find((p) => p.route === route)?.id;
  const sample = SAMPLE_BY_ID[params.get('sample') ?? ''] ?? DEFAULT_SAMPLE;

  if (!currentPage) return <Navigate to="/plantcare/dashboard" replace />;

  const onNavigate = (page: PageView) => navigate(plantcarePath(page));
  const onOpenSample = (id: string) => navigate(`${plantcarePath('results-diagnostics')}?sample=${encodeURIComponent(id)}`);

  return (
    <div className="plantcare min-h-screen bg-pc-bg pt-16 text-pc-ink selection:bg-pc-mint selection:text-pc-deep-ink">
      <Header onNavigate={onNavigate} onOpenSample={onOpenSample} />

      <div key={currentPage} className="page-enter mx-auto max-w-[1600px] p-4 md:p-6 lg:p-8">
        {currentPage === 'dashboard' && <DashboardView onNavigate={onNavigate} onOpenSample={onOpenSample} />}
        {currentPage === 'analyze-plant' && <AnalyzePlantView onOpenSample={onOpenSample} />}
        {currentPage === 'results-diagnostics' && (
          <DiagnosticsView key={sample.id} sample={sample} onNavigate={onNavigate} onOpenSample={onOpenSample} />
        )}
        {currentPage === 'plant-stress-library' && <StressLibraryView onOpenSample={onOpenSample} />}
        {currentPage === 'crops' && <CropsView onOpenSample={onOpenSample} />}
        {currentPage === 'history-logs' && <HistoryLogsView onOpenSample={onOpenSample} />}
        {currentPage === 'model-pipeline' && <MLPipelineView />}
        {currentPage === 'protocol' && <ProtocolView />}
      </div>
    </div>
  );
}
