export type PageView =
  | 'dashboard'
  | 'analyze-plant'
  | 'results-diagnostics'
  | 'plant-stress-library'
  | 'my-plants-health-tracking'
  | 'history-logs'
  | 'model-pipeline'
  | 'settings';

export interface InspectionRecord {
  id: string;
  specimenName: string;
  scientificName: string;
  cultivar?: string;
  tagId: string;
  imageUrl: string;
  timestamp: string;
  dateStr: string;
  statusText: string;
  statusType: 'water-stress' | 'healthy' | 'deficiency' | 'disease' | 'heat';
  confidence: number;
  symptomSummary: string;
  severity: 'Optimal' | 'Mild' | 'Moderate' | 'Severe';
  quadrant?: string;
  ndvi?: number;
  vari?: number;
  temperature?: number;
}

export interface MonitoredPlant {
  id: string;
  name: string;
  scientificName: string;
  icon: string;
  status: 'Optimal' | 'Moderate Stress' | 'Mild Heat Stress' | 'Mild Deficiency';
  statusTrend: 'Improving' | 'Stable' | 'Worsening' | 'Peak';
  trendIcon: string;
  metricLabel: string;
  metricValue: string;
  lastScanned: string;
  quadrant: string;
  imageUrl?: string;
  healthRatio: number;
}

export interface DiagnosticZonePin {
  id: number;
  topPct: string;
  rightPct: string;
  label: string;
  title: string;
  description: string;
  colorClass: string;
}

export interface StressPrediction {
  id: string;
  title: string;
  category: string;
  confidence: number;
  description: string;
  severity: 'Mild' | 'Moderate' | 'Severe';
  colorHex: string;
}

export interface ObservedSymptom {
  id: string;
  feature: string;
  status: string;
  color: 'yellow' | 'orange' | 'green' | 'red';
}

export interface RootCauseProbability {
  cause: string;
  probability: number;
  description: string;
}

export interface RemediationStep {
  priority: number;
  priorityLabel: string;
  timeframe: string;
  title: string;
  actionText: string;
  metricNote: string;
  icon: string;
}

export interface MLPipelineStage {
  stageNumber: string;
  name: string;
  description: string;
  icon: string;
  status?: 'COMPLETE' | 'PROCESSING' | 'QUEUED';
}

export interface ReferenceSpecimen {
  id: string;
  title: string;
  scientificName: string;
  category: string;
  categoryType: 'Abiotic' | 'Fungal' | 'Deficiency' | 'Negative Test';
  imageUrl: string;
  fileName: string;
  description: string;
}

export interface StressGuideItem {
  id: string;
  title: string;
  scientificTaxa: string;
  category: 'Water Stress' | 'Nutrient Deficiency' | 'Pathogen / Disease' | 'Thermal Extremes' | 'Pest Infestation';
  type: 'Abiotic' | 'Biotic';
  severityDefault: 'Mild' | 'Moderate' | 'Critical';
  symptoms: string[];
  visualIndicators: string;
  causes: string[];
  remediation: string[];
  affectedCrops: string[];
  imageUrl: string;
}
