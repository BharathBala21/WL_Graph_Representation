export type PageView =
  | 'dashboard'
  | 'analyze-plant'
  | 'results-diagnostics'
  | 'plant-stress-library'
  | 'crops'
  | 'history-logs'
  | 'model-pipeline'
  | 'protocol';

export type FeatureFamily = 'Morphology' | 'Colour' | 'Texture' | 'Spatial';
export type ConditionType = 'Control' | 'Nutrient deficiency' | 'Pest' | 'Disease';

/** One image's recorded out-of-fold outcome from the Level-3 hierarchical WL pipeline. */
export interface Sample {
  id: string;
  image: string;
  file: string;
  crop: string;
  condition: string;
  classLabel: string;
  fold: number;
  trueStress: number;
  predictedCrop: string;
  predictedStress: number;
  routingCorrect: boolean;
  stressCorrect: boolean;
  /** Graph topology and WL height chosen by inner CV for this image's crop and fold. */
  graph: string;
  height: number;
  /** Global Level-1 WL + SVM (Hybrid, h = 3) prediction and probability of "stressed". */
  globalPrediction: number | null;
  globalProbability: number | null;
  /** 57-class reference prediction (flat RBF SVM). */
  fineGrainedPrediction: string | null;
  maskAreaRatio: number;
  /** 38 handcrafted feature values, in RESEARCH.features order. */
  features: number[];
}

export interface CropSummary {
  id: string;
  name: string;
  scientific: string;
  images: number;
  healthy: number;
  stressed: number;
  conditions: string[];
  routingAccuracy: number;
  endToEndAccuracy: number;
  healthyMeans: number[];
  globalWl: { accuracy: number; f1: number };
  cropSpecific: {
    graph: string;
    height: number;
    accuracy: number;
    accuracySd: number;
    balancedAccuracy: number;
    macroF1: number;
    macroF1Sd: number;
    rocAuc: number;
  };
}

export interface ConditionSummary {
  code: string;
  name: string;
  scientific: string;
  type: ConditionType;
  images: number;
  crops: string[];
  /** Healthy: share kept as healthy. Stresses: share flagged as stressed (Level-3 OOF). */
  detectionRate: number;
  classes: { id: string; crop: string; images: number; rfRecall: number | null; rfF1: number | null }[];
}

export interface GraphSummary {
  graph: string;
  edges: number;
  density: number;
  avgDegree: number;
  components: number;
  isolated: number;
  source: string;
  meaning: string;
}

export interface ResearchData {
  dataset: {
    name: string;
    rawImages: number;
    accepted: number;
    crops: number;
    classes: number;
    conditions: number;
    healthy: number;
    stressed: number;
    folds: number;
  };
  config: {
    n_bins: number;
    bin_labels: string[];
    corr_thresh: number;
    knn_k: number;
    wl_heights: number[];
    random_state: number;
  };
  features: { id: string; family: FeatureFamily }[];
  /** [mean, sd] per feature over the accepted set. */
  featureStats: [number, number][];
  /** [33rd, 67th] percentile per feature — display-only Low/Medium/High node labels. */
  featureTertiles: [number, number][];
  crops: CropSummary[];
  conditions: ConditionSummary[];
  samples: Sample[];
  graphs: GraphSummary[];
  /** Edges per topology as [featureIndex, featureIndex]. */
  edgeLists: Record<string, [number, number][]>;
  level1Grid: { graph: string; height: number; accuracy: number; balancedAccuracy: number; macroF1: number; rocAuc: number }[];
  level1Baselines: { method: string; accuracy: number; balancedAccuracy: number; macroF1: number; rocAuc: number }[];
  hierarchical: { method: string; accuracy: number[]; macroF1: number[]; balancedAccuracy: number[] }[];
  cropClassifier: { model: string; accuracy: number[]; macroF1: number[] }[];
  fineGrained: { method: string; representation: string; accuracy: number; macroF1: number }[];
  rfImportances: { feature: string; importance: number }[];
}
