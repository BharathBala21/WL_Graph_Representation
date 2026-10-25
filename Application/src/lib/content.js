// WL Graph Kernel research project — all content lives here.
// Items marked TODO_* should be replaced before publishing.

export const LINKS = {
  github: 'TODO_GITHUB_URL', // Set to your actual repo URL, e.g. 'https://github.com/YOUR_USER/YOUR_REPO'
  video: 'TODO_DEMO_VIDEO_URL', // YouTube embed URL or /video/demo.mp4
}

export const isTodo = (v) => !v || String(v).startsWith('TODO_')

export const PROJECT = {
  short: 'WL-LeafKernel',
  title:
    'How Does Graph Construction Influence the Effectiveness of Weisfeiler–Lehman Graph Kernels for Fine-Grained Plant-Stress Classification from Leaf Images?',
  tagline:
    'Systematically comparing five feature-graph topologies for WL kernel-based plant-stress classification.',
  type: 'Type 1: Research Project',
  team: 'TODO_TEAM_NAME', // e.g. 'Team 21'
  department: 'TODO_DEPARTMENT', // e.g. 'Department of CSE'
  guide: {
    name: 'TODO_SUPERVISOR_NAME', // e.g. 'Dr. Jane Smith'
    role: 'TODO_SUPERVISOR_ROLE', // e.g. 'Associate Professor, Department of CSE'
  },
}

// ── 12 home-page sections (section bar + command palette) ──────────────────
export const HOME_SECTIONS = [
  { id: 'overview',     label: 'Overview' },
  { id: 'how-to-use',  label: 'How to use' },
  { id: 'objectives',  label: 'Objectives' },
  { id: 'problem',     label: 'Problem' },
  { id: 'methodology', label: 'Methodology' },
  { id: 'results',     label: 'Results' },
  { id: 'novelty',     label: 'Novelty' },
  { id: 'team',        label: 'Team' },
  { id: 'progress',    label: 'Progress' },
  { id: 'docs',        label: 'Docs' },
  { id: 'github',      label: 'GitHub' },
  { id: 'video',       label: 'Video' },
]

// ── Overview hero stats ────────────────────────────────────────────────────
export const HERO_STATS = [
  { v: 38,    l: 'handcrafted features per image' },
  { v: 5,     l: 'graph-construction strategies' },
  { v: 4,     l: 'WL subtree heights (h = 0–3)' },
  { v: 4387,  l: 'OLID-I images (quality-controlled)' },
]

// ── Research objectives ────────────────────────────────────────────────────
export const OBJECTIVES = [
  {
    title: 'Evaluate graph topology impact',
    body:  'Measure how the choice of feature-graph construction strategy affects WL graph-kernel classification performance on plant-stress data.',
  },
  {
    title: 'Compare five graph topologies',
    body:  'Systematically benchmark Correlation, Feature-Similarity kNN, Feature-Family, Spatial/Regional and Hybrid graph-construction strategies.',
  },
  {
    title: 'Investigate WL subtree heights',
    body:  'Assess the effect of WL subtree-refinement height h ∈ {0, 1, 2, 3} on kernel expressiveness and downstream classification.',
  },
  {
    title: 'Compare against ML baselines',
    body:  'Benchmark WL kernel SVM against conventional tabular classifiers — Random Forest and RBF SVM — under the same leakage-safe nested cross-validation protocol.',
  },
  {
    title: 'Class-wise evaluation',
    body:  'Analyse accuracy, balanced accuracy, Macro-F1, per-class recall and confusion matrices to surface class-imbalance effects beyond aggregate accuracy.',
  },
  {
    title: 'Prospective farm evaluation',
    body:  'Prepare for independent external evaluation on leaf images collected during a future farm visit, assessing generalization to real-world field conditions.',
  },
]

// ── Problem statement cards ───────────────────────────────────────────────
export const PROBLEMS = [
  {
    title: 'Field variation and class imbalance',
    body:  'Leaf-stress classification is challenging under real field variation in lighting, background, and growth stage, compounded by strong class imbalance between healthy and stressed samples.',
    icon: 'Leaf',
  },
  {
    title: 'Flat vectors miss feature relationships',
    body:  'Conventional tabular representations treat all 38 handcrafted features as independent, discarding the relational structure among morphology, colour, texture and spatial feature groups.',
    icon: 'Network',
  },
  {
    title: 'Graph topology choice is underexplored',
    body:  'A graph-based representation may preserve inter-feature relationships, but its utility for WL kernels depends critically on the topology used — a design choice rarely compared systematically in plant-pathology literature.',
    icon: 'GitBranch',
  },
]

export const GAPS = [
  {
    title: 'Topology choice is rarely compared.',
    body:  'Most graph-based plant-disease papers fix one topology without ablating alternatives. This project fills that gap by systematically comparing five engineered construction strategies.',
  },
  {
    title: 'WL kernels on handcrafted features are understudied.',
    body:  'WL kernels are well-studied on molecular and citation graphs; their application to handcrafted leaf-feature graphs for stress classification is largely unexplored.',
  },
  {
    title: 'Class imbalance reporting is often incomplete.',
    body:  'Reporting only overall accuracy can mask poor recall on the minority (healthy) class. This project evaluates balanced accuracy, Macro-F1 and per-class recall explicitly.',
  },
]

// ── Methodology pipeline stages ───────────────────────────────────────────
export const STAGES = [
  {
    id: 'input',
    label: 'Leaf image',
    sub: 'OLID-I field / greenhouse image',
    color: 'muted',
    details: {
      what:    'A leaf image from the OLID-I dataset (field or greenhouse) enters the pipeline. Images pass segmentation quality control before features are extracted.',
      bullets: ['Eight crops, 57 stress classes', '4,387 images accepted after QC', 'Tomato subset: 580 images (184 healthy, 396 stressed)'],
    },
  },
  {
    id: 'features',
    label: 'Feature extraction',
    sub: '38 handcrafted morphology, colour, texture & spatial features',
    color: 'primary',
    details: {
      what:    'Thirty-eight handcrafted features are computed from each segmented leaf region, covering four groups: morphology, colour statistics, texture (GLCM/LBP), and spatial/regional descriptors.',
      bullets: ['Morphology: area, perimeter, eccentricity, convexity…', 'Colour: mean/std per channel, colour ratios…', 'Texture: GLCM contrast, correlation, LBP histogram…', 'Spatial: lesion density, lesion distribution, lesion count…'],
    },
  },
  {
    id: 'graph',
    label: 'Graph construction',
    sub: '5 topology strategies; nodes = feature channels',
    color: 'accent',
    details: {
      what:    'Each image is represented as a graph where nodes correspond to the 38 feature channels (not pixels or images). Edges are added according to one of five topology strategies.',
      bullets: [
        'Correlation graph: edge if |Pearson r| > threshold on training fold',
        'Feature-similarity kNN: connect k nearest features by value similarity',
        'Feature-family: intra-group edges among morphology / colour / texture / spatial groups',
        'Spatial/Regional: connect spatially co-located feature channels',
        'Hybrid: combine family + correlation edges',
      ],
    },
  },
  {
    id: 'wl',
    label: 'WL kernel',
    sub: 'Subtree-label refinement h = 0–3',
    color: 'primary',
    details: {
      what:    'Node labels are discretized using training-fold statistics only (preventing leakage). The Weisfeiler–Lehman subtree kernel is computed by iteratively refining labels for h rounds and accumulating label-count histograms.',
      bullets: [
        'h = 0: node-label histogram only (no structural information)',
        'h = 1–3: increasingly deep neighbourhood aggregation',
        'Gram matrix normalised; SVM trained on kernel matrix',
        'Preprocessing (discretisation) fitted on training fold only',
      ],
    },
  },
  {
    id: 'output',
    label: 'Classification & evaluation',
    sub: 'Nested 5-fold CV · WL vs RF vs RBF SVM',
    color: 'ok',
    details: {
      what:    'A support vector machine is trained on the WL kernel matrix. Evaluation uses nested cross-validation: five outer folds for held-out test estimates, five inner folds to select graph topology and WL height. Random Forest and RBF SVM baselines are evaluated under the same protocol.',
      bullets: [
        'Metrics: accuracy, balanced accuracy, Macro-F1, per-class recall',
        'Confusion matrices and configuration-selection stability',
        'No data leakage: preprocessing fitted per training fold',
        'Farm-image evaluation planned (prospective, not yet collected)',
      ],
    },
  },
]

// ── Dataset ────────────────────────────────────────────────────────────────
export const DATASETS = [
  {
    name: 'OLID-I',
    use: 'Plant-stress classification',
    stats: [
      { value: 4387, label: 'Accepted images (after QC)' },
      { value: 8,    label: 'Crops' },
      { value: 57,   label: 'Stress classes' },
    ],
    body: 'Open Leaf Image Dataset I — contains field and greenhouse leaf images across eight crops and 57 crop-stress combinations. 4,387 images pass segmentation quality control. The focused tomato binary experiment uses 580 images: 184 healthy and 396 stressed.',
  },
]

// ── Tomato nested-CV results (verified source: prompt specification) ───────
// These are TOMATO BINARY results only — do not mix with the 57-class task.
export const TOMATO_RESULTS = [
  {
    model: 'Nested WL Kernel',
    acc:   78.97, acc_sd: 6.38,
    bacc:  75.33, bacc_sd: 7.78,
    f1:    75.38, f1_sd:  7.64,
    recall_healthy:  65.32, recall_healthy_sd: null,
    recall_stressed: null,
    ours: true,
    note: 'Configuration selected by inner 5-fold CV from 5 topologies × 4 WL heights.',
  },
  {
    model: 'Random Forest',
    acc:   83.45, acc_sd: 4.58,
    bacc:  76.56, bacc_sd: 6.06,
    f1:    78.66, f1_sd:  6.37,
    recall_healthy:  57.67, recall_healthy_sd: null,
    recall_stressed: null,
    ours: false,
  },
  {
    model: 'RBF SVM',
    acc:   82.59, acc_sd: 5.04,
    bacc:  77.24, bacc_sd: 6.08,
    f1:    78.57, f1_sd:  6.33,
    recall_healthy:  null, recall_healthy_sd: null,
    recall_stressed: null,
    ours: false,
  },
]

// ── Novelty ────────────────────────────────────────────────────────────────
export const NOVELTY = [
  {
    n: '01',
    title: 'Systematic topology comparison for WL kernels',
    points: [
      'Five engineered graph-construction strategies compared under a single, leakage-safe evaluation protocol',
      'Correlation, feature-similarity kNN, feature-family, spatial/regional, and hybrid topologies all evaluated',
      'Contribution: empirical evidence on which topology benefits WL kernel performance',
    ],
  },
  {
    n: '02',
    title: 'WL height analysis on handcrafted leaf features',
    points: [
      'WL subtree heights h = 0, 1, 2, 3 systematically investigated',
      'Distinguishes structural information gain from purely label-count signal',
      'Applied to handcrafted feature graphs — an underexplored setting in plant pathology',
    ],
  },
  {
    n: '03',
    title: 'Leakage-aware experimental design',
    points: [
      'Nested cross-validation: outer fold for test estimates, inner fold for configuration selection',
      'Feature discretisation fitted on training fold only — no leakage from test data',
      'Class-wise recall and balanced accuracy reported explicitly to surface imbalance effects',
      'Prospective external evaluation on farm-collected images planned (not yet collected)',
    ],
  },
]

// ── Team ───────────────────────────────────────────────────────────────────
export const TEAM = [
  {
    name:     'Merugumala Sandeep',
    reg:      '23CSE664',
    photo:    '/team/23CSE664.jpg',
    linkedin: 'https://www.linkedin.com/in/sandeep-merugumala',
    github:   'https://github.com/Sandeep-Merugumala',
  },
  {
    name:     'Chilukuri Subrahmanya Sri Prakash',
    reg:      '23CSE610',
    photo:    '',
    linkedin: 'https://www.linkedin.com/in/chilukuriprakash',
    github:   'https://github.com/prakashchilukuri',
  },
  {
    name:     'Bala Bharath Chandra Sai',
    reg:      '23CSE609',
    photo:    '/team/23CSE609.jpg',
    linkedin: 'https://www.linkedin.com/in/bharathbala21/',
    github:   'https://github.com/BharathBala21',
  },
  {
    name:     'Muppagowni Vamsidhar',
    reg:      '23CSE613',
    photo:    '/team/23CSE613.jpg',
    linkedin: 'https://www.linkedin.com/in/vamsidhar-muppagowni-723797382',
    github:   'https://github.com/Vamsidhar-Muppagowni',
  },
  {
    name:     'Yerneni Vishnu Vardhan',
    reg:      '23CSE154',
    photo:    '/team/23CSE154.jpg',
    linkedin: 'https://www.linkedin.com/in/vishnu-vardhan-yerneni-1158a1298',
    github:   'https://github.com/vishnuyerneni-maker',
  },
]

// ── Progress timeline ──────────────────────────────────────────────────────
export const TIMELINE = [
  {
    phase:  'Phase 1',
    title:  'Dataset preparation and quality control',
    status: 'done',
    body:   'OLID-I dataset curated: 4,387 images accepted after segmentation quality control across eight crops and 57 stress classes.',
  },
  {
    phase:  'Phase 1',
    title:  'Handcrafted feature extraction',
    status: 'done',
    body:   '38 morphology, colour, texture and spatial features extracted per segmented leaf image and saved as feature matrices.',
  },
  {
    phase:  'Phase 2',
    title:  'Graph construction and WL kernel experiments',
    status: 'done',
    body:   'Five graph-construction strategies implemented and WL subtree kernel computed for h ∈ {0, 1, 2, 3}. Gram matrices saved.',
  },
  {
    phase:  'Phase 2',
    title:  'Conventional baseline evaluation',
    status: 'done',
    body:   'Random Forest and RBF SVM baselines evaluated on the tabular feature vectors under the same nested cross-validation protocol.',
  },
  {
    phase:  'Phase 3',
    title:  'Tomato nested-CV audit',
    status: 'done',
    body:   'Nested five-fold CV results verified for the tomato binary stress task: 580 images (184 healthy, 396 stressed).',
  },
  {
    phase:  'Phase 3',
    title:  'Results consolidation and frontend integration',
    status: 'active',
    body:   'Verified results consolidated. Project presentation frontend adapted and integrated with the actual experimental outcomes.',
  },
  {
    phase:  'Planned',
    title:  'Farm-image collection and external evaluation',
    status: 'planned',
    body:   'Leaf images will be collected during a farm visit and used as a prospective external test set to assess generalisation to real-world field conditions. Not yet completed.',
  },
]

// ── How-to-use ─────────────────────────────────────────────────────────────
export const HOW_TO = {
  steps: [
    { title: 'Upload a leaf image', body: 'Select or drag-and-drop a segmented leaf image (PNG / JPEG). The system validates the format and file size before processing.' },
    { title: 'Preprocessing', body: 'Features are extracted and discretised using training-fold statistics. Graph edges are added according to the selected topology.' },
    { title: 'Choose a graph topology', body: 'Select one of the five supported construction strategies. Each produces a different adjacency structure over the 38 feature nodes.' },
    { title: 'WL kernel inference', body: 'The WL subtree kernel is computed and the SVM classifier produces a stress-class prediction with a decision-score margin. Note: this interface is a prototype — backend integration is in progress.' },
  ],
  pages: {
    '/': [
      'Use the section bar (or Ctrl + K) to jump to any of the 12 sections.',
      'Click a stage in the Methodology diagram to see details about each pipeline step.',
      'Results are grouped by experiment: tomato binary task, 57-class task (if verified), and planned farm evaluation.',
    ],
  },
}

// ── Documentation links ────────────────────────────────────────────────────
// Only link to files that actually exist. Update paths as artifacts become available.
export const DOCUMENTS = [
  {
    title: 'Research methodology notes',
    file:  null, // Set to actual path when available, e.g. '/pdf/methodology.pdf'
    pages: null,
    desc:  'Graph construction strategies, WL kernel algorithm, and nested cross-validation design.',
    internal: true,
  },
  {
    title: 'Tomato nested-CV audit report',
    file:  null,
    pages: null,
    desc:  'Verified results for the tomato binary stress detection experiment (5 outer folds).',
    internal: true,
  },
]

export const REFERENCES = [
  'N. Shervashidze et al., "Weisfeiler-Lehman Graph Kernels," JMLR, vol. 12, pp. 2539–2561, 2011.',
  'K. Barbedo, "A review on the detection of plant disease using image processing," Electronic Letters in Plant Pathology, 2018.',
  'OLID-I: Open Leaf Image Dataset I — multi-crop, multi-stress field and greenhouse images.',
  'L. Breiman, "Random Forests," Machine Learning, vol. 45, pp. 5–32, 2001.',
  'C. Cortes and V. Vapnik, "Support vector networks," Machine Learning, vol. 20, pp. 273–297, 1995.',
]

export const DISCLAIMER =
  'For research and educational purposes only. Results represent experimentally verified cross-validation estimates on the OLID-I dataset. Farm-image evaluation is planned but not yet completed.'

// ─── Utilities ─────────────────────────────────────────────────────────────
export const formatRelativeTime = (dateStr) => {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return diffMins + 'm ago'
  if (diffHours < 24) return diffHours + 'h ago'
  if (diffDays < 7) return diffDays + 'd ago'
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i]
}
