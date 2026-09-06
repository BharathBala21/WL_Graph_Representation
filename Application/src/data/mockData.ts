import {
  InspectionRecord,
  MonitoredPlant,
  ReferenceSpecimen,
  StressGuideItem,
  DiagnosticZonePin,
  StressPrediction,
  ObservedSymptom,
  RootCauseProbability,
  RemediationStep,
  MLPipelineStage
} from '../types';

export const RECENT_INSPECTIONS: InspectionRecord[] = [
  {
    id: 'PCAI-98421',
    specimenName: 'Field Tomato #04',
    scientificName: 'Solanum lycopersicum',
    cultivar: 'Roma',
    tagId: '#04',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBMiDgfnnLYcUUsdxm7Exw-SA6swXGrlE_dD_I073_e0JhtESnSIAixpEEScG0-R5_EZV1qDKXOg8zEC4LGYOoPYSqgY8waadfiUKD3PYFT_3Ydwua8Nlir_htUuEgJt-RDyYdeV-n3eFa4SefXDprGJ_tpO858GG_YnHGUyxNX_ltacFHKMNmdim3SaL9B_YWO4gO40hxtKKvoxv1LYSYvZqtV2n4YBAfDSLsIBGlHxXBc0A9B56cR',
    timestamp: '2 hours ago',
    dateStr: 'Oct 24, 2024 (14:32 PST)',
    statusText: 'Water Stress Detected (Moderate)',
    statusType: 'water-stress',
    confidence: 82,
    symptomSummary: 'Symptom: Interveinal chlorosis & curled leaf margins',
    severity: 'Moderate',
    quadrant: 'Quadrant B-02',
    ndvi: 0.684,
    vari: 0.512,
    temperature: 28.4
  },
  {
    id: 'PCAI-98402',
    specimenName: 'Hydroponic Basil B2',
    scientificName: 'Ocimum basilicum',
    cultivar: 'Genovese',
    tagId: 'B2',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCdY71TQWRNkNtZth-ZzPxGtEb4yobSSPl_wqu_IGqcZPGFB3841G-4iXOxf_tmNzNUtuUjGQFcBmpkTmb9EY6YCl1PYARaid4FKcEiJGbC9YtL8DjA6Y4uW8uVg5LmhYOImYqeHNdXAIhGxOtxye4Ykrp0cG4owvvKjbOlln8BbFbt1fqzP7jabVAnJ5bFxIGKQ00PxOK6IHp4LkepZC_uraQ7TBvcCT6r5iVNGsUag5ec9eP5ljv-',
    timestamp: 'Yesterday',
    dateStr: 'Oct 23, 2024 (09:14 PST)',
    statusText: 'Healthy / Optimal',
    statusType: 'healthy',
    confidence: 96,
    symptomSummary: 'Tissue turgidity and chlorophyll uniform',
    severity: 'Optimal',
    quadrant: 'Greenhouse Block A',
    ndvi: 0.891,
    vari: 0.742,
    temperature: 23.1
  },
  {
    id: 'PCAI-98388',
    specimenName: 'Bell Pepper Plot C',
    scientificName: 'Capsicum annuum',
    cultivar: 'California Wonder',
    tagId: 'C-01',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDG4s2KwROsg0zC8Cc7zas7-wvdIwGoL81aaXyKf0KnJTI3Xq3-kAwZ3EqvEybqhhsPGxnMEdPLD06jdoOqqEjfpYXRoQy14xQpZz2AFXTyLYOISPt0g14HR8A76i4K7TgKgx7V9mwP-vw18noQuOJw7KGcyULgY68tyekP_pKv-zhp-ho-wS7oGSRWOGuLgYQfqBwTbMPxp8_8L_j-a8aEOPKOK6-Uy83Zo84BkNfY1wcg_Y5fmgjL',
    timestamp: '2 days ago',
    dateStr: 'Oct 22, 2024 (16:45 PST)',
    statusText: 'Nitrogen Deficiency (Mild)',
    statusType: 'deficiency',
    confidence: 71,
    symptomSummary: 'General pale yellowing in lower canopy leaves',
    severity: 'Mild',
    quadrant: 'Plot C-South',
    ndvi: 0.612,
    vari: 0.445,
    temperature: 25.6
  },
  {
    id: 'PCAI-98350',
    specimenName: 'Sweet Corn Canopy #11',
    scientificName: 'Zea mays',
    cultivar: 'Golden Bantam',
    tagId: 'M-11',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDB48q0TXC4vozgXnzR55I7-A67QkoHraN7864ouwn2f1oWyPWYaCeHeH0e4Bf9dcheMQTS12n32dLolt0sTnyQV9miNErnemhlLEbXQ2GIihijk5_UuQo6T-3XOEAwe-qDaUK4OF5g9HwwWsejKvVXB7Q4n9lBrk4KVmFSVlz9PzRJPpG3w4lOGlB7diJ7LN5tbfNLEFNn2J8OXxyTFAaLdlcuFlEWrGc40mDrRAcagfQdYmlL6xdw',
    timestamp: '3 days ago',
    dateStr: 'Oct 21, 2024 (11:20 PST)',
    statusText: 'Common Rust (Puccinia sorghi)',
    statusType: 'disease',
    confidence: 88,
    symptomSummary: 'Raised cinnamon-brown pustules scattered on adaxial leaf blades',
    severity: 'Moderate',
    quadrant: 'Field Block D',
    ndvi: 0.589,
    vari: 0.420,
    temperature: 27.2
  },
  {
    id: 'PCAI-98312',
    specimenName: 'Soybean Row 4',
    scientificName: 'Glycine max',
    cultivar: 'Asgrow AG27XF0',
    tagId: 'SB-04',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAsFpWu4Ehq27CKVOF533r_hfUYPgBjMtBR0bD1tSZjvYHFwdbXtKT7yXeId3_eu16ATQE8xvYGE5kXYeilho7av8mz0KAbow-oBjIWI-MjcKi6azO8ErrPXHwvFmYIf6jVyM3ZWGtHLL-p_WlPPomuRXjcBVnlnWqr_2bWgBN5Z5j8d4dO1Tjav4C6CMO_3W8AHJaLtHJeEs-NJuv2WqzF6bRv_pFB8wn8SzeuSU4d44FAnIlLaIKT',
    timestamp: '4 days ago',
    dateStr: 'Oct 20, 2024 (15:05 PST)',
    statusText: 'Iron Deficiency Chlorosis (IDC)',
    statusType: 'deficiency',
    confidence: 79,
    symptomSummary: 'Sharp yellowing between vivid dark green veins in younger leaves',
    severity: 'Moderate',
    quadrant: 'Plot A-North',
    ndvi: 0.635,
    vari: 0.480,
    temperature: 24.9
  },
  {
    id: 'PCAI-98290',
    specimenName: 'Citrus Orchard #02',
    scientificName: 'Citrus limon',
    cultivar: 'Eureka',
    tagId: 'CT-02',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCdpYDMe8gUrarNRBhCaQY9M6HOOdVZ96SKJsJ6QVoHrfkSbQYlyn5urhIEpIKrmeOv9-V_31yNeH63SGAM-BWlL9BI3vFM50ezYIHEKD7rwpL-q3WWfNH42L31_ljGRPi2TKTc__6X_TGPIxve0aDyZdXIKE5Q5TxumbrSDXMJt0O8bXwFqcOSGNgtzlwHMfFqZLbeGlBPdnx0TEW5_7F04fM1tvQssq5nZsfYKUh5LUe4zRaNdtP5',
    timestamp: '5 days ago',
    dateStr: 'Oct 19, 2024 (10:18 PST)',
    statusText: 'Optimal Foliar Turgor',
    statusType: 'healthy',
    confidence: 98,
    symptomSummary: 'Glossy dark green waxy surface, negative pathogen lesions',
    severity: 'Optimal',
    quadrant: 'Grove Block 3',
    ndvi: 0.840,
    vari: 0.710,
    temperature: 22.8
  }
];

export const MONITORED_PLANTS: MonitoredPlant[] = [
  {
    id: 'plant-1',
    name: 'Tomato',
    scientificName: 'Solanum lycopersicum',
    icon: 'nutrition',
    status: 'Moderate Stress',
    statusTrend: 'Improving',
    trendIcon: 'trending_up',
    metricLabel: 'Hydration',
    metricValue: '48% VWC',
    lastScanned: '2 hours ago',
    quadrant: 'Garden Bed B',
    healthRatio: 68
  },
  {
    id: 'plant-2',
    name: 'Sweet Corn',
    scientificName: 'Zea mays',
    icon: 'grass',
    status: 'Optimal',
    statusTrend: 'Stable',
    trendIcon: 'horizontal_rule',
    metricLabel: 'Canopy Index',
    metricValue: '0.78 NDVI',
    lastScanned: '1 day ago',
    quadrant: 'Field Block D',
    healthRatio: 94
  },
  {
    id: 'plant-3',
    name: 'Strawberry',
    scientificName: 'Fragaria × ananassa',
    icon: 'spa',
    status: 'Mild Heat Stress',
    statusTrend: 'Peak',
    trendIcon: 'thermostat',
    metricLabel: 'Thermal Peak',
    metricValue: '31.4°C',
    lastScanned: '6 hours ago',
    quadrant: 'Tunnel Bay 4',
    healthRatio: 76
  },
  {
    id: 'plant-4',
    name: 'Hydroponic Basil',
    scientificName: 'Ocimum basilicum',
    icon: 'eco',
    status: 'Optimal',
    statusTrend: 'Stable',
    trendIcon: 'horizontal_rule',
    metricLabel: 'SPAD Index',
    metricValue: '44.2',
    lastScanned: 'Yesterday',
    quadrant: 'Hydroponic B2',
    healthRatio: 98
  },
  {
    id: 'plant-5',
    name: 'Bell Pepper',
    scientificName: 'Capsicum annuum',
    icon: 'potted_plant',
    status: 'Mild Deficiency',
    statusTrend: 'Improving',
    trendIcon: 'trending_up',
    metricLabel: 'N-Status',
    metricValue: 'Sub-Optimal',
    lastScanned: '2 days ago',
    quadrant: 'Plot C-South',
    healthRatio: 74
  }
];

export const REFERENCE_SPECIMENS: ReferenceSpecimen[] = [
  {
    id: 'ref-tomato',
    title: 'Tomato Osmotic',
    scientificName: 'Solanum lycopersicum',
    category: 'Water Stress (Abiotic)',
    categoryType: 'Abiotic',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDtKSnA_1HLvjdt_HDGeVe811EdrxLCX8YVwjV29D0dSc-7BjQnqD1Pa7qbgFZvKsWcA-81CIQuV7M9JMN2oDxGxiP7IvXmj3s3e6G4m_uTcAvUiLB-d8FjGfYces-uZNH6-jmiA7kEcqMrVbdbbqrPZjloKvg5b-9sUzoFYskRa5SmvcOUsLP3dsYTcFbdXsUpmF37G57Udhjz9rdrk3wpTzgmGt2rH2IXMb110d39O1JpC_vDSCov',
    fileName: 'tomato_osmotic_water_stress.jpg',
    description: 'Leaf margin dehydration, inward longitudinal curling, and dull foliage caused by localized root hydration deficit.'
  },
  {
    id: 'ref-maize',
    title: 'Maize Common Rust',
    scientificName: 'Zea mays',
    category: 'Puccinia sorghi (Fungal)',
    categoryType: 'Fungal',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDB48q0TXC4vozgXnzR55I7-A67QkoHraN7864ouwn2f1oWyPWYaCeHeH0e4Bf9dcheMQTS12n32dLolt0sTnyQV9miNErnemhlLEbXQ2GIihijk5_UuQo6T-3XOEAwe-qDaUK4OF5g9HwwWsejKvVXB7Q4n9lBrk4KVmFSVlz9PzRJPpG3w4lOGlB7diJ7LN5tbfNLEFNn2J8OXxyTFAaLdlcuFlEWrGc40mDrRAcagfQdYmlL6xdw',
    fileName: 'maize_rust_puccinia_sorghi.jpg',
    description: 'Elevated golden to dark chestnut pustules on both upper and lower leaf surfaces, erupting through the epidermal cell walls.'
  },
  {
    id: 'ref-soybean',
    title: 'Soybean IDC',
    scientificName: 'Glycine max',
    category: 'Nutrient Deficiency',
    categoryType: 'Deficiency',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAsFpWu4Ehq27CKVOF533r_hfUYPgBjMtBR0bD1tSZjvYHFwdbXtKT7yXeId3_eu16ATQE8xvYGE5kXYeilho7av8mz0KAbow-oBjIWI-MjcKi6azO8ErrPXHwvFmYIf6jVyM3ZWGtHLL-p_WlPPomuRXjcBVnlnWqr_2bWgBN5Z5j8d4dO1Tjav4C6CMO_3W8AHJaLtHJeEs-NJuv2WqzF6bRv_pFB8wn8SzeuSU4d44FAnIlLaIKT',
    fileName: 'soybean_iron_deficiency_chlorosis.jpg',
    description: 'Acute lime-yellow interveinal discoloration with high contrast retained deep green venation network.'
  },
  {
    id: 'ref-citrus',
    title: 'Citrus Control',
    scientificName: 'Citrus limon',
    category: 'Vigorous (Negative Test)',
    categoryType: 'Negative Test',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCdpYDMe8gUrarNRBhCaQY9M6HOOdVZ96SKJsJ6QVoHrfkSbQYlyn5urhIEpIKrmeOv9-V_31yNeH63SGAM-BWlL9BI3vFM50ezYIHEKD7rwpL-q3WWfNH42L31_ljGRPi2TKTc__6X_TGPIxve0aDyZdXIKE5Q5TxumbrSDXMJt0O8bXwFqcOSGNgtzlwHMfFqZLbeGlBPdnx0TEW5_7F04fM1tvQssq5nZsfYKUh5LUe4zRaNdtP5',
    fileName: 'citrus_healthy_control.jpg',
    description: 'Ideal healthy foliage baseline exhibiting strong waxy cuticle layer, uniform chlorophyll concentration, and pristine margins.'
  }
];

export const DIAGNOSTIC_PINS: DiagnosticZonePin[] = [
  {
    id: 1,
    topPct: '28%',
    rightPct: '32%',
    label: 'ZONE 01 • CHLOROSIS',
    title: 'Chlorotic Margin',
    description: 'Loss of chlorophyll-a/b. Interveinal yellowing with preserved axial green veins.',
    colorClass: 'bg-primary text-on-primary'
  },
  {
    id: 2,
    topPct: '58%',
    rightPct: '24%',
    label: 'ZONE 02 • MARGINAL CURL',
    title: 'Inward Curling',
    description: 'Stomatal defense action. Inward cupping and turgor reduction along distal leaflet.',
    colorClass: 'bg-tertiary text-on-tertiary'
  },
  {
    id: 3,
    topPct: '78%',
    rightPct: '34%',
    label: 'ZONE 03 • NECROSIS',
    title: 'Early Necrosis',
    description: 'Focal tissue desiccation. Early cellular senescence and brown dessication at margin apex.',
    colorClass: 'bg-error text-on-error'
  }
];

export const STRESS_PREDICTIONS: StressPrediction[] = [
  {
    id: 'water-stress',
    title: 'Water Stress',
    category: 'Sub-acute Moisture Deficit',
    confidence: 82,
    description: 'Consistent with lower canopy conservation response: prominent lateral leaflet flaccidity, marginal inward cupping, and secondary interveinal chlorosis.',
    severity: 'Moderate',
    colorHex: '#006e2d'
  },
  {
    id: 'nutrient-deficiency',
    title: 'Nutrient Deficiency',
    category: 'Nitrogen (N) / Potassium (K)',
    confidence: 51,
    description: 'Basal chlorosis mimics mobile macronutrient re-translocation triggered by root uptake inhibition.',
    severity: 'Mild',
    colorHex: '#717970'
  },
  {
    id: 'heat-vpd',
    title: 'Heat / VPD Stress',
    category: 'Vapor Pressure Excess',
    confidence: 29,
    description: 'Elevated leaf transpiration demands exceeding vascular supply rates.',
    severity: 'Mild',
    colorHex: '#c0c9be'
  }
];

export const OBSERVED_SYMPTOMS: ObservedSymptom[] = [
  { id: '1', feature: 'Interveinal Chlorosis', status: 'Marked Yellowing', color: 'yellow' },
  { id: '2', feature: 'Leaf Inward Curling', status: 'Bilateral Cupping', color: 'yellow' },
  { id: '3', feature: 'Turgor Pressure Loss', status: 'Flaccid Tissue', color: 'green' },
  { id: '4', feature: 'Marginal Apex Necrosis', status: 'Focal Browning', color: 'red' },
  { id: '5', feature: 'Vascular Petiole Integrity', status: 'Intact / Normal', color: 'green' }
];

export const ROOT_CAUSES: RootCauseProbability[] = [
  {
    cause: 'Underwatering / Infrequent Irrigation',
    probability: 84,
    description: 'High tension in substrate moisture tension matrix.'
  },
  {
    cause: 'Root Substrate Aeration / Compaction',
    probability: 42,
    description: 'Compacted media limiting capillary hydration pathways.'
  },
  {
    cause: 'Atmospheric VPD Spike (> 1.6 kPa)',
    probability: 38,
    description: 'Dry microclimate pulling moisture faster than root replenishment.'
  },
  {
    cause: 'Secondary Nutrient Leaching',
    probability: 22,
    description: 'Nutrient lockout secondary to erratic moisture oscillations.'
  }
];

export const REMEDIATION_STEPS: RemediationStep[] = [
  {
    priority: 1,
    priorityLabel: 'Priority 1 • Immediate',
    timeframe: 'Immediate',
    title: 'Verify Root-Zone Moisture',
    actionText: 'Insert a moisture probe or finger 5-8 cm into the root zone. If dry and crumbly, apply slow deep soak until 10% runoff occurs. Avoid saturated flooding.',
    metricNote: 'Target Substrate: 55-65% Vol.',
    icon: 'timer'
  },
  {
    priority: 2,
    priorityLabel: 'Priority 2 • 24 Hours',
    timeframe: '24 Hours',
    title: 'Inspect Abaxial Leaf Face',
    actionText: 'Examine undersides of symptomatic leaflets with 10x hand lens. Ensure two-spotted spider mites (Tetranychus urticae) are not draining epidermal fluids.',
    metricNote: 'Check: Fine stippling & web',
    icon: 'search_check'
  },
  {
    priority: 3,
    priorityLabel: 'Priority 3 • 48 Hours',
    timeframe: '48 Hours',
    title: 'Shield Midday Irradiance',
    actionText: 'Provide 30% shade cloth buffering or relocate container away from harsh noon sun until leaf turgor restores. Reduce thermal transpirational load.',
    metricNote: 'Max PPFD: 450 μmol/m²/s',
    icon: 'wb_twilight'
  },
  {
    priority: 4,
    priorityLabel: 'Priority 4 • 72h Scan',
    timeframe: '72h Scan',
    title: 'Perform Follow-up Scan',
    actionText: 'Re-scan this exact leaflet in 3 days. Healthy recovery will halt inward curling progression. Yellowed regions will stabilize without spreading axially.',
    metricNote: 'Auto-reminder scheduled',
    icon: 'history'
  }
];

export const ML_PIPELINE_STAGES: MLPipelineStage[] = [
  {
    stageNumber: 'STAGE 01',
    name: 'Image Ingestion',
    description: '3024x3024 Raw RGB pixel normalization & chromatic calibration.',
    icon: 'photo_camera'
  },
  {
    stageNumber: 'STAGE 02',
    name: 'Preprocessing',
    description: 'Background soil masking and leaf blade boundary segmentation via Otsu.',
    icon: 'blur_on'
  },
  {
    stageNumber: 'STAGE 03',
    name: 'Feature Embeddings',
    description: 'ViT multi-head attention mapping chlorosis and vein geometry.',
    icon: 'grid_view'
  },
  {
    stageNumber: 'STAGE 04',
    name: 'Classifier Head',
    description: 'Multi-label gradient-boosted ensemble across 48 crop stress profiles.',
    icon: 'device_hub'
  },
  {
    stageNumber: 'STAGE 05',
    name: 'Softmax Calibration',
    description: 'Bayesian uncertainty estimation yielding 87.4% model reliability.',
    icon: 'analytics'
  },
  {
    stageNumber: 'STAGE 06',
    name: 'Knowledge Engine',
    description: 'Agronomic rules synthesized to output actionable remediation protocols.',
    icon: 'auto_stories'
  }
];

export const STRESS_GUIDES: StressGuideItem[] = [
  {
    id: 'guide-water-stress',
    title: 'Water Stress Diagnosis (Drought & Desiccation)',
    scientificTaxa: 'Hydraulic Conductivity Loss',
    category: 'Water Stress',
    type: 'Abiotic',
    severityDefault: 'Moderate',
    symptoms: [
      'Bilateral inward leaf margin curling',
      'Loss of petiole and leaflet turgor',
      'Lower canopy interveinal chlorosis',
      'Premature blossom and fruit drop'
    ],
    visualIndicators: 'Crisp marginal necrosis beginning at leaf apex, flaccid adaxial surface, dull greyish-green color shift.',
    causes: [
      'Inadequate irrigation volume or frequency',
      'High vapor pressure deficit (VPD > 1.8 kPa)',
      'Substrate hydrophobic drying',
      'Root system compaction'
    ],
    remediation: [
      'Immediate slow, bottom-up capillary or pulse drip irrigation until container field capacity reached',
      'Temporarily buffer direct sun exposure with 30-40% shading cloth',
      'Mulch top 3-5 cm of soil bed to reduce evaporation losses',
      'Verify volumetric water content with digital TDR tensiometer'
    ],
    affectedCrops: ['Solanum lycopersicum (Tomato)', 'Capsicum annuum (Pepper)', 'Zea mays (Corn)', 'Fragaria × ananassa (Strawberry)'],
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDtKSnA_1HLvjdt_HDGeVe811EdrxLCX8YVwjV29D0dSc-7BjQnqD1Pa7qbgFZvKsWcA-81CIQuV7M9JMN2oDxGxiP7IvXmj3s3e6G4m_uTcAvUiLB-d8FjGfYces-uZNH6-jmiA7kEcqMrVbdbbqrPZjloKvg5b-9sUzoFYskRa5SmvcOUsLP3dsYTcFbdXsUpmF37G57Udhjz9rdrk3wpTzgmGt2rH2IXMb110d39O1JpC_vDSCov'
  },
  {
    id: 'guide-nitrogen-deficiency',
    title: 'Nitrogen (N) Deficiency Chlorosis',
    scientificTaxa: 'Macronutrient Translocation',
    category: 'Nutrient Deficiency',
    type: 'Abiotic',
    severityDefault: 'Moderate',
    symptoms: [
      'Uniform pale green to bright yellowing in oldest basal leaves',
      'Slowed vegetative shoot elongation and thin spindly stems',
      'Reddish/purplish tints in petioles and veins under severe stress',
      'Premature foliar senescence from bottom up'
    ],
    visualIndicators: 'Generalized chlorosis beginning from leaf tips and traveling backwards down main midrib; veins yellow alongside lamina.',
    causes: [
      'Insufficient nitrate/ammonium in fertigation mix',
      'Heavy root-zone leaching due to overwatering',
      'Soil pH outside optimal range (pH > 7.5 or < 5.5)',
      'High C:N ratio mulch immobilizing available nitrogen'
    ],
    remediation: [
      'Apply water-soluble calcium nitrate or balanced N-P-K fertigation at 100-150 ppm N',
      'Conduct foliar spray of 1% urea solution in early morning for rapid uptake',
      'Adjust root-zone pH to 6.0 - 6.5',
      'Check substrate electrical conductivity (EC)'
    ],
    affectedCrops: ['Zea mays (Maize)', 'Glycine max (Soybean)', 'Solanum lycopersicum', 'Brassica oleracea'],
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDG4s2KwROsg0zC8Cc7zas7-wvdIwGoL81aaXyKf0KnJTI3Xq3-kAwZ3EqvEybqhhsPGxnMEdPLD06jdoOqqEjfpYXRoQy14xQpZz2AFXTyLYOISPt0g14HR8A76i4K7TgKgx7V9mwP-vw18noQuOJw7KGcyULgY68tyekP_pKv-zhp-ho-wS7oGSRWOGuLgYQfqBwTbMPxp8_8L_j-a8aEOPKOK6-Uy83Zo84BkNfY1wcg_Y5fmgjL'
  },
  {
    id: 'guide-iron-chlorosis',
    title: 'Iron (Fe) Deficiency Interveinal Chlorosis',
    scientificTaxa: 'Micronutrient Immobility',
    category: 'Nutrient Deficiency',
    type: 'Abiotic',
    severityDefault: 'Moderate',
    symptoms: [
      'High-contrast yellowing between vividly dark green veins',
      'Symptoms manifest first on newest apical foliage',
      'Under severe cases, complete leaf bleaching to ivory white',
      'Marginal scorch and necrotic spotting on apical leaves'
    ],
    visualIndicators: 'Sharp fine green reticulate vein pattern standing out against bright yellow interveinal lamina on terminal shoots.',
    causes: [
      'High bicarbonate or calcareous alkaline soils (pH > 7.4)',
      'Waterlogged, cold, or compacted root zone preventing root respiration',
      'Excessive soil phosphorus or heavy metals tying up iron'
    ],
    remediation: [
      'Apply Fe-EDDHA chelated iron drench (stable up to pH 9.0)',
      'Acidify fertigation water with citric or sulfuric acid to target pH 5.8-6.2',
      'Improve soil aeration and allow root media to cycle between wet and damp',
      'Apply foliar Fe-EDTA spray in low-light conditions'
    ],
    affectedCrops: ['Glycine max (Soybean)', 'Citrus limon (Citrus)', 'Fragaria × ananassa', 'Vitis vinifera (Grapevine)'],
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAsFpWu4Ehq27CKVOF533r_hfUYPgBjMtBR0bD1tSZjvYHFwdbXtKT7yXeId3_eu16ATQE8xvYGE5kXYeilho7av8mz0KAbow-oBjIWI-MjcKi6azO8ErrPXHwvFmYIf6jVyM3ZWGtHLL-p_WlPPomuRXjcBVnlnWqr_2bWgBN5Z5j8d4dO1Tjav4C6CMO_3W8AHJaLtHJeEs-NJuv2WqzF6bRv_pFB8wn8SzeuSU4d44FAnIlLaIKT'
  },
  {
    id: 'guide-maize-rust',
    title: 'Common Rust (Puccinia sorghi)',
    scientificTaxa: 'Puccinia sorghi Schwein.',
    category: 'Pathogen / Disease',
    type: 'Biotic',
    severityDefault: 'Critical',
    symptoms: [
      'Small, circular to elongate raised cinnamon-brown pustules (uredinia)',
      'Pustules scattered loosely across both adaxial and abaxial blade surfaces',
      'Epidermis ruptures releasing powdery rust-colored spores',
      'Leaves turn chlorotic and desiccate prematurely under heavy pressure'
    ],
    visualIndicators: 'Raised brownish-red powdery blisters rupturing leaf epidermis on both upper and lower surface.',
    causes: [
      'Wind-borne fungal urediniospores carried from southern zones',
      'High relative humidity (> 95%) and moderate temperatures (16–25°C)',
      'Extended leaf wetness duration (> 6 hours)'
    ],
    remediation: [
      'Deploy registered strobilurin or triazole fungicides at first sign of pustules',
      'Plant rust-resistant hybrids featuring Rp single-gene or general adult-plant resistance',
      'Monitor weather patterns and eliminate volunteer corn / weed hosts',
      'Avoid late afternoon overhead sprinkler irrigation'
    ],
    affectedCrops: ['Zea mays (Corn / Maize)', 'Sorghum bicolor', 'Teosinte'],
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDB48q0TXC4vozgXnzR55I7-A67QkoHraN7864ouwn2f1oWyPWYaCeHeH0e4Bf9dcheMQTS12n32dLolt0sTnyQV9miNErnemhlLEbXQ2GIihijk5_UuQo6T-3XOEAwe-qDaUK4OF5g9HwwWsejKvVXB7Q4n9lBrk4KVmFSVlz9PzRJPpG3w4lOGlB7diJ7LN5tbfNLEFNn2J8OXxyTFAaLdlcuFlEWrGc40mDrRAcagfQdYmlL6xdw'
  },
  {
    id: 'guide-heat-scorch',
    title: 'Temperature Extremes & Sunscald Necrosis',
    scientificTaxa: 'Photo-Thermal Cellular Rupture',
    category: 'Thermal Extremes',
    type: 'Abiotic',
    severityDefault: 'Moderate',
    symptoms: [
      'Bleached, papery white to silver-tan patches on exposed leaf facets',
      'Marginal scorching and crispy desiccation',
      'Leaf cupping and epinasty under acute midday radiant peaks',
      'Necrotic collapse of epidermal cells exposed to direct solar beam'
    ],
    visualIndicators: 'Thin, parchment-like dead leaf tissue with sharp delineation between damaged and shaded portions.',
    causes: [
      'High leaf canopy temperature exceeding 35°C',
      'Excessive photosynthetically active radiation (PAR > 1800 μmol/m²/s)',
      'Water droplets acting as lenses during peak irradiance',
      'Sudden removal of canopy shade or defoliation'
    ],
    remediation: [
      'Install 30% - 50% aluminized reflective thermal shade netting',
      'Maintain adequate root-zone hydration to support transpirational cooling',
      'Operate overhead micro-misting systems to lower ambient canopy VPD',
      'Avoid wetting leaves during midday peak solar intensity'
    ],
    affectedCrops: ['Fragaria × ananassa (Strawberry)', 'Solanum lycopersicum (Tomato)', 'Capsicum annuum (Pepper)', 'Lactuca sativa (Lettuce)'],
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBMiDgfnnLYcUUsdxm7Exw-SA6swXGrlE_dD_I073_e0JhtESnSIAixpEEScG0-R5_EZV1qDKXOg8zEC4LGYOoPYSqgY8waadfiUKD3PYFT_3Ydwua8Nlir_htUuEgJt-RDyYdeV-n3eFa4SefXDprGJ_tpO858GG_YnHGUyxNX_ltacFHKMNmdim3SaL9B_YWO4gO40hxtKKvoxv1LYSYvZqtV2n4YBAfDSLsIBGlHxXBc0A9B56cR'
  }
];
