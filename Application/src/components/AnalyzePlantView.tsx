import React, { useState, useRef } from 'react';
import { PageView, ReferenceSpecimen } from '../types';
import { REFERENCE_SPECIMENS } from '../data/mockData';

interface AnalyzePlantViewProps {
  onNavigate: (page: PageView) => void;
  onRunAnalysis: (specimenData: {
    name: string;
    imageUrl: string;
    taxon: string;
    fileName: string;
  }) => void;
  onOpenModelConfig: () => void;
  initialSpecimen?: {
    name: string;
    imageUrl: string;
    fileName: string;
  };
}

export const AnalyzePlantView: React.FC<AnalyzePlantViewProps> = ({
  onNavigate,
  onRunAnalysis,
  onOpenModelConfig,
  initialSpecimen,
}) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'camera'>('upload');
  const [selectedTaxon, setSelectedTaxon] = useState('Tomato (Solanum lycopersicum) — Solanaceae');
  const [rotationDegree, setRotationDegree] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStage, setAnalysisStage] = useState<number>(2); // 1, 2, 3, 4
  const [stageProgress, setStageProgress] = useState(78);

  // Active Specimen Image
  const [specimenImage, setSpecimenImage] = useState<string>(
    initialSpecimen?.imageUrl ||
      'https://lh3.googleusercontent.com/aida-public/AB6AXuA67ll4tB_hUlQdvJMXGCgUz1wD7R1yvE74EQ9UCwICLUVwFmC80ULTpisQ8pPU7lShRYg_9nN3pU69L-ldc02kamNk71yi8Tcv4fyO_0pNdn-lRhM_vpm18Y3b2-mYU8mzTAXcSmYHqnBwkgWagOTxNig9pRb7yzCaDyTGVkPAk9fi-Jeyl9PAnyP_CLgXGUJSFvu3W9YgHkgo8k0MXZKvJ0hgQWy2RQ06Rfe98-wmfBkHiZDK-PSv'
  );
  const [specimenName, setSpecimenName] = useState<string>(
    initialSpecimen?.name || 'tomato_leaf_sample_09.jpg'
  );
  const [fileDetails, setFileDetails] = useState<string>('tomato_leaf_sample_09.jpg • 4.2 MB • 3024 × 3024');

  // Camera handling
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startCamera = async () => {
    try {
      setCameraError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsCameraActive(true);
      }
    } catch (err) {
      console.warn('Camera access unavailable or declined:', err);
      setCameraError('Camera stream unavailable. You can upload an image or select a specimen below.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth || 1280;
      canvas.height = videoRef.current.videoHeight || 720;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
        setSpecimenImage(dataUrl);
        setSpecimenName('camera_foliar_capture.jpg');
        setFileDetails('camera_foliar_capture.jpg • High-Res Ingest • Dynamic FOV');
        stopCamera();
        setActiveTab('upload');
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (loadEvt) => {
        if (loadEvt.target?.result) {
          setSpecimenImage(loadEvt.target.result as string);
          setSpecimenName(file.name);
          const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
          setFileDetails(`${file.name} • ${sizeMb} MB • Calibration OK`);
          setRotationDegree(0);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const loadReferenceSpecimen = (specimen: ReferenceSpecimen) => {
    setSpecimenImage(specimen.imageUrl);
    setSpecimenName(specimen.fileName);
    setFileDetails(`${specimen.fileName} • Specimen Ref • Validated Archive`);
    setRotationDegree(0);
    if (specimen.id === 'ref-tomato') {
      setSelectedTaxon('Tomato (Solanum lycopersicum) — Solanaceae');
    } else if (specimen.id === 'ref-maize') {
      setSelectedTaxon('Corn / Maize (Zea mays) — Poaceae');
    } else if (specimen.id === 'ref-soybean') {
      setSelectedTaxon('Soybean (Glycine max) — Fabaceae');
    } else if (specimen.id === 'ref-citrus') {
      setSelectedTaxon('Citrus / Lemon (Citrus limon) — Rutaceae');
    }
  };

  const handleRunAnalysis = () => {
    setIsAnalyzing(true);
    setAnalysisStage(1);

    // Progressive stage simulation
    setTimeout(() => {
      setAnalysisStage(2);
      setStageProgress(45);
    }, 400);

    setTimeout(() => {
      setStageProgress(90);
    }, 900);

    setTimeout(() => {
      setAnalysisStage(3);
    }, 1300);

    setTimeout(() => {
      setAnalysisStage(4);
    }, 1800);

    setTimeout(() => {
      setIsAnalyzing(false);
      onRunAnalysis({
        name: specimenName,
        imageUrl: specimenImage,
        taxon: selectedTaxon,
        fileName: specimenName,
      });
      onNavigate('results-diagnostics');
    }, 2400);
  };

  return (
    <div id="analyze-plant-view" className="max-w-[1440px] w-full mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Top Workflow Header & Breadcrumb Protocol */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-2 border-b border-[#dae2fd]/40">
        <div className="space-y-1">
          <div className="flex items-center gap-1.5 text-[#006e2d] font-label-md text-[11px] uppercase tracking-wider font-bold">
            <span className="material-symbols-outlined text-[18px]">neurology</span>
            <span>Inference Protocol v2.4 • Optical Spectrometry</span>
          </div>
          <h1 className="font-headline-lg text-[28px] md:text-[32px] text-[#131b2e] font-extrabold tracking-tight">
            Analyze Your Plant
          </h1>
          <p className="font-body-md text-[14px] text-[#404941] max-w-2xl">
            Capture or ingest high-resolution leaf or plant imagery for deep learning stress classification and phenotypic validation.
          </p>
        </div>

        {/* Pipeline Workflow Stepper */}
        <div className="bg-[#eaedff] p-1.5 rounded-xl flex items-center gap-2 shadow-xs self-start md:self-auto border border-[#dae2fd]">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white shadow-xs">
            <span className="w-5 h-5 rounded-full bg-[#14532d] text-white flex items-center justify-center font-code-sm text-[11px] font-bold">
              1
            </span>
            <span className="font-title-sm text-[13px] text-[#131b2e] font-bold">Source</span>
          </div>
          <span className="material-symbols-outlined text-[#717970] text-[16px]">chevron_right</span>

          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${isAnalyzing ? 'bg-white shadow-xs text-[#131b2e] font-bold' : 'text-[#404941]'}`}>
            <span className={`w-5 h-5 rounded-full flex items-center justify-center font-code-sm text-[11px] font-semibold ${isAnalyzing ? 'bg-[#006e2d] text-white' : 'bg-[#e2e7ff] text-[#404941]'}`}>
              2
            </span>
            <span className="font-title-sm text-[13px]">Preprocessing</span>
          </div>
          <span className="material-symbols-outlined text-[#717970] text-[16px]">chevron_right</span>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-[#404941]">
            <span className="w-5 h-5 rounded-full bg-[#e2e7ff] text-[#404941] flex items-center justify-center font-code-sm text-[11px] font-semibold">
              3
            </span>
            <span className="font-title-sm text-[13px]">Inference</span>
          </div>
        </div>
      </div>

      {/* Main Interactive Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Source Upload / Camera Ingestion Workspace */}
        <div className="lg:col-span-7 flex flex-col gap-5">
          {/* Surface Card for Upload & Preview */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 space-y-5">
            {/* Mode Selector Tabs */}
            <div className="flex items-center justify-between">
              <div className="inline-flex p-1 bg-[#f2f3ff] rounded-lg gap-1 border border-[#dae2fd]/60">
                <button
                  id="tab-upload"
                  onClick={() => {
                    setActiveTab('upload');
                    stopCamera();
                  }}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-lg font-title-sm text-[13px] transition-all cursor-pointer ${
                    activeTab === 'upload'
                      ? 'bg-white text-[#131b2e] font-bold shadow-xs'
                      : 'text-[#404941] hover:text-[#131b2e]'
                  }`}
                >
                  <span className="material-symbols-outlined text-[18px] text-[#006e2d]">cloud_upload</span>
                  <span>Upload Image</span>
                </button>

                <button
                  id="tab-camera"
                  onClick={() => {
                    setActiveTab('camera');
                    startCamera();
                  }}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-lg font-title-sm text-[13px] transition-all cursor-pointer ${
                    activeTab === 'camera'
                      ? 'bg-white text-[#131b2e] font-bold shadow-xs'
                      : 'text-[#404941] hover:text-[#131b2e]'
                  }`}
                >
                  <span className="material-symbols-outlined text-[18px] text-[#006e2d]">photo_camera</span>
                  <span>Camera Stream</span>
                </button>
              </div>

              <div className="hidden sm:flex items-center gap-2 text-[#404941] font-code-sm text-[11px]">
                <span className="w-2 h-2 rounded-full bg-[#006e2d] animate-pulse"></span>
                <span>Sensor Ready</span>
              </div>
            </div>

            {/* Tab Content 1: Upload View */}
            {activeTab === 'upload' && (
              <div className="space-y-4">
                {/* Dropzone Banner */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="relative group cursor-pointer bg-[#f2f3ff] hover:bg-[#eaedff] transition-all rounded-xl p-6 text-center flex flex-col items-center justify-center gap-2 border-2 border-dashed border-[#dae2fd] hover:border-[#006e2d]/50"
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png, image/jpeg, image/heic"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <div className="w-12 h-12 rounded-full bg-[#7cf994]/50 text-[#007230] flex items-center justify-center group-hover:scale-110 transition-transform">
                    <span className="material-symbols-outlined text-[28px]">add_a_photo</span>
                  </div>
                  <div className="space-y-1">
                    <p className="font-title-sm text-title-sm text-[#131b2e] font-semibold">
                      Drag & drop leaf specimen, or{' '}
                      <span className="text-[#006e2d] underline decoration-[#006e2d]/40 font-bold">
                        Browse Files
                      </span>
                    </p>
                    <p className="font-body-sm text-[12px] text-[#717970]">
                      Supports High-Res JPG, PNG, HEIC (Max 25MB). Auto-calibrated for macro detail.
                    </p>
                  </div>
                </div>

                {/* Active Specimen Inspection Panel */}
                <div className="bg-[#f2f3ff] rounded-xl p-3 space-y-3 border border-[#dae2fd]/60">
                  <div className="flex items-center justify-between px-1">
                    <div className="flex items-center gap-2">
                      <span className="font-label-sm text-[10px] px-2 py-0.5 rounded-full bg-[#14532d] text-white font-bold">
                        PRE-INSPECTION BUFFER
                      </span>
                      <span className="font-code-sm text-[11px] text-[#404941]">FOV: 1.2x Macro</span>
                    </div>
                    <div className="flex items-center gap-1 text-[#006e2d] font-code-sm text-[11px] font-bold">
                      <span className="material-symbols-outlined text-[16px]">check_circle</span>
                      <span>Geometry Valid</span>
                    </div>
                  </div>

                  {/* Leaf Preview Frame with Optical Reticle Grid & ROI Box */}
                  <div className="relative w-full aspect-[4/3] rounded-lg overflow-hidden bg-[#dae2fd] flex items-center justify-center group select-none">
                    <img
                      src={specimenImage}
                      alt="Active plant specimen"
                      className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                      style={{ transform: `rotate(${rotationDegree}deg)` }}
                    />

                    {/* Optical Reticle Grid & ROI Box */}
                    <div className="absolute inset-0 pointer-events-none p-5 flex flex-col justify-between">
                      <div className="flex justify-between items-start">
                        <div className="bg-[#283044]/80 backdrop-blur-md px-2.5 py-1 rounded text-white font-code-sm text-[11px] flex items-center gap-1.5 shadow-md">
                          <span className="w-1.5 h-1.5 rounded-full bg-[#7cf994] animate-ping"></span>
                          <span>Target ROI: Chlorotic Margin</span>
                        </div>
                        <div className="w-6 h-6 border-t-2 border-r-2 border-[#7ffc97] opacity-80"></div>
                      </div>

                      {/* Center Target Box */}
                      <div className="w-32 h-32 self-center border border-dashed border-[#7ffc97] rounded-lg flex items-end p-1.5 shadow-inner">
                        <span className="font-code-sm text-[10px] text-[#7ffc97] bg-[#283044]/90 px-1.5 py-0.5 rounded">
                          Zone 01 [92%]
                        </span>
                      </div>

                      <div className="flex justify-between items-end">
                        <div className="w-6 h-6 border-b-2 border-l-2 border-[#7ffc97] opacity-80"></div>
                        <div className="font-code-sm text-[11px] bg-[#283044]/80 backdrop-blur-md px-2.5 py-1 rounded text-white shadow-md">
                          RGB Spectrum Mask: ACTIVE
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Image Manipulation Toolbar */}
                  <div className="flex flex-wrap items-center justify-between gap-2 px-1 pt-1">
                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => {
                          // Simple center zoom reset
                          setRotationDegree(0);
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-[#eaedff] text-[#131b2e] font-title-sm text-[12px] transition-colors shadow-xs border border-[#dae2fd]/60 cursor-pointer"
                      >
                        <span className="material-symbols-outlined text-[16px]">crop</span>
                        <span>Crop & Re-center</span>
                      </button>

                      <button
                        onClick={() => setRotationDegree((prev) => (prev + 90) % 360)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-[#eaedff] text-[#131b2e] font-title-sm text-[12px] transition-colors shadow-xs border border-[#dae2fd]/60 cursor-pointer"
                      >
                        <span className="material-symbols-outlined text-[16px]">rotate_right</span>
                        <span>Rotate 90°</span>
                      </button>

                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-[#ffdad6]/50 text-[#ba1a1a] font-title-sm text-[12px] transition-colors shadow-xs border border-[#dae2fd]/60 cursor-pointer"
                      >
                        <span className="material-symbols-outlined text-[16px]">restart_alt</span>
                        <span>Retake</span>
                      </button>
                    </div>

                    <div className="font-code-sm text-[11px] text-[#717970] truncate max-w-[280px]">
                      {fileDetails}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tab Content 2: Live Camera Stream */}
            {activeTab === 'camera' && (
              <div className="space-y-4">
                <div className="relative w-full aspect-[4/3] rounded-xl overflow-hidden bg-[#283044] flex flex-col items-center justify-center p-4 text-center">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`w-full h-full object-cover ${isCameraActive ? 'block' : 'hidden'}`}
                  ></video>

                  {!isCameraActive && (
                    <div className="flex flex-col items-center justify-center text-white p-6">
                      <div className="w-16 h-16 rounded-full bg-white/10 flex items-center justify-center text-[#7cf994] mb-3">
                        <span className="material-symbols-outlined text-[32px]">videocam</span>
                      </div>
                      <h3 className="font-headline-sm text-headline-sm text-white font-bold">
                        Camera Optical Ingestion
                      </h3>
                      <p className="font-body-md text-[13px] text-white/70 max-w-sm mt-1">
                        Direct camera feed with auto-focus edge detection for real-time greenhouse and field leaf stress evaluation.
                      </p>

                      {cameraError && (
                        <p className="font-body-sm text-[12px] text-[#ffb77d] mt-3 bg-white/10 px-3 py-1.5 rounded-lg">
                          {cameraError}
                        </p>
                      )}

                      <button
                        onClick={startCamera}
                        className="mt-4 px-5 py-2.5 rounded-lg bg-[#006e2d] text-white font-title-sm text-[13px] font-bold hover:bg-[#14532d] transition-all cursor-pointer shadow-md"
                      >
                        Request Camera Permission
                      </button>
                    </div>
                  )}

                  {isCameraActive && (
                    <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-4 z-20">
                      <button
                        onClick={capturePhoto}
                        className="px-6 py-2.5 rounded-full bg-[#7cf994] text-[#003b1b] font-title-sm text-[14px] font-extrabold shadow-xl hover:scale-105 transition-transform flex items-center gap-2 cursor-pointer"
                      >
                        <span className="material-symbols-outlined text-[20px]">photo_camera</span>
                        <span>Capture Leaf Snapshot</span>
                      </button>
                      <button
                        onClick={stopCamera}
                        className="px-4 py-2.5 rounded-full bg-black/60 text-white font-title-sm text-[13px] hover:bg-black/80 transition-colors cursor-pointer"
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Specimen Classification Context Selector */}
            <div className="space-y-1.5 pt-1">
              <label className="font-title-sm text-[13px] text-[#131b2e] font-bold flex items-center justify-between">
                <span>Host Plant Taxon (Optional Prior):</span>
                <span className="font-label-sm text-[10px] text-[#006e2d] uppercase font-bold tracking-wider">
                  Refines model weightings
                </span>
              </label>
              <div className="relative">
                <select
                  value={selectedTaxon}
                  onChange={(e) => setSelectedTaxon(e.target.value)}
                  className="w-full h-11 px-3.5 pr-10 bg-[#f2f3ff] rounded-lg text-[#131b2e] font-body-md text-[13px] border border-[#dae2fd] appearance-none focus:outline-none focus:bg-white focus:ring-2 focus:ring-[#006e2d]/30 focus:border-[#006e2d] transition-all cursor-pointer"
                >
                  <option value="Tomato (Solanum lycopersicum) — Solanaceae">
                    Tomato (Solanum lycopersicum) — Solanaceae
                  </option>
                  <option value="Corn / Maize (Zea mays) — Poaceae">Corn / Maize (Zea mays) — Poaceae</option>
                  <option value="Soybean (Glycine max) — Fabaceae">Soybean (Glycine max) — Fabaceae</option>
                  <option value="Bell Pepper (Capsicum annuum) — Solanaceae">
                    Bell Pepper (Capsicum annuum) — Solanaceae
                  </option>
                  <option value="Citrus / Lemon (Citrus limon) — Rutaceae">
                    Citrus / Lemon (Citrus limon) — Rutaceae
                  </option>
                  <option value="Auto-Detect Host Taxon (Zero-Shot Classification)">
                    Auto-Detect Host Taxon (Zero-Shot Classification)
                  </option>
                </select>
                <span className="material-symbols-outlined absolute right-3 top-3 pointer-events-none text-[#717970]">
                  expand_more
                </span>
              </div>
            </div>

            {/* Primary Execution CTA */}
            <div className="pt-2 flex flex-col sm:flex-row items-center gap-3">
              <button
                id="run-analysis-btn"
                disabled={isAnalyzing}
                onClick={handleRunAnalysis}
                className={`w-full sm:flex-1 h-12 bg-[#14532d] hover:bg-[#006e2d] text-white font-title-md text-[15px] font-bold rounded-lg flex items-center justify-center gap-2 shadow-md transition-all cursor-pointer group ${
                  isAnalyzing ? 'opacity-80 pointer-events-none' : ''
                }`}
              >
                <span
                  className={`material-symbols-outlined text-[20px] ${
                    isAnalyzing ? 'animate-spin' : 'group-hover:rotate-12'
                  } transition-transform`}
                >
                  {isAnalyzing ? 'sync' : 'auto_awesome'}
                </span>
                <span>{isAnalyzing ? 'Running Neural Inference...' : 'Run AI Stress Analysis'}</span>
              </button>

              <button
                onClick={onOpenModelConfig}
                className="w-full sm:w-auto h-12 px-5 bg-[#f2f3ff] hover:bg-[#e2e7ff] text-[#131b2e] font-title-sm text-[13px] font-semibold rounded-lg flex items-center justify-center gap-2 transition-colors border border-[#dae2fd] cursor-pointer"
              >
                <span className="material-symbols-outlined text-[18px] text-[#717970]">tune</span>
                <span>Model Config</span>
              </button>
            </div>
          </div>

          {/* In-Page Analysis Progress Indicator / Live Diagnostics Simulator */}
          <div className="bg-white rounded-xl p-5 shadow-sm border border-[#dae2fd]/40 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-[#006e2d] animate-pulse"></span>
                <h2 className="font-title-md text-title-md text-[#131b2e] font-bold">
                  Neural Diagnostic Stream
                </h2>
              </div>
              <span className="font-code-sm text-[11px] px-2.5 py-0.5 rounded-full bg-[#7cf994] text-[#007230] font-bold">
                ResNet50 + ViT-Hybrid
              </span>
            </div>

            {/* Pipeline Processing Stages */}
            <div className="space-y-3.5">
              {/* Stage 1 */}
              <div className="flex items-start gap-3">
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5 font-bold ${
                    analysisStage >= 1
                      ? 'bg-[#14532d] text-white'
                      : 'bg-[#eaedff] text-[#717970]'
                  }`}
                >
                  <span className="material-symbols-outlined text-[14px]">check</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                      Examining visible symptoms
                    </span>
                    <span className="font-code-sm text-[11px] text-[#006e2d] font-bold">
                      {analysisStage >= 1 ? 'COMPLETE' : 'QUEUED'}
                    </span>
                  </div>
                  <p className="font-body-sm text-[12px] text-[#404941]">
                    Adaxial interveinal chlorosis & apical leaf-roll identified.
                  </p>
                </div>
              </div>

              {/* Stage 2 */}
              <div className="flex items-start gap-3">
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5 font-bold ${
                    analysisStage === 2
                      ? 'bg-[#7cf994] text-[#007230] animate-pulse'
                      : analysisStage > 2
                      ? 'bg-[#14532d] text-white'
                      : 'bg-[#eaedff] text-[#717970]'
                  }`}
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {analysisStage > 2 ? 'check' : 'sync'}
                  </span>
                </div>
                <div className="flex-1 min-w-0 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                      Analyzing leaf characteristics (Venation & Chlorophyll Index)
                    </span>
                    <span className="font-code-sm text-[11px] text-[#006e2d] font-bold">
                      {analysisStage > 2 ? 'COMPLETE' : `${stageProgress}%`}
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-[#eaedff] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#006e2d] transition-all duration-500 rounded-full"
                      style={{ width: `${analysisStage > 2 ? 100 : stageProgress}%` }}
                    ></div>
                  </div>
                  <p className="font-body-sm text-[12px] text-[#404941]">
                    SPAD equivalent proxy: 21.4 (Deficit &gt; 35% compared to healthy baseline).
                  </p>
                </div>
              </div>

              {/* Stage 3 */}
              <div className={`flex items-start gap-3 ${analysisStage < 3 ? 'opacity-60' : ''}`}>
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5 font-bold ${
                    analysisStage === 3
                      ? 'bg-[#7cf994] text-[#007230] animate-pulse'
                      : analysisStage > 3
                      ? 'bg-[#14532d] text-white'
                      : 'bg-[#eaedff] text-[#717970]'
                  }`}
                >
                  <span className="font-code-sm text-[11px]">
                    {analysisStage > 3 ? (
                      <span className="material-symbols-outlined text-[14px]">check</span>
                    ) : (
                      '3'
                    )}
                  </span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                      Estimating possible stress conditions
                    </span>
                    <span
                      className={`font-code-sm text-[11px] font-bold ${
                        analysisStage === 3 ? 'text-[#006e2d]' : 'text-[#717970]'
                      }`}
                    >
                      {analysisStage === 3 ? 'PROCESSING' : analysisStage > 3 ? 'COMPLETE' : 'QUEUED'}
                    </span>
                  </div>
                  <p className="font-body-sm text-[12px] text-[#404941]">
                    Cross-referencing abiotic drought vs. fungal early blight signatures.
                  </p>
                </div>
              </div>

              {/* Stage 4 */}
              <div className={`flex items-start gap-3 ${analysisStage < 4 ? 'opacity-60' : ''}`}>
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5 font-bold ${
                    analysisStage >= 4
                      ? 'bg-[#7cf994] text-[#007230]'
                      : 'bg-[#eaedff] text-[#717970]'
                  }`}
                >
                  <span className="font-code-sm text-[11px]">4</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                      Generating agronomic recommendations
                    </span>
                    <span
                      className={`font-code-sm text-[11px] font-bold ${
                        analysisStage >= 4 ? 'text-[#006e2d]' : 'text-[#717970]'
                      }`}
                    >
                      {analysisStage >= 4 ? 'PROCESSING' : 'QUEUED'}
                    </span>
                  </div>
                  <p className="font-body-sm text-[12px] text-[#404941]">
                    Formulating intervention pathways & fertigation mitigation guidance.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Guidelines, Edge Cases, and Test Bank */}
        <div className="lg:col-span-5 space-y-5">
          {/* Optimal Photo Guidelines Card */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-[#e2e7ff] text-[#003b1b] flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined text-[20px]">verified</span>
              </div>
              <div>
                <h2 className="font-title-md text-title-md text-[#131b2e] font-bold">
                  Optimal Capture Protocols
                </h2>
                <span className="font-label-sm text-[11px] text-[#717970]">
                  Maximizes model precision to &gt;96.4%
                </span>
              </div>
            </div>

            {/* Tip Modules */}
            <div className="space-y-2.5">
              <div className="flex items-start gap-3 p-2 rounded-lg hover:bg-[#f2f3ff] transition-colors">
                <span className="material-symbols-outlined text-[#006e2d] text-[20px] mt-0.5 shrink-0">
                  wb_sunny
                </span>
                <div>
                  <p className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                    Diffused Ambient Illumination
                  </p>
                  <p className="font-body-sm text-[12px] text-[#404941]">
                    Avoid direct high-noon solar glare or heavy shadowy occlusions that skew pixel luminance.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-2 rounded-lg hover:bg-[#f2f3ff] transition-colors">
                <span className="material-symbols-outlined text-[#006e2d] text-[20px] mt-0.5 shrink-0">
                  filter_center_focus
                </span>
                <div>
                  <p className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                    Sharp Macro Focal Plane
                  </p>
                  <p className="font-body-sm text-[12px] text-[#404941]">
                    Retain clear definition on tertiary leaf venation, leaf margins, and necrotic lesion boundaries.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-2 rounded-lg hover:bg-[#f2f3ff] transition-colors">
                <span className="material-symbols-outlined text-[#006e2d] text-[20px] mt-0.5 shrink-0">
                  crop_free
                </span>
                <div>
                  <p className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                    Fill Frame (&gt;60% Viewport)
                  </p>
                  <p className="font-body-sm text-[12px] text-[#404941]">
                    Position the symptomatic tissue centrally, avoiding soil, drip tapes, or non-vegetative artifacts.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-2 rounded-lg hover:bg-[#f2f3ff] transition-colors">
                <span className="material-symbols-outlined text-[#006e2d] text-[20px] mt-0.5 shrink-0">
                  flip
                </span>
                <div>
                  <p className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                    Adaxial & Abaxial Inspection
                  </p>
                  <p className="font-body-sm text-[12px] text-[#404941]">
                    Check both upper surface and leaf underside for fungal sporulation or mite colonization.
                  </p>
                </div>
              </div>
            </div>

            {/* Micro Specimen Comparison Gauge */}
            <div className="bg-[#f2f3ff] p-3.5 rounded-xl space-y-2 border border-[#dae2fd]/40">
              <div className="flex justify-between items-center text-[11px] font-bold">
                <span className="text-[#131b2e]">Optical Quality Index</span>
                <span className="text-[#006e2d]">94 / 100 (Optimal)</span>
              </div>
              <div className="w-full h-2 bg-[#eaedff] rounded-full overflow-hidden flex gap-0.5">
                <div className="h-full bg-[#006e2d] rounded-l" style={{ width: '70%' }}></div>
                <div className="h-full bg-[#62df7d]" style={{ width: '24%' }}></div>
                <div className="h-full bg-[#dae2fd]" style={{ width: '6%' }}></div>
              </div>
              <div className="flex justify-between text-[11px] font-code-sm text-[#717970]">
                <span>Luminance: Normal</span>
                <span>Focus: High</span>
                <span>Color Temp: 5400K</span>
              </div>
            </div>
          </div>

          {/* Edge Case Alert Callout Card */}
          <div className="bg-[#ffdcc3]/50 rounded-xl p-5 space-y-2 shadow-xs border border-[#ffb77d]/50">
            <div className="flex items-center gap-2 text-[#703a00]">
              <span className="material-symbols-outlined text-[20px]">warning</span>
              <h3 className="font-title-sm text-[12px] font-bold uppercase tracking-wider">
                Agronomic Diagnostic Notice
              </h3>
            </div>
            <p className="font-body-sm text-[12px] text-[#131b2e] leading-relaxed">
              <strong className="font-bold text-[#4f2700]">Notice on Ambiguous Symptomatology:</strong>{' '}
              Overlapping physical manifestations (e.g., osmotic drought stress versus early Fusarium vascular wilt) frequently exhibit reciprocal chlorotic cues. The inference pipeline computes multi-class posterior probability distributions rather than deterministic single diagnoses.
            </p>
          </div>

          {/* Quick Reference Specimen Test Bank */}
          <div className="bg-white rounded-xl p-5 md:p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-title-md text-title-md text-[#131b2e] font-bold">
                Reference Specimen Library
              </h3>
              <button
                onClick={() => onNavigate('plant-stress-library')}
                className="font-label-sm text-[11px] text-[#006e2d] font-bold cursor-pointer hover:underline"
              >
                View Full Atlas
              </button>
            </div>
            <p className="font-body-sm text-[12px] text-[#717970]">
              Test the vision pipeline instantly using validated field specimens:
            </p>

            <div className="grid grid-cols-2 gap-2.5">
              {REFERENCE_SPECIMENS.map((specimen) => (
                <button
                  key={specimen.id}
                  id={`ref-specimen-${specimen.id}`}
                  onClick={() => loadReferenceSpecimen(specimen)}
                  className="flex flex-col p-2 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] transition-all text-left group border border-[#dae2fd]/40 hover:border-[#006e2d]/40 cursor-pointer"
                >
                  <div className="w-full aspect-[16/10] rounded-lg overflow-hidden bg-[#dae2fd] mb-2">
                    <img
                      src={specimen.imageUrl}
                      alt={specimen.title}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                    />
                  </div>
                  <span className="font-title-sm text-[12px] text-[#131b2e] truncate font-bold">
                    {specimen.title}
                  </span>
                  <span
                    className={`font-code-sm text-[10px] font-bold truncate ${
                      specimen.categoryType === 'Abiotic'
                        ? 'text-[#703a00]'
                        : specimen.categoryType === 'Fungal'
                        ? 'text-[#ba1a1a]'
                        : specimen.categoryType === 'Deficiency'
                        ? 'text-[#703a00]'
                        : 'text-[#006e2d]'
                    }`}
                  >
                    {specimen.category}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
