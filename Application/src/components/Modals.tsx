import React, { useState } from 'react';
import { InspectionRecord } from '../types';

interface BatchScanModalProps {
  isOpen: boolean;
  onClose: () => void;
  onStartBatch: (count: number) => void;
}

export const BatchScanModal: React.FC<BatchScanModalProps> = ({ isOpen, onClose, onStartBatch }) => {
  const [selectedFolder, setSelectedFolder] = useState('Greenhouse_Quadrant_B_Oct24');
  const [fileCount, setFileCount] = useState(14);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);

  if (!isOpen) return null;

  const handleStart = () => {
    setIsProcessing(true);
    let p = 0;
    const interval = setInterval(() => {
      p += 15;
      if (p >= 100) {
        clearInterval(interval);
        setTimeout(() => {
          setIsProcessing(false);
          onStartBatch(fileCount);
          onClose();
        }, 500);
      } else {
        setProgress(p);
      }
    }, 200);
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-xs z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-lg w-full p-6 space-y-5 shadow-2xl animate-in fade-in">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[#006e2d] text-[24px]">folder_zip</span>
            <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
              Batch Specimen Ingestion
            </h2>
          </div>
          <button onClick={onClose} className="text-[#717970] hover:text-[#131b2e] cursor-pointer">
            <span className="material-symbols-outlined text-[24px]">close</span>
          </button>
        </div>

        <p className="font-body-md text-[13px] text-[#404941]">
          Ingest an entire folder or ZIP archive of high-resolution crop foliar photos to run parallel classification and generate a consolidated CSV audit.
        </p>

        <div className="space-y-3 bg-[#f2f3ff] p-4 rounded-xl border border-[#dae2fd]/60">
          <div className="flex justify-between text-[12px] font-title-sm">
            <span className="text-[#131b2e] font-semibold">Active Ingestion Batch:</span>
            <span className="font-code-sm text-[#006e2d] font-bold">{fileCount} Specimens</span>
          </div>
          <div className="font-code-sm text-[12px] text-[#404941] bg-white p-2.5 rounded-lg border border-[#dae2fd]">
            📁 /{selectedFolder} (*.jpg, *.png)
          </div>

          {isProcessing && (
            <div className="space-y-1.5 pt-2">
              <div className="flex justify-between text-[11px] font-code-sm text-[#006e2d] font-bold">
                <span>Inferencing ResNet50-ViT Head...</span>
                <span>{progress}%</span>
              </div>
              <div className="w-full h-2 bg-[#eaedff] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#006e2d] transition-all duration-200 rounded-full"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <button
            onClick={onClose}
            disabled={isProcessing}
            className="px-4 py-2 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] text-[#131b2e] font-title-sm text-[13px] font-bold cursor-pointer"
          >
            Cancel
          </button>
          <button
            onClick={handleStart}
            disabled={isProcessing}
            className="px-5 py-2.5 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[13px] font-bold flex items-center gap-2 cursor-pointer shadow-sm"
          >
            <span className="material-symbols-outlined text-[18px]">play_arrow</span>
            <span>{isProcessing ? 'Processing...' : 'Start Batch Analysis'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};

interface CompareScansModalProps {
  isOpen: boolean;
  onClose: () => void;
  activeInspection?: InspectionRecord | null;
}

export const CompareScansModal: React.FC<CompareScansModalProps> = ({
  isOpen,
  onClose,
  activeInspection,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-xs z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-4xl w-full p-6 space-y-5 shadow-2xl max-h-[90vh] overflow-y-auto animate-in fade-in">
        <div className="flex items-center justify-between border-b border-[#dae2fd]/60 pb-3">
          <div>
            <span className="font-label-sm text-[11px] px-2.5 py-0.5 rounded-full bg-[#7cf994] text-[#007230] font-bold">
              Comparative Morphometry
            </span>
            <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold mt-1">
              Specimen Comparison: Baseline vs. Active Stress
            </h2>
          </div>
          <button onClick={onClose} className="text-[#717970] hover:text-[#131b2e] cursor-pointer">
            <span className="material-symbols-outlined text-[24px]">close</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Baseline Specimen */}
          <div className="space-y-3 bg-[#f2f3ff] p-4 rounded-xl border border-[#dae2fd]/60">
            <div className="flex items-center justify-between">
              <span className="font-title-sm text-[13px] text-[#006e2d] font-bold">
                Healthy Reference Baseline
              </span>
              <span className="font-code-sm text-[11px] text-[#717970]">Turgor: Optimal</span>
            </div>
            <div className="w-full aspect-[4/3] rounded-lg overflow-hidden bg-[#dae2fd]">
              <img
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCdpYDMe8gUrarNRBhCaQY9M6HOOdVZ96SKJsJ6QVoHrfkSbQYlyn5urhIEpIKrmeOv9-V_31yNeH63SGAM-BWlL9BI3vFM50ezYIHEKD7rwpL-q3WWfNH42L31_ljGRPi2TKTc__6X_TGPIxve0aDyZdXIKE5Q5TxumbrSDXMJt0O8bXwFqcOSGNgtzlwHMfFqZLbeGlBPdnx0TEW5_7F04fM1tvQssq5nZsfYKUh5LUe4zRaNdtP5"
                alt="Healthy control"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="space-y-1 text-[12px] font-body-sm text-[#404941]">
              <p>• NDVI Index: <strong>0.892</strong> (High photosynthetic capacity)</p>
              <p>• SPAD Chlorophyll: <strong>46.4</strong> (Uniform leaf greening)</p>
              <p>• Leaf Margin: <strong>Smooth, flat plane without curling</strong></p>
            </div>
          </div>

          {/* Current Scanned Specimen */}
          <div className="space-y-3 bg-[#ffdcc3]/30 p-4 rounded-xl border border-[#ffb77d]/60">
            <div className="flex items-center justify-between">
              <span className="font-title-sm text-[13px] text-[#703a00] font-bold">
                Active Specimen (#PCAI-98421)
              </span>
              <span className="font-code-sm text-[11px] text-[#703a00] font-bold">Water Stress (82%)</span>
            </div>
            <div className="w-full aspect-[4/3] rounded-lg overflow-hidden bg-[#dae2fd]">
              <img
                src={
                  activeInspection?.imageUrl ||
                  'https://lh3.googleusercontent.com/aida-public/AB6AXuBMiDgfnnLYcUUsdxm7Exw-SA6swXGrlE_dD_I073_e0JhtESnSIAixpEEScG0-R5_EZV1qDKXOg8zEC4LGYOoPYSqgY8waadfiUKD3PYFT_3Ydwua8Nlir_htUuEgJt-RDyYdeV-n3eFa4SefXDprGJ_tpO858GG_YnHGUyxNX_ltacFHKMNmdim3SaL9B_YWO4gO40hxtKKvoxv1LYSYvZqtV2n4YBAfDSLsIBGlHxXBc0A9B56cR'
                }
                alt="Active specimen"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="space-y-1 text-[12px] font-body-sm text-[#404941]">
              <p>• NDVI Index: <strong className="text-[#ba1a1a]">0.684 (-23% vs baseline)</strong></p>
              <p>• SPAD Chlorophyll: <strong className="text-[#ba1a1a]">28.1 (Deficit chlorosis)</strong></p>
              <p>• Leaf Margin: <strong className="text-[#703a00]">Bilateral inward curl & apical wilting</strong></p>
            </div>
          </div>
        </div>

        <div className="flex justify-end pt-2 border-t border-[#dae2fd]">
          <button
            onClick={onClose}
            className="px-5 py-2.5 rounded-lg bg-[#003b1b] text-white font-title-sm text-[13px] font-bold cursor-pointer"
          >
            Done Comparing
          </button>
        </div>
      </div>
    </div>
  );
};

interface ShareWithAgronomistModalProps {
  isOpen: boolean;
  onClose: () => void;
  specimenId?: string;
}

export const ShareWithAgronomistModal: React.FC<ShareWithAgronomistModalProps> = ({
  isOpen,
  onClose,
  specimenId = 'PCAI-98421',
}) => {
  const [recipient, setRecipient] = useState('extension-agronomist@ucdavis.edu');
  const [notes, setNotes] = useState(
    'Please review foliar chlorosis patterns on Field Tomato #04. Suspected root moisture deficit vs early vascular wilt.'
  );
  const [copied, setCopied] = useState(false);
  const [sent, setSent] = useState(false);

  if (!isOpen) return null;

  const handleCopyLink = () => {
    navigator.clipboard?.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    setSent(true);
    setTimeout(() => {
      setSent(false);
      onClose();
    }, 1500);
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-xs z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-lg w-full p-6 space-y-5 shadow-2xl animate-in fade-in">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[#003b1b] text-[24px]">share</span>
            <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
              Share Diagnostic with Agronomist
            </h2>
          </div>
          <button onClick={onClose} className="text-[#717970] hover:text-[#131b2e] cursor-pointer">
            <span className="material-symbols-outlined text-[24px]">close</span>
          </button>
        </div>

        <form onSubmit={handleSend} className="space-y-4">
          <div>
            <label className="font-title-sm text-[12px] text-[#131b2e] font-semibold">
              Recipient Email / Agronomy Extension
            </label>
            <input
              type="email"
              required
              value={recipient}
              onChange={(e) => setRecipient(e.target.value)}
              className="w-full h-10 px-3 mt-1 rounded-lg bg-[#f2f3ff] text-[#131b2e] text-[13px] border border-[#dae2fd] focus:outline-none focus:bg-white"
            />
          </div>

          <div>
            <label className="font-title-sm text-[12px] text-[#131b2e] font-semibold">
              Clinical Observations & Notes
            </label>
            <textarea
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full p-3 mt-1 rounded-lg bg-[#f2f3ff] text-[#131b2e] text-[12px] border border-[#dae2fd] focus:outline-none focus:bg-white resize-none"
            />
          </div>

          <div className="p-3 bg-[#f2f3ff] rounded-xl flex items-center justify-between border border-[#dae2fd]/60">
            <div className="font-code-sm text-[11px] text-[#404941] truncate max-w-[280px]">
              https://plantcare.ai/r/{specimenId}
            </div>
            <button
              type="button"
              onClick={handleCopyLink}
              className="px-3 py-1 bg-white rounded-md text-[11px] font-title-sm text-[#006e2d] font-bold border border-[#dae2fd] hover:bg-[#eaedff] cursor-pointer"
            >
              {copied ? 'Copied!' : 'Copy Link'}
            </button>
          </div>

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-[#f2f3ff] text-[#131b2e] font-title-sm text-[12px] font-bold cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-5 py-2.5 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[12px] font-bold flex items-center gap-1.5 cursor-pointer shadow-sm"
            >
              <span className="material-symbols-outlined text-[16px]">send</span>
              <span>{sent ? 'Dispatched!' : 'Send Report'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

interface ModelConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ModelConfigModal: React.FC<ModelConfigModalProps> = ({ isOpen, onClose }) => {
  const [selectedArch, setSelectedArch] = useState('ResNet50 + ViT-B/16 Hybrid');
  const [temperature, setTemperature] = useState(0.7);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-xs z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-md w-full p-6 space-y-5 shadow-2xl animate-in fade-in">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[#006e2d] text-[24px]">tune</span>
            <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
              Neural Inference Engine Config
            </h2>
          </div>
          <button onClick={onClose} className="text-[#717970] hover:text-[#131b2e] cursor-pointer">
            <span className="material-symbols-outlined text-[24px]">close</span>
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="font-title-sm text-[12px] text-[#131b2e] font-semibold">
              Deep Neural Architecture Backend
            </label>
            <select
              value={selectedArch}
              onChange={(e) => setSelectedArch(e.target.value)}
              className="w-full h-10 px-3 mt-1.5 rounded-lg bg-[#f2f3ff] text-[#131b2e] text-[13px] border border-[#dae2fd] focus:outline-none focus:bg-white"
            >
              <option value="ResNet50 + ViT-B/16 Hybrid">ResNet-50 + ViT-B/16 (Highest Precision)</option>
              <option value="EfficientNet-B4">EfficientNet-B4 (Balanced Micro-Symptoms)</option>
              <option value="MobileNetV3 Edge">MobileNetV3 Edge (Low-Power Field Deployment)</option>
            </select>
          </div>

          <div>
            <div className="flex justify-between items-center text-[12px] font-title-sm">
              <span className="text-[#131b2e]">Softmax Temperature Scaling (T)</span>
              <span className="font-code-sm text-[#006e2d] font-bold">{temperature}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full h-2 mt-2 bg-[#eaedff] rounded-lg appearance-none cursor-pointer accent-[#006e2d]"
            />
            <p className="font-body-sm text-[11px] text-[#717970] mt-1">
              Controls dispersion of candidate stress predictions in the posterior distribution.
            </p>
          </div>
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <button
            onClick={onClose}
            className="px-5 py-2.5 rounded-lg bg-[#003b1b] text-white font-title-sm text-[13px] font-bold cursor-pointer"
          >
            Apply Config
          </button>
        </div>
      </div>
    </div>
  );
};
