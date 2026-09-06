import React, { useState } from 'react';
import { PageView } from '../types';

interface SettingsViewProps {
  onNavigate: (page: PageView) => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({ onNavigate }) => {
  const [confidenceThreshold, setConfidenceThreshold] = useState(75);
  const [autoCrop, setAutoCrop] = useState(true);
  const [syncInterval, setSyncInterval] = useState('1 hour');
  const [emailAlerts, setEmailAlerts] = useState(true);
  const [showSavedToast, setShowSavedToast] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setShowSavedToast(true);
    setTimeout(() => setShowSavedToast(false), 2500);
  };

  return (
    <div id="settings-view" className="max-w-[1440px] w-full mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div className="bg-white rounded-xl p-6 md:p-8 shadow-sm border border-[#dae2fd]/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="font-label-sm text-[11px] px-2.5 py-0.5 rounded-full bg-[#7cf994] text-[#007230] font-bold uppercase tracking-wider">
              Agronomy Preferences
            </span>
            <span className="font-code-sm text-[12px] text-[#717970]">Profile: Dr. Sarah Jenkins</span>
          </div>
          <h1 className="font-headline-lg text-[28px] md:text-[32px] text-[#003b1b] font-extrabold tracking-tight">
            System & Optical Calibration Settings
          </h1>
          <p className="font-body-md text-[14px] text-[#404941] max-w-2xl">
            Configure inference confidence gates, sensor telemetry polling cadence, and automated reporting protocols.
          </p>
        </div>

        {showSavedToast && (
          <div className="px-4 py-2 rounded-lg bg-[#7cf994] text-[#007230] font-title-sm text-[13px] font-bold shadow-md flex items-center gap-2 animate-in fade-in">
            <span className="material-symbols-outlined text-[18px]">check_circle</span>
            <span>Settings Updated Successfully</span>
          </div>
        )}
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Section 1: Inference & Detection Gates */}
          <div className="bg-white rounded-xl p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
            <div className="flex items-center gap-2 border-b border-[#dae2fd]/40 pb-3">
              <span className="material-symbols-outlined text-[#006e2d] text-[22px]">tune</span>
              <h2 className="font-headline-sm text-[18px] text-[#131b2e] font-bold">
                Inference Sensitivity & Filters
              </h2>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between items-center text-[13px] font-title-sm">
                  <span className="text-[#131b2e]">Minimum Diagnostic Confidence Gate</span>
                  <span className="font-code-sm text-[#006e2d] font-bold">{confidenceThreshold}%</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="95"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                  className="w-full h-2 mt-2 bg-[#eaedff] rounded-lg appearance-none cursor-pointer accent-[#006e2d]"
                />
                <p className="font-body-sm text-[11px] text-[#717970] mt-1">
                  Predictions below this percentage will be marked as "Ambiguous / Inconclusive" and flag manual hand-lens verification.
                </p>
              </div>

              <div className="flex items-center justify-between pt-2">
                <div>
                  <span className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                    Auto-Segmentation & Soil Masking
                  </span>
                  <p className="font-body-sm text-[11px] text-[#717970]">
                    Automatically strips pots, ground, and background shadows via U-Net.
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={autoCrop}
                  onChange={(e) => setAutoCrop(e.target.checked)}
                  className="w-5 h-5 rounded text-[#006e2d] focus:ring-[#006e2d] accent-[#006e2d] cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Section 2: Sensor Fleet & Telemetry Cadence */}
          <div className="bg-white rounded-xl p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
            <div className="flex items-center gap-2 border-b border-[#dae2fd]/40 pb-3">
              <span className="material-symbols-outlined text-[#003b1b] text-[22px]">sensors</span>
              <h2 className="font-headline-sm text-[18px] text-[#131b2e] font-bold">
                Telemetry & Notification Sync
              </h2>
            </div>

            <div className="space-y-3">
              <div>
                <label className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                  Sensor Polling Cadence
                </label>
                <select
                  value={syncInterval}
                  onChange={(e) => setSyncInterval(e.target.value)}
                  className="w-full h-10 px-3 mt-1.5 rounded-lg bg-[#f2f3ff] text-[#131b2e] text-[13px] border border-[#dae2fd] focus:outline-none focus:bg-white"
                >
                  <option value="15 mins">Every 15 minutes (High Power / Real-time)</option>
                  <option value="1 hour">Every 1 hour (Standard Balanced)</option>
                  <option value="6 hours">Every 6 hours (Solar / Battery Conservative)</option>
                  <option value="24 hours">Daily Summary</option>
                </select>
              </div>

              <div className="flex items-center justify-between pt-2">
                <div>
                  <span className="font-title-sm text-[13px] text-[#131b2e] font-semibold">
                    Immediate Pathogen & Moisture Deficit Alerts
                  </span>
                  <p className="font-body-sm text-[11px] text-[#717970]">
                    Send push notification when severe stress (&gt;80%) is registered.
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={emailAlerts}
                  onChange={(e) => setEmailAlerts(e.target.checked)}
                  className="w-5 h-5 rounded text-[#006e2d] focus:ring-[#006e2d] accent-[#006e2d] cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Action Save Button */}
        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={() => onNavigate('dashboard')}
            className="px-5 py-2.5 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] text-[#131b2e] font-title-sm text-[13px] font-bold transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="px-6 py-2.5 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[13px] font-bold shadow-sm transition-all cursor-pointer"
          >
            Save Configuration
          </button>
        </div>
      </form>
    </div>
  );
};
