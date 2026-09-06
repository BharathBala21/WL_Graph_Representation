import React from 'react';
import { PageView } from '../types';
import { ML_PIPELINE_STAGES } from '../data/mockData';

interface MLPipelineViewProps {
  onNavigate: (page: PageView) => void;
  onExportPDF: () => void;
}

export const MLPipelineView: React.FC<MLPipelineViewProps> = ({ onNavigate, onExportPDF }) => {
  return (
    <div id="ml-pipeline-view" className="max-w-[1440px] w-full mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div className="bg-white rounded-xl p-6 md:p-8 shadow-sm border border-[#dae2fd]/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="font-label-sm text-[11px] px-2.5 py-0.5 rounded-full bg-[#7cf994] text-[#007230] font-bold uppercase tracking-wider">
              Neural Architecture Card
            </span>
            <span className="font-code-sm text-[12px] text-[#717970]">v2.4.8-PROD</span>
          </div>
          <h1 className="font-headline-lg text-[28px] md:text-[32px] text-[#003b1b] font-extrabold tracking-tight">
            Dual CNN + Vision Transformer Pipeline
          </h1>
          <p className="font-body-md text-[14px] text-[#404941] max-w-2xl">
            Synergistic multi-scale deep learning model pairing convolutional inductive bias with global self-attention mechanisms for botanical stress classification.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onExportPDF}
            className="px-4 py-2.5 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[13px] font-bold shadow-sm flex items-center gap-2 cursor-pointer transition-all"
          >
            <span className="material-symbols-outlined text-[18px]">download</span>
            <span>Download Transparency Card</span>
          </button>
        </div>
      </div>

      {/* Model Performance KPI Ribbon */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-xl border border-[#dae2fd]/40 shadow-xs">
          <span className="font-label-sm text-[11px] text-[#717970] uppercase font-bold">
            Validation Accuracy
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="font-headline-md text-[28px] text-[#006e2d] font-extrabold">96.4%</span>
            <span className="font-code-sm text-[12px] text-[#006e2d]">Top-1</span>
          </div>
          <span className="font-body-sm text-[11px] text-[#717970] mt-0.5">Across 48,000 foliar test samples</span>
        </div>

        <div className="bg-white p-5 rounded-xl border border-[#dae2fd]/40 shadow-xs">
          <span className="font-label-sm text-[11px] text-[#717970] uppercase font-bold">
            Macro F1-Score
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="font-headline-md text-[28px] text-[#003b1b] font-extrabold">0.948</span>
            <span className="font-code-sm text-[12px] text-[#717970]">Balanced</span>
          </div>
          <span className="font-body-sm text-[11px] text-[#717970] mt-0.5">Weighted for rare fungal pustules</span>
        </div>

        <div className="bg-white p-5 rounded-xl border border-[#dae2fd]/40 shadow-xs">
          <span className="font-label-sm text-[11px] text-[#717970] uppercase font-bold">
            Inference Latency
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="font-headline-md text-[28px] text-[#131b2e] font-extrabold">320 ms</span>
            <span className="font-code-sm text-[12px] text-[#006e2d]">TensorRT</span>
          </div>
          <span className="font-body-sm text-[11px] text-[#717970] mt-0.5">FP16 precision runtime</span>
        </div>

        <div className="bg-white p-5 rounded-xl border border-[#dae2fd]/40 shadow-xs">
          <span className="font-label-sm text-[11px] text-[#717970] uppercase font-bold">
            Taxa Mapped
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="font-headline-md text-[28px] text-[#703a00] font-extrabold">48 Crops</span>
            <span className="font-code-sm text-[12px] text-[#717970]">12 Families</span>
          </div>
          <span className="font-body-sm text-[11px] text-[#717970] mt-0.5">Includes Solanaceae, Poaceae, Fabaceae</span>
        </div>
      </div>

      {/* 6-Stage Deep Architecture Detailed Stack */}
      <div className="bg-white rounded-xl p-6 md:p-8 shadow-sm border border-[#dae2fd]/40 space-y-5">
        <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
          Six-Stage Agronomic Inference Pipeline
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {ML_PIPELINE_STAGES.map((stage) => (
            <div
              key={stage.stageNumber}
              className="bg-[#f2f3ff] p-5 rounded-xl border border-[#dae2fd]/60 flex flex-col justify-between space-y-3"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-code-sm text-[11px] text-[#006e2d] font-bold">
                    {stage.stageNumber}
                  </span>
                  <div className="w-8 h-8 rounded-lg bg-white flex items-center justify-center text-[#003b1b] shadow-xs">
                    <span className="material-symbols-outlined text-[18px]">{stage.icon}</span>
                  </div>
                </div>
                <h3 className="font-title-md text-[15px] text-[#131b2e] font-bold mt-2">
                  {stage.name}
                </h3>
                <p className="font-body-sm text-[12px] text-[#404941] mt-1 leading-relaxed">
                  {stage.description}
                </p>
              </div>

              <div className="pt-2 border-t border-[#dae2fd]/40 flex items-center justify-between text-[11px] font-code-sm text-[#717970]">
                <span>Status: VERIFIED</span>
                <span className="text-[#006e2d] font-bold">Passed ISO Test</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Technical Methodology & Uncertainty Calibration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[#006e2d] text-[22px]">layers</span>
            <h3 className="font-title-md text-title-md text-[#131b2e] font-bold">
              Grad-CAM Visual Attention Mechanism
            </h3>
          </div>
          <p className="font-body-sm text-[13px] text-[#404941] leading-relaxed">
            Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the leaf image for predicting the condition.
          </p>
          <div className="p-3.5 rounded-lg bg-[#f2f3ff] border border-[#dae2fd] text-[12px] font-code-sm text-[#003b1b]">
            Heatmap Activation: L_c^Grad-CAM = ReLU(∑ α_k^c * A^k)
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-sm border border-[#dae2fd]/40 space-y-4">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[#703a00] text-[22px]">analytics</span>
            <h3 className="font-title-md text-title-md text-[#131b2e] font-bold">
              Bayesian Uncertainty & Softmax Calibration
            </h3>
          </div>
          <p className="font-body-sm text-[13px] text-[#404941] leading-relaxed">
            Raw deep neural net logits are temperature-scaled using Platt calibration against validation holdouts. Ambiguous leaf photos produce diffuse probability mass instead of overconfident erroneous verdicts, prompting secondary physical inspection.
          </p>
          <div className="p-3.5 rounded-lg bg-[#f2f3ff] border border-[#dae2fd] text-[12px] font-code-sm text-[#703a00]">
            Expected Calibration Error (ECE): 1.84% (Ideal &lt; 2.5%)
          </div>
        </div>
      </div>
    </div>
  );
};
