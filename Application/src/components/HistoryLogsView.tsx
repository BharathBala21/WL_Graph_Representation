import React, { useState } from 'react';
import { PageView, InspectionRecord } from '../types';
import { RECENT_INSPECTIONS } from '../data/mockData';

interface HistoryLogsViewProps {
  onNavigate: (page: PageView) => void;
  onSelectInspection: (record: InspectionRecord) => void;
  onExportCSV: () => void;
}

export const HistoryLogsView: React.FC<HistoryLogsViewProps> = ({
  onNavigate,
  onSelectInspection,
  onExportCSV,
}) => {
  const [filterSeverity, setFilterSeverity] = useState<string>('All');
  const [searchFilter, setSearchFilter] = useState<string>('');

  const filteredRecords = RECENT_INSPECTIONS.filter((record) => {
    const matchesSeverity = filterSeverity === 'All' || record.severity === filterSeverity;
    const matchesSearch =
      record.specimenName.toLowerCase().includes(searchFilter.toLowerCase()) ||
      record.scientificName.toLowerCase().includes(searchFilter.toLowerCase()) ||
      record.id.toLowerCase().includes(searchFilter.toLowerCase()) ||
      record.tagId.toLowerCase().includes(searchFilter.toLowerCase());
    return matchesSeverity && matchesSearch;
  });

  return (
    <div id="history-logs-view" className="max-w-[1440px] w-full mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div className="bg-white rounded-xl p-6 md:p-8 shadow-sm border border-[#dae2fd]/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="font-label-sm text-[11px] px-2.5 py-0.5 rounded-full bg-[#7cf994] text-[#007230] font-bold uppercase tracking-wider">
              Diagnostic Audit Trail
            </span>
            <span className="font-code-sm text-[12px] text-[#717970]">
              {RECENT_INSPECTIONS.length} Logged Specimens
            </span>
          </div>
          <h1 className="font-headline-lg text-[28px] md:text-[32px] text-[#003b1b] font-extrabold tracking-tight">
            Analysis History & Diagnostic Logs
          </h1>
          <p className="font-body-md text-[14px] text-[#404941] max-w-2xl">
            Immutable log of all specimen scans, confidence metrics, and detected stress signatures over time.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onExportCSV}
            className="px-4 py-2.5 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[13px] font-bold shadow-sm flex items-center gap-2 cursor-pointer transition-all"
          >
            <span className="material-symbols-outlined text-[18px]">file_download</span>
            <span>Export Complete CSV</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white rounded-xl p-4 shadow-sm border border-[#dae2fd]/40 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 overflow-x-auto w-full sm:w-auto">
          {['All', 'Optimal', 'Mild', 'Moderate'].map((sev) => (
            <button
              key={sev}
              onClick={() => setFilterSeverity(sev)}
              className={`px-3 py-1.5 rounded-lg font-label-sm text-[11px] font-bold cursor-pointer transition-colors ${
                filterSeverity === sev
                  ? 'bg-[#14532d] text-white'
                  : 'bg-[#f2f3ff] text-[#404941] hover:bg-[#eaedff]'
              }`}
            >
              {sev} Severity
            </button>
          ))}
        </div>

        <div className="relative w-full sm:w-80">
          <span className="material-symbols-outlined absolute left-3 top-2.5 text-[#717970] text-[18px]">
            search
          </span>
          <input
            type="text"
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            placeholder="Search specimen, tag ID, or crop..."
            className="w-full h-9 pl-9 pr-4 bg-[#f2f3ff] rounded-lg text-[#131b2e] font-body-sm text-[12px] border border-[#dae2fd] focus:outline-none focus:bg-white"
          />
        </div>
      </div>

      {/* Logs Table */}
      <div className="bg-white rounded-xl shadow-sm border border-[#dae2fd]/40 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-[#f2f3ff] border-b border-[#dae2fd] text-[11px] font-label-sm text-[#717970] uppercase tracking-wider">
                <th className="p-3.5 pl-5">Specimen / Image</th>
                <th className="p-3.5">Taxon & Tag</th>
                <th className="p-3.5">Assessment Status</th>
                <th className="p-3.5">Confidence</th>
                <th className="p-3.5">Indices (NDVI / VARI)</th>
                <th className="p-3.5">Timestamp</th>
                <th className="p-3.5 pr-5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#dae2fd]/40 text-[13px] font-body-sm">
              {filteredRecords.map((record) => (
                <tr
                  key={record.id}
                  onClick={() => onSelectInspection(record)}
                  className="hover:bg-[#f2f3ff]/70 transition-colors cursor-pointer group"
                >
                  <td className="p-3.5 pl-5">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 rounded-lg overflow-hidden bg-[#dae2fd] shrink-0">
                        <img
                          src={record.imageUrl}
                          alt={record.specimenName}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                        />
                      </div>
                      <div>
                        <span className="font-title-sm text-[13px] font-bold text-[#131b2e] group-hover:text-[#003b1b]">
                          {record.specimenName}
                        </span>
                        <p className="font-code-sm text-[11px] text-[#717970]">{record.id}</p>
                      </div>
                    </div>
                  </td>

                  <td className="p-3.5">
                    <span className="font-body-sm text-[12px] italic text-[#404941]">
                      {record.scientificName}
                    </span>
                    <span className="block font-code-sm text-[11px] text-[#717970]">
                      {record.quadrant || 'Plot Block'}
                    </span>
                  </td>

                  <td className="p-3.5">
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-label-sm text-[11px] font-bold ${
                        record.statusType === 'water-stress'
                          ? 'bg-[#ffdcc3] text-[#2f1500]'
                          : record.statusType === 'healthy'
                          ? 'bg-[#7cf994] text-[#007230]'
                          : 'bg-[#dae2fd] text-[#404941]'
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          record.statusType === 'water-stress'
                            ? 'bg-[#4f2700]'
                            : record.statusType === 'healthy'
                            ? 'bg-[#006e2d]'
                            : 'bg-[#717970]'
                        }`}
                      ></span>
                      {record.statusText}
                    </span>
                  </td>

                  <td className="p-3.5">
                    <div className="flex items-center gap-2">
                      <div className="w-14 h-1.5 bg-[#dae2fd] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#006e2d] rounded-full"
                          style={{ width: `${record.confidence}%` }}
                        ></div>
                      </div>
                      <span className="font-code-sm text-[12px] font-bold text-[#131b2e]">
                        {record.confidence}%
                      </span>
                    </div>
                  </td>

                  <td className="p-3.5 font-code-sm text-[11px] text-[#404941]">
                    <div>NDVI: {record.ndvi || 0.72}</div>
                    <div>VARI: {record.vari || 0.54}</div>
                  </td>

                  <td className="p-3.5 font-code-sm text-[11px] text-[#717970]">
                    {record.dateStr}
                  </td>

                  <td className="p-3.5 pr-5 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectInspection(record);
                      }}
                      className="px-3 py-1 rounded-lg bg-[#eaedff] text-[#003b1b] font-title-sm text-[11px] font-bold hover:bg-[#dae2fd] transition-colors"
                    >
                      View Report
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
