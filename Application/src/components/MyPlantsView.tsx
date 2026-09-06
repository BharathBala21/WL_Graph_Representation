import React, { useState } from 'react';
import { PageView, MonitoredPlant } from '../types';
import { MONITORED_PLANTS } from '../data/mockData';

interface MyPlantsViewProps {
  onNavigate: (page: PageView) => void;
  onScanPlant: (plant: MonitoredPlant) => void;
}

export const MyPlantsView: React.FC<MyPlantsViewProps> = ({ onNavigate, onScanPlant }) => {
  const [plants, setPlants] = useState<MonitoredPlant[]>(MONITORED_PLANTS);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newPlantName, setNewPlantName] = useState('');
  const [newPlantSpecies, setNewPlantSpecies] = useState('');
  const [newPlantLocation, setNewPlantLocation] = useState('');

  const handleAddPlant = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPlantName.trim()) return;
    const newPlant: MonitoredPlant = {
      id: `plant-${Date.now()}`,
      name: newPlantName,
      scientificName: newPlantSpecies || 'Cultivar hybrid',
      icon: 'psychiatry',
      status: 'Optimal',
      statusTrend: 'Stable',
      trendIcon: 'horizontal_rule',
      metricLabel: 'Canopy Index',
      metricValue: '0.82 NDVI',
      lastScanned: 'Just now',
      quadrant: newPlantLocation || 'Block E-01',
      healthRatio: 92,
    };
    setPlants([newPlant, ...plants]);
    setShowAddModal(false);
    setNewPlantName('');
    setNewPlantSpecies('');
    setNewPlantLocation('');
  };

  return (
    <div id="my-plants-view" className="max-w-[1440px] w-full mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div className="bg-white rounded-xl p-6 md:p-8 shadow-sm border border-[#dae2fd]/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="font-label-sm text-[11px] px-2.5 py-0.5 rounded-full bg-[#7cf994] text-[#007230] font-bold uppercase tracking-wider">
              Telemetry Tracking
            </span>
            <span className="font-code-sm text-[12px] text-[#717970]">{plants.length} Active Crops</span>
          </div>
          <h1 className="font-headline-lg text-[28px] md:text-[32px] text-[#003b1b] font-extrabold tracking-tight">
            My Monitored Plants & Plots
          </h1>
          <p className="font-body-md text-[14px] text-[#404941] max-w-2xl">
            Continuous optical & hydration monitoring across your greenhouse bays, hydroponic rows, and open field quadrants.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowAddModal(true)}
            className="px-4 py-2.5 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[13px] font-bold shadow-sm flex items-center gap-2 cursor-pointer transition-all"
          >
            <span className="material-symbols-outlined text-[18px]">add</span>
            <span>Add New Crop</span>
          </button>
        </div>
      </div>

      {/* Grid of Monitored Plants */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {plants.map((plant) => (
          <div
            key={plant.id}
            id={`monitored-plant-${plant.id}`}
            className="bg-white rounded-xl p-5 shadow-sm border border-[#dae2fd]/40 flex flex-col justify-between hover:shadow-md hover:border-[#006e2d]/40 transition-all space-y-4"
          >
            <div className="space-y-3">
              {/* Plant Header */}
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold text-white shrink-0 ${
                      plant.status === 'Optimal'
                        ? 'bg-[#14532d]'
                        : plant.status.includes('Moderate')
                        ? 'bg-[#703a00]'
                        : 'bg-[#003b1b]'
                    }`}
                  >
                    <span className="material-symbols-outlined text-[24px]">{plant.icon}</span>
                  </div>
                  <div>
                    <h3 className="font-headline-sm text-[17px] text-[#131b2e] font-bold">
                      {plant.name}
                    </h3>
                    <p className="font-body-sm text-[11px] text-[#717970] italic">
                      {plant.scientificName}
                    </p>
                  </div>
                </div>

                <span
                  className={`font-label-sm text-[10px] px-2.5 py-1 rounded-full font-bold ${
                    plant.status === 'Optimal'
                      ? 'bg-[#7cf994] text-[#007230]'
                      : plant.status.includes('Moderate')
                      ? 'bg-[#ffdcc3] text-[#4f2700]'
                      : 'bg-[#dae2fd] text-[#404941]'
                  }`}
                >
                  {plant.status}
                </span>
              </div>

              {/* Progress & Health Bar */}
              <div className="space-y-1.5 bg-[#f2f3ff] p-3 rounded-lg border border-[#dae2fd]/40">
                <div className="flex justify-between items-center text-[12px] font-bold">
                  <span className="text-[#404941]">Health Index</span>
                  <span className="text-[#006e2d]">{plant.healthRatio}%</span>
                </div>
                <div className="w-full h-2 bg-[#eaedff] rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      plant.healthRatio > 85
                        ? 'bg-[#006e2d]'
                        : plant.healthRatio > 70
                        ? 'bg-[#62df7d]'
                        : 'bg-[#ffdcc3]'
                    }`}
                    style={{ width: `${plant.healthRatio}%` }}
                  ></div>
                </div>
                <div className="flex justify-between items-center text-[11px] text-[#717970] font-code-sm pt-0.5">
                  <span>Loc: {plant.quadrant}</span>
                  <span>{plant.metricLabel}: {plant.metricValue}</span>
                </div>
              </div>

              <div className="flex items-center justify-between text-[11px] text-[#717970] px-1 font-body-sm">
                <span>Last Scanned: {plant.lastScanned}</span>
                <span className="flex items-center gap-1 font-semibold text-[#006e2d]">
                  <span className="material-symbols-outlined text-[14px]">{plant.trendIcon}</span>
                  <span>{plant.statusTrend}</span>
                </span>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2 pt-2 border-t border-[#dae2fd]/40">
              <button
                onClick={() => onScanPlant(plant)}
                className="flex-1 py-2 rounded-lg bg-[#003b1b] hover:bg-[#14532d] text-white font-title-sm text-[12px] font-bold flex items-center justify-center gap-1.5 transition-all cursor-pointer shadow-xs"
              >
                <span className="material-symbols-outlined text-[16px]">photo_camera</span>
                <span>Scan Plant</span>
              </button>

              <button
                onClick={() => onNavigate('results-diagnostics')}
                className="px-3 py-2 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] text-[#131b2e] font-title-sm text-[12px] font-bold transition-colors cursor-pointer border border-[#dae2fd]"
                title="View Diagnostic History"
              >
                <span className="material-symbols-outlined text-[16px]">history</span>
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Add Plant Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <form
            onSubmit={handleAddPlant}
            className="bg-white rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl animate-in fade-in"
          >
            <div className="flex items-center justify-between">
              <h2 className="font-headline-sm text-headline-sm text-[#131b2e] font-bold">
                Add Monitored Crop
              </h2>
              <button
                type="button"
                onClick={() => setShowAddModal(false)}
                className="text-[#717970] hover:text-[#131b2e]"
              >
                <span className="material-symbols-outlined text-[24px]">close</span>
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="font-title-sm text-[12px] text-[#131b2e] font-semibold">
                  Crop / Cultivar Name
                </label>
                <input
                  type="text"
                  required
                  value={newPlantName}
                  onChange={(e) => setNewPlantName(e.target.value)}
                  placeholder="e.g. Greenhouse Cucumber Plot A"
                  className="w-full h-10 px-3 mt-1 rounded-lg bg-[#f2f3ff] text-[#131b2e] text-[13px] border border-[#dae2fd] focus:outline-none focus:bg-white"
                />
              </div>

              <div>
                <label className="font-title-sm text-[12px] text-[#131b2e] font-semibold">
                  Scientific Taxon
                </label>
                <input
                  type="text"
                  value={newPlantSpecies}
                  onChange={(e) => setNewPlantSpecies(e.target.value)}
                  placeholder="e.g. Cucumis sativus"
                  className="w-full h-10 px-3 mt-1 rounded-lg bg-[#f2f3ff] text-[#131b2e] text-[13px] border border-[#dae2fd] focus:outline-none focus:bg-white"
                />
              </div>

              <div>
                <label className="font-title-sm text-[12px] text-[#131b2e] font-semibold">
                  Location / Greenhouse Quadrant
                </label>
                <input
                  type="text"
                  value={newPlantLocation}
                  onChange={(e) => setNewPlantLocation(e.target.value)}
                  placeholder="e.g. Tunnel Bay 2 - Bed 4"
                  className="w-full h-10 px-3 mt-1 rounded-lg bg-[#f2f3ff] text-[#131b2e] text-[13px] border border-[#dae2fd] focus:outline-none focus:bg-white"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setShowAddModal(false)}
                className="px-4 py-2 rounded-lg bg-[#f2f3ff] text-[#131b2e] font-title-sm text-[12px] font-bold"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-5 py-2 rounded-lg bg-[#003b1b] text-white font-title-sm text-[12px] font-bold"
              >
                Save Crop
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};
