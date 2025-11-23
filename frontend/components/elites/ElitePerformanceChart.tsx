'use client';

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Cell } from 'recharts';
import { EliteTagComparison } from '@/types/elite';
import { useState, useEffect } from 'react';

interface ChartProps {
  data: EliteTagComparison[];
  onSectorClick?: (sectorTag: string) => void;
}

export default function ElitePerformanceChart({ data, onSectorClick }: ChartProps) {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // For mobile: Show top 8 categories by performance edge
  const displayData = isMobile
    ? [...data].sort((a, b) => b.performance_edge - a.performance_edge).slice(0, 8)
    : data;

  // Colors for top performers
  const getBarColor = (value: number, index: number) => {
    if (isMobile && index < 3) {
      return ['#f59e0b', '#eab308', '#84cc16'][index]; // Gold, Yellow, Lime for top 3
    }
    return '#10b981'; // Default emerald
  };

  // Calculate stats for subtitle
  const totalElites = data.reduce((sum, item) => sum + item.elite_trader_count, 0);
  const avgEdge = data.reduce((sum, item) => sum + item.performance_edge, 0) / data.length;
  const topSector = displayData[0];

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload[0]) return null;
    const data = payload[0].payload;
    return (
      <div className="bg-zinc-950 border border-white/10 rounded-lg p-3 shadow-xl">
        <div className="text-white font-bold text-sm mb-2">{data.tag}</div>
        <div className="space-y-1.5 text-xs">
          <div className="flex justify-between gap-4">
            <span className="text-emerald-400 font-medium">Elite ROI:</span>
            <span className="text-white font-bold">{data.elite_avg_roi.toFixed(1)}%</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-zinc-400 font-medium">Avg ROI:</span>
            <span className="text-white font-mono">{data.non_elite_avg_roi.toFixed(1)}%</span>
          </div>
          <div className="flex justify-between gap-4 pt-1.5 border-t border-white/10">
            <span className="text-amber-400 font-medium">Edge:</span>
            <span className="text-amber-300 font-bold">+{data.performance_edge.toFixed(1)}%</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-zinc-400">Elite Traders:</span>
            <span className="text-white font-mono">{data.elite_trader_count}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-zinc-400">Events:</span>
            <span className="text-white font-mono">{data.number_of_events?.toLocaleString() || 0}</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-zinc-900/30 border border-white/5 rounded-xl p-4 md:p-6 flex flex-col">
      <div className="mb-4 md:mb-6">
        <h3 className="text-xs md:text-sm font-bold text-zinc-400 uppercase tracking-wider">
          Elite Edge by Category (ROI %)
        </h3>
        <div className="flex items-center gap-3 mt-2 flex-wrap">
          {isMobile ? (
            <p className="text-[10px] text-zinc-500">Top 8 performers • Avg edge: <span className="text-amber-400 font-bold">+{avgEdge.toFixed(1)}%</span></p>
          ) : (
            <>
              <span className="text-[10px] text-zinc-500">
                {totalElites} Elite Traders • Avg Edge: <span className="text-amber-400 font-bold">+{avgEdge.toFixed(1)}%</span>
              </span>
              {topSector && (
                <span className="text-[10px] px-2 py-0.5 bg-amber-500/10 text-amber-400 rounded border border-amber-500/20">
                  Top: {topSector.tag} (+{topSector.performance_edge.toFixed(1)}%)
                </span>
              )}
            </>
          )}
        </div>
      </div>

      {/* Mobile: Vertical layout with taller chart */}
      {isMobile ? (
        <div className="w-full" style={{ height: '400px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={displayData}
              layout="vertical"
              margin={{ top: 5, right: 10, left: 5, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={true} vertical={false} />
              <XAxis
                type="number"
                stroke="#71717a"
                fontSize={10}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `${value}%`}
              />
              <YAxis
                type="category"
                dataKey="tag"
                stroke="#71717a"
                fontSize={10}
                tickLine={false}
                axisLine={false}
                width={80}
              />
              <Tooltip content={<CustomTooltip />} cursor={false} />
              <Bar
                dataKey="elite_avg_roi"
                name="Elite ROI"
                radius={[0, 4, 4, 0]}
                cursor="pointer"
                onClick={(data: any) => data?.tag && onSectorClick?.(data.tag)}
              >
                {displayData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={getBarColor(entry.elite_avg_roi, index)}
                    className="hover:opacity-80 transition-opacity"
                  />
                ))}
              </Bar>
              <Bar
                dataKey="non_elite_avg_roi"
                name="Avg ROI"
                fill="#3f3f46"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        // Desktop: Horizontal layout
        <div className="w-full h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={displayData}
              margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
              <XAxis
                dataKey="tag"
                stroke="#71717a"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis
                stroke="#71717a"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip content={<CustomTooltip />} cursor={false} />
              <Legend
                wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                iconSize={10}
              />
              <Bar
                dataKey="elite_avg_roi"
                name="Elite ROI"
                fill="#10b981"
                radius={[4, 4, 0, 0]}
                barSize={18}
                cursor="pointer"
                onClick={(data: any) => data?.tag && onSectorClick?.(data.tag)}
              />
              <Bar
                dataKey="non_elite_avg_roi"
                name="Avg Market ROI"
                fill="#3f3f46"
                radius={[4, 4, 0, 0]}
                barSize={18}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}