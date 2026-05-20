"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type ScoreData = {
  averages: { score: number; face_biased: number; multimodal: number; ridge: number; knn: number; };
  actionAverages: { left: number; right: number; };
  swipes: { left: number; right: number; leftPercent: number; rightPercent: number; };
  methodDistribution: Record<string, number>;
  scoreDistribution: Record<string, number>;
  velocity: number;
  latest: { score: number | null; face_biased: number | null; multimodal: number | null; ridge: number | null; knn: number | null; action: string | null; method: string | null; divergence: number | null; screenshot: string | null; } | null;
  dynamicThreshold: {
    enabled: boolean;
    active: boolean;
    threshold: number;
    rawThreshold: number | null;
    fallbackThreshold: number;
    targetRightRate: number;
    window: number;
    minHistory: number;
    minThreshold: number;
    maxThreshold: number;
    historyCount: number;
    projectedRightCount: number;
    projectedRightPercent: number;
    actualLeftCount: number;
    actualRightCount: number;
    actualRightPercent: number;
  };
  trend?: { time: number; threshold: number; score: number | null; action: string | null }[];
  records: number;
};

export default function Dashboard() {
  const [data, setData] = useState<ScoreData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionOffset, setSessionOffset] = useState<number>(() => {
    if (typeof window === 'undefined') return 0;
    const savedOffset = localStorage.getItem('bumble_session_offset');
    return savedOffset ? parseInt(savedOffset, 10) : 0;
  });
  const [viewMode, setViewMode] = useState<'all' | 'session'>(() => {
    if (typeof window === 'undefined') return 'all';
    const savedMode = localStorage.getItem('bumble_view_mode');
    return savedMode === 'session' || savedMode === 'all' ? savedMode : 'all';
  });
  const [totalRecords, setTotalRecords] = useState<number>(0);
  const [trendWindow, setTrendWindow] = useState<number>(30 * 60 * 1000);
  const [currentTime, setCurrentTime] = useState<number>(() => typeof window === 'undefined' ? 1779344400000 : Date.now());

  const handleSetViewMode = (mode: 'all' | 'session') => {
    setViewMode(mode);
    localStorage.setItem('bumble_view_mode', mode);
  };

  const handleReset = () => {
    setSessionOffset(totalRecords);
    setViewMode('session');
    localStorage.setItem('bumble_session_offset', totalRecords.toString());
    localStorage.setItem('bumble_view_mode', 'session');
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const currentOffset = viewMode === 'session' ? sessionOffset : 0;
        const res = await fetch(`/api/scores?offset=${currentOffset}`);
        if (!res.ok) throw new Error("Failed to fetch data");
        const json = await res.json();
        if (json.error) setError(json.error);
        else { 
          setData(json); 
          setTotalRecords(json.totalRecords);
          setError(null); 
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [viewMode, sessionOffset]);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(Date.now());
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number | null | undefined) => num === null || num === undefined ? "--" : num.toFixed(2);

  if (error) return <div className="min-h-screen bg-[#050505] flex items-center justify-center text-rose-500 p-8">{error}</div>;
  if (!data) return <div className="min-h-screen bg-[#050505] flex items-center justify-center p-8"><div className="w-12 h-12 border-4 border-white/10 border-t-purple-500 rounded-full animate-spin"></div></div>;

  const topMethods = Object.entries(data.methodDistribution).sort((a, b) => b[1] - a[1]);


  return (
    <main className="relative min-h-screen bg-[#050505] text-white overflow-x-hidden selection:bg-purple-500/30 font-sans p-6 lg:p-10 flex flex-col">
      <div className="fixed top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-purple-900/20 blur-[120px] pointer-events-none animate-pulse" style={{ animationDuration: '10s' }} />
      <div className="fixed bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-indigo-900/20 blur-[120px] pointer-events-none animate-pulse delay-1000" style={{ animationDuration: '7s' }} />
      
      <div className="relative z-10 w-full max-w-[1600px] mx-auto flex flex-col gap-8">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs font-semibold tracking-widest text-emerald-400 uppercase mb-4 shadow-[0_0_15px_rgba(52,211,153,0.1)]">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              Operational Telemetry
            </div>
            <h1 className="text-4xl md:text-5xl font-black tracking-tighter bg-gradient-to-br from-white via-gray-200 to-gray-500 bg-clip-text text-transparent">
              BumbleLog Matrix
            </h1>
          </div>
          <div className="flex flex-wrap gap-4 justify-end">
            <div className="bg-white/5 px-4 py-2 rounded-2xl border border-white/5 backdrop-blur-md flex flex-col items-end justify-center hidden sm:flex">
              <span className="text-gray-500 uppercase text-[9px] tracking-widest font-bold">Threshold</span>
              <span className="text-amber-400 text-sm font-mono font-bold">{formatNumber(data.dynamicThreshold.threshold)}</span>
            </div>
            <div className="bg-white/5 px-4 py-2 rounded-2xl border border-white/5 backdrop-blur-md flex flex-col items-end justify-center hidden sm:flex">
              <span className="text-gray-500 uppercase text-[9px] tracking-widest font-bold">Driver</span>
              <span className="text-indigo-400 text-sm font-bold uppercase">{topMethods[0]?.[0] || 'N/A'}</span>
            </div>
            <div className="bg-white/5 px-4 py-2 rounded-2xl border border-white/5 backdrop-blur-md flex flex-col items-end justify-center hidden sm:flex">
              <span className="text-gray-500 uppercase text-[9px] tracking-widest font-bold">Divergence</span>
              <span className="text-purple-400 text-sm font-mono font-bold uppercase">{formatNumber(data.latest?.divergence)}</span>
            </div>
            <div className="bg-white/5 px-5 py-3 rounded-2xl border border-white/5 backdrop-blur-md flex flex-col items-end">
              <span className="text-gray-500 uppercase text-[10px] tracking-widest font-bold">Processing Speed</span>
              <div className="flex items-baseline gap-1">
                <span className="text-white text-2xl font-mono font-bold">{data.velocity.toFixed(1)}</span>
                <span className="text-gray-400 text-xs font-mono">swipes/min</span>
              </div>
            </div>
            <div className="bg-white/5 px-5 py-3 rounded-2xl border border-white/5 backdrop-blur-md flex flex-col items-end">
              <span className="text-gray-500 uppercase text-[10px] tracking-widest font-bold">Total Processed</span>
              <span className="text-white text-2xl font-mono font-bold">{totalRecords.toLocaleString()}</span>
            </div>
            <Link href="/gallery" className="bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-300 border border-indigo-500/30 px-6 py-3 rounded-2xl flex flex-col items-center justify-center transition-colors shadow-[0_0_15px_rgba(79,70,229,0.15)] group">
              <span className="text-[10px] tracking-widest uppercase font-bold text-indigo-400/80 mb-0.5">Access Repository</span>
              <span className="text-sm font-black uppercase tracking-widest group-hover:scale-105 transition-transform">View Gallery</span>
            </Link>
          </div>
        </header>

        {/* Main 2-Column Layout */}
        <div className="flex flex-col xl:flex-row gap-8">
          
          {/* LEFT COLUMN - MASSIVE LIVE PREVIEW */}
          <div className="w-full xl:w-[480px] shrink-0 flex flex-col">
            <div className="flex-1 bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl p-6 shadow-2xl relative overflow-hidden flex flex-col group">
              <h2 className="text-gray-400 font-medium text-xs tracking-widest uppercase mb-4">Live Signal Intercept</h2>
              {data.latest?.screenshot ? (
                <div className="relative w-full flex-1 min-h-[600px] rounded-2xl overflow-hidden bg-black/50 border border-white/10 shadow-2xl">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={`/api/image?file=${encodeURIComponent(data.latest.screenshot)}`} 
                    alt="Latest Profile"
                    className="absolute inset-0 w-full h-full object-cover object-center transition-transform duration-1000 group-hover:scale-[1.03]"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent" />
                  
                  {/* Floating Small Metrics (Label on top, Number on bottom) */}
                  <div className="absolute top-4 right-4 flex flex-col gap-2 items-end">
                    {[{l:'Multi', v:data.latest.multimodal}, {l:'Ridge', v:data.latest.ridge}, {l:'KNN', v:data.latest.knn}].map(m => (
                       <div key={m.l} className="flex flex-col items-center bg-black/60 px-3 py-1.5 rounded-lg backdrop-blur-xl border border-white/10 shadow-lg min-w-[50px]">
                          <span className="text-[9px] uppercase tracking-widest text-gray-400 mb-0.5">{m.l}</span>
                          <span className="text-sm font-mono font-bold text-gray-100 leading-none">{formatNumber(m.v)}</span>
                       </div>
                    ))}
                  </div>

                  <div className="absolute inset-x-0 bottom-0 p-8 flex flex-col gap-4">
                    <div className="flex justify-between items-end">
                      <div className="flex flex-col">
                        <span className="text-white/60 text-xs uppercase tracking-widest font-bold mb-1">Final Score</span>
                        <span className="text-6xl font-black text-white drop-shadow-2xl font-mono tracking-tighter">
                          {formatNumber(data.latest.score)}
                        </span>
                      </div>
                      {data.latest.action && (
                        <div key={data.latest.screenshot || 'empty'} className={`animate-pop-stamp text-6xl font-black italic uppercase tracking-tighter drop-shadow-[0_0_25px_currentColor] ${data.latest.action === 'right' ? 'text-emerald-400' : 'text-rose-500'}`}>
                          {data.latest.action === 'right' ? 'SMASH' : 'PASS'}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="w-full flex-1 min-h-[600px] rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
                  <span className="text-gray-600 uppercase text-xs tracking-widest animate-pulse">Awaiting Signal...</span>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT COLUMN - TELEMETRY */}
          <div className="flex-1 flex flex-col gap-8">
            
            {/* Top Stats Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              
              <div className="bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl p-8 shadow-2xl relative overflow-hidden flex flex-col justify-center">
                <div className="flex justify-between items-start mb-8 gap-4 flex-wrap">
                  <h2 className="text-gray-400 font-medium text-xs tracking-widest uppercase">Decision Engine & Thresholds</h2>
                  <div className="flex items-center gap-1.5 bg-black/40 p-1 rounded-lg border border-white/5 z-20">
                    <button 
                      onClick={() => handleSetViewMode('all')}
                      className={`px-3 py-1.5 rounded text-[9px] font-bold uppercase tracking-widest transition-colors ${viewMode === 'all' ? 'bg-white/20 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}
                    >
                      All-Time
                    </button>
                    <button 
                      onClick={() => handleSetViewMode('session')}
                      className={`px-3 py-1.5 rounded text-[9px] font-bold uppercase tracking-widest transition-colors ${viewMode === 'session' ? 'bg-white/20 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}
                    >
                      Latest Setup
                    </button>
                    <button 
                      onClick={handleReset}
                      className="ml-1 px-3 py-1.5 rounded text-[9px] font-bold uppercase tracking-widest bg-rose-500/20 text-rose-400 hover:bg-rose-500/40 transition-colors border border-rose-500/20 shadow-sm"
                      title="Reset Stats"
                    >
                      Reset
                    </button>
                  </div>
                </div>

                <div className="flex justify-between items-end mb-4 relative z-10 gap-4">
                  <div className="flex flex-col flex-1">
                    <span className="text-rose-400 font-mono text-5xl font-bold">{data.swipes.leftPercent.toFixed(1)}%</span>
                    <span className="text-gray-500 text-[10px] uppercase tracking-widest mt-2">Left Swipes ({data.swipes.left})</span>
                    <span className="text-rose-300/80 text-sm mt-3 font-mono bg-rose-500/10 px-3 py-1.5 rounded-lg w-fit border border-rose-500/20">Avg Score: {formatNumber(data.actionAverages.left)}</span>
                  </div>
                  <div className="flex flex-col items-end flex-1 text-right">
                    <span className="text-emerald-400 font-mono text-5xl font-bold">{data.swipes.rightPercent.toFixed(1)}%</span>
                    <span className="text-gray-500 text-[10px] uppercase tracking-widest mt-2">Right Swipes ({data.swipes.right})</span>
                    <span className="text-emerald-300/80 text-sm mt-3 font-mono bg-emerald-500/10 px-3 py-1.5 rounded-lg w-fit border border-emerald-500/20">Avg Score: {formatNumber(data.actionAverages.right)}</span>
                  </div>
                </div>
                <div className="h-5 w-full bg-white/5 rounded-full overflow-hidden flex shadow-inner relative z-10 mt-4">
                  <div className="h-full bg-gradient-to-r from-rose-600 to-rose-400 relative transition-all duration-1000" style={{ width: `${data.swipes.leftPercent}%` }} />
                  <div className="h-full bg-gradient-to-r from-emerald-400 to-emerald-600 relative transition-all duration-1000" style={{ width: `${data.swipes.rightPercent}%` }} />
                </div>
              </div>

              {/* Score Distribution Analytics */}
              <div className="bg-gradient-to-br from-blue-500/10 to-indigo-500/10 backdrop-blur-2xl border border-blue-500/20 rounded-3xl p-8 shadow-2xl relative flex flex-col justify-between">
                <div>
                  <h2 className="text-blue-400/80 font-medium text-xs tracking-widest uppercase mb-4">Final Score Distribution</h2>
                  <div className="flex items-end gap-3 h-48 mt-4 pb-2">
                     {Object.entries(data.scoreDistribution || {}).map(([label, count]) => {
                        const maxCount = Math.max(...Object.values(data.scoreDistribution || {}), 1);
                        const heightPct = (count / maxCount) * 100;
                        return (
                          <div key={label} className="flex flex-col items-center flex-1 h-full justify-end group">
                             <span className="text-lg text-gray-200 font-mono font-bold mb-1 transition-all duration-300">{count}</span>
                             <div className="w-full bg-white/5 rounded-t-md relative flex items-end justify-center group-hover:bg-white/10 transition-colors" style={{ height: '100%' }}>
                                <div className="w-full bg-gradient-to-t from-blue-600 to-indigo-400 rounded-t-md transition-all duration-1000" style={{ height: `${heightPct}%` }} />
                             </div>
                             <span className="text-[8px] uppercase tracking-widest text-gray-400 mt-2">{label}</span>
                          </div>
                        )
                     })}
                  </div>
                </div>
              </div>
            </div>

            {/* Dynamic Threshold Trend - Spanning full width */}
            {data.trend && data.trend.length > 0 && (
              <div className="bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl p-8 shadow-2xl flex flex-col gap-6">
                <div className="flex justify-between items-center z-10 flex-wrap gap-4">
                  <div>
                    <h2 className="text-amber-400/85 font-medium text-xs tracking-widest uppercase mb-1">Dynamic Threshold Trend</h2>
                    <p className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">Real-time sliding window calculations over the active timeframe</p>
                  </div>
                  <div className="flex bg-black/40 p-1 rounded-lg border border-white/5">
                    {[
                      { l: '1m', v: 60 * 1000 },
                      { l: '5m', v: 5 * 60 * 1000 },
                      { l: '30m', v: 30 * 60 * 1000 },
                      { l: '1h', v: 60 * 60 * 1000 },
                      { l: '6h', v: 6 * 60 * 60 * 1000 },
                      { l: '12h', v: 12 * 60 * 60 * 1000 },
                      { l: '1d', v: 24 * 60 * 60 * 1000 },
                    ].map(opt => (
                      <button 
                        key={opt.l} 
                        onClick={() => setTrendWindow(opt.v)}
                        className={`px-3 py-1.5 rounded text-[10px] font-bold uppercase tracking-widest transition-colors ${trendWindow === opt.v ? 'bg-amber-500/20 text-amber-300 shadow-sm border border-amber-500/30' : 'text-gray-500 hover:text-gray-300 border border-transparent'}`}
                      >
                        {opt.l}
                      </button>
                    ))}
                  </div>
                </div>
                
                <div className="relative h-48 w-full z-10 bg-black/40 rounded-xl border border-white/5 p-4 overflow-hidden flex gap-6">
                  {(() => {
                    const now = currentTime || (data.trend && data.trend.length > 0 ? data.trend[data.trend.length - 1].time : 1779344400000);
                    const minTime = now - trendWindow;
                    const filteredTrend = data.trend ? data.trend.filter(t => t.time >= minTime) : [];
                    
                    const currentVal = data.trend.length > 0 ? data.trend[data.trend.length - 1].threshold : 54;

                    // Downsample the data into 80 time-based buckets (approx every 10px on a standard screen)
                    const bucketCount = 80;
                    const bucketWidth = trendWindow / bucketCount;
                    const sampledTrend: typeof filteredTrend = [];
                    
                    for (let i = 0; i < bucketCount; i++) {
                      const bucketStart = minTime + i * bucketWidth;
                      const bucketEnd = bucketStart + bucketWidth;
                      
                      const pointsInBucket = filteredTrend.filter(t => t.time >= bucketStart && t.time < bucketEnd);
                      if (pointsInBucket.length > 0) {
                        sampledTrend.push(pointsInBucket[pointsInBucket.length - 1]);
                      } else {
                        const lastKnownPoint = filteredTrend.filter(t => t.time < bucketEnd).pop();
                        if (lastKnownPoint) {
                          sampledTrend.push({
                            time: bucketEnd,
                            threshold: lastKnownPoint.threshold,
                            score: null,
                            action: null
                          });
                        }
                      }
                    }

                    // Fallback to a flat line at the current threshold if there's no data in the active timescale window
                    const displayTrend = sampledTrend.length >= 2
                      ? sampledTrend
                      : [
                          { time: minTime, threshold: currentVal, score: null, action: null },
                          { time: now, threshold: currentVal, score: null, action: null }
                        ];

                    const thresholdValues = displayTrend.map(t => t.threshold);
                    const maxThresholdVal = Math.max(...thresholdValues);
                    const minThresholdVal = Math.min(...thresholdValues);

                    // Zoom in tightly around the min and max threshold values of the timescale
                    let minThreshold = Math.max(0, minThresholdVal - 2);
                    let maxThreshold = Math.min(100, maxThresholdVal + 2);
                    if (maxThreshold === minThreshold) {
                      minThreshold = Math.max(0, minThreshold - 5);
                      maxThreshold = Math.min(100, maxThreshold + 5);
                    }

                    const points = displayTrend.map((t) => {
                      const x = Math.min(100, ((t.time - minTime) / trendWindow) * 100);
                      const y = 100 - ((t.threshold - minThreshold) / (maxThreshold - minThreshold)) * 100;
                      return `${x},${y}`;
                    }).join(' ');

                    const scorePoints = displayTrend.filter(t => t.score !== null).map((t) => {
                      const x = Math.min(100, ((t.time - minTime) / trendWindow) * 100);
                      const y = 100 - ((t.score! - minThreshold) / (maxThreshold - minThreshold)) * 100;
                      return { x, y, action: t.action };
                    });

                    const firstX = Math.min(100, ((displayTrend[0].time - minTime) / trendWindow) * 100);
                    const lastX = Math.min(100, ((displayTrend[displayTrend.length - 1].time - minTime) / trendWindow) * 100);
                    const lastY = 100 - ((displayTrend[displayTrend.length - 1].threshold - minThreshold) / (maxThreshold - minThreshold)) * 100;
                    const areaPath = `M ${firstX},100 L ${points} L ${lastX},100 Z`;

                    // Stock market trend math
                    const startVal = displayTrend[0].threshold;
                    const isUpTrend = currentVal >= startVal;
                    const strokeColor = isUpTrend ? '#10b981' : '#f43f5e';
                    const gradientColor = isUpTrend ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)';
                    const glowClass = isUpTrend ? 'drop-shadow-[0_0_6px_rgba(16,185,129,0.5)]' : 'drop-shadow-[0_0_6px_rgba(244,63,94,0.5)]';
                    const nowColorClass = isUpTrend ? 'text-emerald-400' : 'text-rose-400';
                    const nowLabelColorClass = isUpTrend ? 'text-emerald-500' : 'text-rose-500';

                    return (
                      <>
                        {/* Left Indicators stack */}
                        <div className="flex flex-col justify-between text-[11px] font-mono font-bold w-16 shrink-0 border-r border-white/5 pr-3 select-none">
                          <div className="flex flex-col">
                            <span className="text-[9px] uppercase tracking-wider text-white/40">Max</span>
                            <span className="text-gray-300 font-mono text-xs">{maxThresholdVal.toFixed(1)}</span>
                          </div>
                          <div className="flex flex-col py-1">
                            <span className={`text-[9px] uppercase tracking-wider ${nowLabelColorClass}`}>Now</span>
                            <span className={`text-sm font-black font-mono ${nowColorClass}`}>{currentVal.toFixed(1)}</span>
                          </div>
                          <div className="flex flex-col">
                            <span className="text-[9px] uppercase tracking-wider text-white/40">Min</span>
                            <span className="text-gray-300 font-mono text-xs">{minThresholdVal.toFixed(1)}</span>
                          </div>
                        </div>

                        {/* Right SVG Chart */}
                        <div className="relative flex-1 h-full overflow-hidden flex items-end">
                          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full overflow-visible absolute inset-0 py-2">
                            <defs>
                              <linearGradient id="threshold-area-gradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor={gradientColor} />
                                <stop offset="100%" stopColor="rgba(255, 255, 255, 0.0)" />
                              </linearGradient>
                            </defs>

                            {/* Grid lines */}
                            <line x1="0" y1="25" x2="100" y2="25" stroke="rgba(255,255,255,0.02)" strokeWidth="0.2" strokeDasharray="1,1" />
                            <line x1="0" y1="50" x2="100" y2="50" stroke="rgba(255,255,255,0.05)" strokeWidth="0.2" />
                            <line x1="0" y1="75" x2="100" y2="75" stroke="rgba(255,255,255,0.02)" strokeWidth="0.2" strokeDasharray="1,1" />
                            
                            {/* Area under the line */}
                            <path d={areaPath} fill="url(#threshold-area-gradient)" />

                            {/* Dynamic Threshold Line */}
                            <polyline points={points} fill="none" stroke={strokeColor} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" className={glowClass} />

                            {/* Event Rug Plot at the very bottom margin */}
                            {scorePoints.map((pt, i) => (
                               <line 
                                 key={i} 
                                 x1={pt.x} 
                                 y1="97" 
                                 x2={pt.x} 
                                 y2="100" 
                                 stroke={pt.action === 'right' ? '#10b981' : pt.action === 'left' ? '#f43f5e' : '#4b5563'} 
                                 strokeWidth="0.5"
                                 opacity="0.25"
                                 vectorEffect="non-scaling-stroke"
                               />
                            ))}
                          </svg>

                          {/* HTML Overlay to prevent vector stretching of text and dots */}
                          <div className="absolute inset-0 py-2 pointer-events-none z-20">
                            {/* Bloomberg-style horizontal grid labels */}
                            <div className="absolute right-2 top-[25%] -translate-y-1/2 text-[9px] font-mono font-bold text-white/20 select-none">
                              {(maxThreshold - (maxThreshold - minThreshold) * 0.25).toFixed(1)}
                            </div>
                            <div className="absolute right-2 top-[50%] -translate-y-1/2 text-[9px] font-mono font-bold text-white/20 select-none">
                              {(maxThreshold - (maxThreshold - minThreshold) * 0.5).toFixed(1)}
                            </div>
                            <div className="absolute right-2 top-[75%] -translate-y-1/2 text-[9px] font-mono font-bold text-white/20 select-none">
                              {(maxThreshold - (maxThreshold - minThreshold) * 0.75).toFixed(1)}
                            </div>

                            {/* Pulsing Live dot at the right edge */}
                            <div 
                              className="absolute -translate-x-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full z-30 flex items-center justify-center"
                              style={{ 
                                left: `${lastX}%`, 
                                top: `${lastY}%`
                              }}
                            >
                              <span 
                                className="absolute w-full h-full rounded-full opacity-100"
                                style={{ backgroundColor: strokeColor, boxShadow: `0 0 10px ${strokeColor}` }}
                              />
                              <span 
                                className="absolute w-full h-full rounded-full animate-ping opacity-75"
                                style={{ border: `2px solid ${strokeColor}` }}
                              />
                            </div>
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </div>
              </div>
            )}

            {/* Model Breakdown Grid */}
            <div className="flex flex-col gap-6">
              
              <div className="bg-white/10 backdrop-blur-2xl border border-white/20 rounded-3xl p-8 shadow-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
                <div>
                  <h3 className="text-gray-400 font-medium text-xs tracking-widest uppercase mb-2">Blended Final Score</h3>
                  <h2 className="text-2xl font-semibold text-white">Overall Output (Global Avg)</h2>
                </div>
                <span className="text-6xl font-black bg-gradient-to-br from-white to-gray-400 bg-clip-text text-transparent font-mono tracking-tighter">
                  {formatNumber(data.averages.score)}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-6">
                {[
                  { key: 'face_biased', title: 'Face Biased', subtitle: 'Aesthetic Focus', gradient: 'from-purple-500 to-indigo-500', glow: 'group-hover:opacity-20' },
                  { key: 'multimodal', title: 'Multimodal', subtitle: 'Vision + Text', gradient: 'from-blue-500 to-cyan-500', glow: 'group-hover:opacity-20' },
                  { key: 'ridge', title: 'Ridge', subtitle: 'Linear Model', gradient: 'from-amber-500 to-orange-500', glow: 'group-hover:opacity-10' },
                  { key: 'knn', title: 'k-NN', subtitle: 'Similarity', gradient: 'from-pink-500 to-rose-500', glow: 'group-hover:opacity-20' }
                ].map((metric) => {
                  const avg = data.averages[metric.key as keyof typeof data.averages];
                  return (
                    <div key={metric.key} className="group relative bg-white/5 backdrop-blur-xl border border-white/5 rounded-3xl p-6 transition-all duration-300 hover:bg-white/10 hover:-translate-y-1 overflow-hidden flex flex-col justify-between min-h-[160px]">
                      <div className={`absolute -inset-1 bg-gradient-to-r ${metric.gradient} opacity-0 ${metric.glow} blur-xl transition-opacity duration-500`} />
                      <div className="relative z-10">
                        <h3 className="text-gray-500 font-medium text-[10px] tracking-widest uppercase mb-1">{metric.subtitle}</h3>
                        <h2 className="text-lg font-semibold text-gray-200 mb-6">{metric.title}</h2>
                        <span className={`text-6xl font-black tracking-tighter bg-gradient-to-br ${metric.gradient} bg-clip-text text-transparent font-mono`}>
                          {formatNumber(avg)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>
        </div>

      </div>
    </main>
  );
}
