"use client";
import { useEffect, useState } from "react";
import Link from "next/link";

type Profile = {
  id: number;
  timestamp: string;
  screenshot: string;
  score: number;
  action: string;
};

export default function GalleryPage() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const limit = 50;
  const [hasMore, setHasMore] = useState(true);
  
  const fetchProfiles = async (pageToFetch: number, append = false) => {
    try {
      const offset = (pageToFetch - 1) * limit;
      const res = await fetch(`/api/history?limit=${limit}&offset=${offset}`);
      if (!res.ok) return;
      const json = await res.json();
      if (json.data) {
        if (append) {
           setProfiles(prev => [...prev, ...json.data]);
        } else {
           setProfiles(json.data);
        }
        setHasMore(json.data.length === limit);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProfiles(1, false);
  }, []);

  const loadMore = () => {
    const nextPage = page + 1;
    setPage(nextPage);
    fetchProfiles(nextPage, true);
  };

  return (
    <main className="relative min-h-screen bg-[#050505] text-white overflow-x-hidden selection:bg-purple-500/30 font-sans p-6 lg:p-10 flex flex-col">
      <div className="fixed top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-indigo-900/20 blur-[120px] pointer-events-none animate-pulse" style={{ animationDuration: '10s' }} />
      <div className="fixed bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-emerald-900/10 blur-[120px] pointer-events-none animate-pulse delay-1000" style={{ animationDuration: '7s' }} />

      <div className="relative z-10 w-full max-w-[1600px] mx-auto">
        <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-10 pb-6 border-b border-white/10">
           <div>
             <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs font-semibold tracking-widest text-indigo-400 uppercase mb-4 shadow-[0_0_15px_rgba(99,102,241,0.1)]">
               Signal Intel Repository
             </div>
             <h1 className="text-4xl md:text-5xl font-black tracking-tighter bg-gradient-to-br from-white via-gray-200 to-gray-500 bg-clip-text text-transparent">
                Intel Gallery
             </h1>
           </div>
           <Link href="/" className="bg-white/5 hover:bg-white/10 border border-white/10 transition-colors px-6 py-3 rounded-xl text-xs font-bold tracking-widest uppercase flex items-center gap-3">
             <span className="text-lg leading-none">←</span> Back to Matrix
           </Link>
        </header>

        {loading && profiles.length === 0 ? (
          <div className="flex justify-center p-20">
             <div className="w-12 h-12 border-4 border-white/10 border-t-indigo-500 rounded-full animate-spin"></div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
               {profiles.map((p, idx) => (
                 <div key={`${p.id}-${idx}`} className="group relative bg-white/5 rounded-2xl overflow-hidden border border-white/10 shadow-xl aspect-[3/4] flex flex-col">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img 
                      src={`/api/image?file=${encodeURIComponent(p.screenshot)}`} 
                      alt={`Profile ${p.id}`}
                      loading="lazy"
                      className="absolute inset-0 w-full h-full object-cover object-center transition-transform duration-700 group-hover:scale-105"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                    <div className="absolute bottom-0 inset-x-0 p-5 flex justify-between items-end">
                       <span className="text-4xl font-black font-mono text-white tracking-tighter drop-shadow-xl">{p.score.toFixed(1)}</span>
                       <span className={`text-2xl font-black italic uppercase tracking-tighter transform -skew-x-6 drop-shadow-lg ${p.action === 'right' ? 'text-emerald-400' : 'text-rose-500'}`}>
                         {p.action === 'right' ? 'SMASH' : 'PASS'}
                       </span>
                    </div>
                 </div>
               ))}
            </div>
            {hasMore && (
              <div className="mt-12 flex justify-center">
                 <button onClick={loadMore} className="bg-white/5 hover:bg-white/10 border border-white/10 px-8 py-4 rounded-2xl text-xs uppercase tracking-widest font-bold transition-all shadow-xl">
                    Load Older Intel...
                 </button>
              </div>
            )}
            {!hasMore && profiles.length > 0 && (
              <div className="mt-12 text-center text-gray-500 text-xs tracking-widest uppercase font-bold">
                 End of Repository
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}
