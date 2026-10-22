import { useEffect, useRef, useState } from "react";
import DATA from "../../samples/replay/episodes.json";
import SUMMARY from "../../samples/replay/summary.json";
import Scene3D from "./Scene3D.jsx";

const SUBS = {
  random: "Picks a direction at random every step.",
  straight: "Always heads straight for its star. Never looks at obstacles.",
  teacher: "Tries every move one second ahead and skips the ones that hit something.",
  trained: "A neural network that looks at the whole scene and picks a move.",
};
const LEGEND = [
  ["dot", "A drone. Its trail shows where it has been."],
  ["star", "The drone's target. Same colour as the drone that owns it."],
  ["obstacle", "An obstacle. Darker means lower to the ground."],
  ["over", "Dashed trail: the drone is flying over or under an obstacle, with no contact."],
  ["arrived", "Green ring: the drone has reached its star."],
  ["crash", "Red ring: the drone is touching an obstacle or another drone."],
  ["attn", "Orange line: what the trained model is weighing most (toggle above)."],
];
const DRONE_RADIUS = 0.5;
const total = Math.max(...DATA.episodes.map((e) => e.frames.length));
const [xmin, xmax, ymin, ymax, , zmax] = DATA.bounds;

function readTokens() {
  const s = getComputedStyle(document.documentElement);
  const g = (n) => s.getPropertyValue(n).trim();
  return {
    ground: g("--ground"), ink: g("--ink"), muted: g("--muted"), line: g("--line"), crash: g("--crash"), ok: g("--ok"),
    hi: g("--obstacle-hi"), lo: g("--obstacle-lo"), attn: g("--attn"), drones: [g("--d0"), g("--d1"), g("--d2")],
  };
}

function useTokens() {
  const [tokens, setTokens] = useState(readTokens);
  useEffect(() => {
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const update = () => setTokens(readTokens());
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return tokens;
}

function mix(a, b, t) {
  const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return "rgb(" + x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",") + ")";
}

function star(ctx, x, y, r) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
  }
  ctx.closePath();
}

const overCache = new WeakMap();
function overFlags(ep) {
  if (!overCache.has(ep)) {
    overCache.set(ep, ep.frames.map((f) => f.pos.map((q) => ep.obstacles.some((o) =>
      Math.hypot(q[0] - o.pos[0], q[1] - o.pos[1]) < o.radius + DRONE_RADIUS))));
  }
  return overCache.get(ep);
}

function crashesUpTo(ep, k) {
  let c = 0;
  for (let t = 1; t <= k; t++) ep.frames[t].hit.forEach((h, i) => { if (h && !ep.frames[t - 1].hit[i]) c++; });
  return c;
}

function useCanvasSize(ref) {
  const [size, setSize] = useState(0);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setSize(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

function drawMap(canvas, size, ep, k, showAttn, C) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const sx = (x) => ((x - xmin) / (xmax - xmin)) * size;
  const sy = (y) => size - ((y - ymin) / (ymax - ymin)) * size;
  const sr = (r) => (r / (xmax - xmin)) * size;
  ctx.fillStyle = C.ground;
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  for (let v = xmin; v <= xmax; v += 10) {
    ctx.beginPath(); ctx.moveTo(sx(v), 0); ctx.lineTo(sx(v), size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, sy(v)); ctx.lineTo(size, sy(v)); ctx.stroke();
  }
  ep.obstacles.forEach((o) => {
    ctx.fillStyle = mix(C.hi, C.lo, 1 - o.pos[2] / zmax);
    ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.arc(sx(o.pos[0]), sy(o.pos[1]), sr(o.radius), 0, 2 * Math.PI); ctx.fill();
    ctx.globalAlpha = 1;
  });
  const idx = Math.min(k, ep.frames.length - 1);
  const frame = ep.frames[idx];
  const n = ep.targets.length;
  if (showAttn && frame.attn) {
    const nObs = ep.obstacles.length;
    const nodePos = (j) => (j < n ? frame.pos[j] : j < n + nObs ? ep.obstacles[j - n].pos : ep.targets[j - n - nObs]);
    frame.attn.forEach((row, i) => row.forEach(([j, w]) => {
      const q = nodePos(j);
      ctx.strokeStyle = C.attn; ctx.globalAlpha = Math.min(1, 0.25 + w * 3); ctx.lineWidth = 1 + w * 8;
      ctx.beginPath(); ctx.moveTo(sx(frame.pos[i][0]), sy(frame.pos[i][1])); ctx.lineTo(sx(q[0]), sy(q[1])); ctx.stroke();
      ctx.globalAlpha = 1;
    }));
  }
  ep.targets.forEach((t, i) => {
    star(ctx, sx(t[0]), sy(t[1]), Math.max(8, size * 0.022));
    ctx.fillStyle = C.drones[i]; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.stroke();
  });
  const over = overFlags(ep);
  for (let i = 0; i < n; i++) {
    ctx.strokeStyle = C.drones[i]; ctx.lineWidth = 2;
    let t = 1;
    while (t <= idx) {
      const dashed = over[t][i] && !ep.frames[t].hit[i];
      const p = ep.frames[t - 1].pos[i];
      ctx.setLineDash(dashed ? [6, 5] : []);
      ctx.lineWidth = dashed ? 3 : 2;
      ctx.globalAlpha = dashed ? 1 : 0.8;
      ctx.beginPath(); ctx.moveTo(sx(p[0]), sy(p[1]));
      while (t <= idx && (over[t][i] && !ep.frames[t].hit[i]) === dashed) {
        const q = ep.frames[t].pos[i];
        ctx.lineTo(sx(q[0]), sy(q[1]));
        t++;
      }
      ctx.stroke();
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  frame.pos.forEach((q, i) => {
    const x = sx(q[0]), y = sy(q[1]), r = Math.max(5, size * 0.014);
    if (frame.hit[i]) { ctx.strokeStyle = C.crash; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(x, y, r * 2.4, 0, 2 * Math.PI); ctx.stroke(); }
    ctx.fillStyle = C.drones[i]; ctx.strokeStyle = frame.arrived[i] ? C.ok : C.ink; ctx.lineWidth = frame.arrived[i] ? 3 : 1.5;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  });
}

function Panel({ ep, k, showAttn, tokens }) {
  const ref = useRef(null);
  const size = useCanvasSize(ref);
  useEffect(() => { if (size) drawMap(ref.current, size, ep, k, showAttn, tokens); }, [ep, k, showAttn, tokens, size]);
  const idx = Math.min(k, ep.frames.length - 1);
  const n = ep.targets.length;
  const arrived = ep.frames[idx].arrived.filter(Boolean).length;
  const crashes = crashesUpTo(ep, idx);
  const ended = idx === ep.frames.length - 1 && idx < total - 1;
  return (
    <section className="panel">
      <h2>{DATA.titles[ep.policy]}</h2>
      <p className="sub">{SUBS[ep.policy]}</p>
      <canvas ref={ref} />
      <div className="status">
        <span className={"chip" + (arrived === n ? " good" : "")}>arrived {arrived}/{n}</span>
        <span className={"chip" + (crashes ? " bad" : "")}>crashes {crashes}</span>
        {ended && <span className="chip good">all arrived at {(idx * DATA.dt).toFixed(1)} s</span>}
      </div>
    </section>
  );
}

function LegendIcon({ kind, C }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    const dpr = window.devicePixelRatio || 1;
    cv.width = cv.height = 30 * dpr;
    const c = cv.getContext("2d");
    c.scale(dpr, dpr);
    if (kind === "dot" || kind === "arrived" || kind === "crash") {
      if (kind === "crash") { c.strokeStyle = C.crash; c.lineWidth = 3; c.beginPath(); c.arc(15, 15, 12, 0, 7); c.stroke(); }
      c.fillStyle = C.drones[0]; c.strokeStyle = kind === "arrived" ? C.ok : C.ink; c.lineWidth = kind === "arrived" ? 3 : 1.5;
      c.beginPath(); c.arc(15, 15, kind === "crash" ? 6 : 8, 0, 7); c.fill(); c.stroke();
    } else if (kind === "star") {
      star(c, 15, 15, 12); c.fillStyle = C.drones[1]; c.fill(); c.strokeStyle = C.ink; c.lineWidth = 1; c.stroke();
    } else if (kind === "over") {
      c.strokeStyle = C.drones[0]; c.lineWidth = 2; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(3, 25); c.lineTo(27, 5); c.stroke();
    } else if (kind === "obstacle") {
      c.fillStyle = C.lo; c.beginPath(); c.arc(15, 15, 12, 0, 7); c.fill();
    } else {
      c.strokeStyle = C.attn; c.lineWidth = 3; c.beginPath(); c.moveTo(3, 25); c.lineTo(27, 5); c.stroke();
    }
  }, [kind, C]);
  return <canvas ref={ref} />;
}

export default function App() {
  const tokens = useTokens();
  const [k, setK] = useState(Math.round((total - 1) * 0.45));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [showAttn, setShowAttn] = useState(false);
  const [pilot, setPilot] = useState("trained");
  const [resetSignal, setResetSignal] = useState(0);

  useEffect(() => {
    if (!playing) return;
    let raf, last = null, acc = 0;
    const tick = (ts) => {
      acc += ((ts - (last ?? ts)) / 1000) * (1 / DATA.dt) * speed;
      last = ts;
      const steps = Math.floor(acc);
      acc -= steps;
      if (steps) setK((prev) => {
        const next = Math.min(total - 1, Math.max(0, prev + steps));
        if (next >= total - 1) setPlaying(false);
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed]);

  const togglePlay = () => {
    if (!playing && k >= total - 1) setK(0);
    setPlaying(!playing);
  };

  return (
    <div className="wrap">
      <header>
        <h1>Four pilots, one map, three drones</h1>
      </header>

      <div className="controls" role="group" aria-label="Replay controls">
        <button id="play" type="button" onClick={togglePlay}>{playing ? "Pause" : k >= total - 1 ? "Replay" : "Play"}</button>
        <div className="scrub">
          <input id="scrub" type="range" min="0" max={total - 1} step="1" value={k} aria-label="Replay time"
            onChange={(e) => setK(Number(e.target.value))} />
          <span className="clock">{(k * DATA.dt).toFixed(1)} s</span>
        </div>
        <select id="speed" aria-label="Playback speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
          <option value="1">1× speed</option>
          <option value="2">2× speed</option>
          <option value="4">4× speed</option>
        </select>
        <label className="toggle">
          <input id="attn" type="checkbox" checked={showAttn} onChange={(e) => setShowAttn(e.target.checked)} /> Show what the model pays attention to
        </label>
      </div>

      <div className="grid">
        {DATA.episodes.map((ep) => <Panel key={ep.policy} ep={ep} k={k} showAttn={showAttn} tokens={tokens} />)}
      </div>

      <section className="card view3d">
        <div className="view3d-head">
          <h3>3D view</h3>
          <div className="pilots" role="group" aria-label="Pilot shown in 3D">
            {DATA.episodes.map((ep) => (
              <button key={ep.policy} type="button" aria-pressed={pilot === ep.policy} onClick={() => setPilot(ep.policy)}>
                {DATA.titles[ep.policy].split(" (")[0].split(":")[0]}
              </button>
            ))}
          </div>
          <button type="button" onClick={() => setResetSignal((r) => r + 1)}>Reset camera</button>
        </div>
        <Scene3D ep={DATA.episodes.find((e) => e.policy === pilot)} k={k} bounds={DATA.bounds} tokens={tokens} resetSignal={resetSignal} />
        <p className="note">Same moment as the maps above, now with height. Drag to rotate, scroll to zoom, right-drag to pan. Dashed lines drop from each drone and target to the ground so you can read their altitude.</p>
      </section>

      <div className="below">
        <div className="card">
          <h3>How to read the map</h3>
          <ul className="legend">
            {LEGEND.map(([kind, text]) => <li key={kind}><LegendIcon kind={kind} C={tokens} /><span>{text}</span></li>)}
          </ul>
          <p className="note">The map is seen from above. Obstacles float at different heights, so a drone can pass over or under one. Its trail turns dashed while it does; a real hit shows as a red ring and counts as a crash.</p>
        </div>
        <div className="card">
          <h3>Across {DATA.summary_episodes ?? 30} random maps</h3>
          <div className="tablewrap">
            <table>
              <thead><tr><th>Pilot</th><th className="num">At least one drone arrives</th><th className="num">All three arrive</th><th className="num">Maps with a crash</th></tr></thead>
              <tbody>
                {SUMMARY.map((r) => (
                  <tr key={r.policy}>
                    <td>{DATA.titles[r.policy]}</td>
                    <td className="num">{(r.any * 100).toFixed(0)}%</td>
                    <td className="num">{(r.all * 100).toFixed(0)}%</td>
                    <td className="num">{(r.crash * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">The trained model learned by copying the look-ahead planner and then practising on its own. The planner checks every move a second ahead using the exact simulator; the model sees one snapshot and answers in about a millisecond. The orange attention lines show what the model weighs most at that moment. They are a hint, not proof of why it moved.</p>
        </div>
      </div>
    </div>
  );
}
