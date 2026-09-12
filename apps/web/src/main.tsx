import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { z } from "zod";
import "./style.css";

const Stroke = z.tuple([
  z.number(),
  z.number(),
  z.number(),
  z.number(),
  z.number(),
  z.number(),
]);
const Event = z.object({
  run_id: z.string(),
  target_id: z.string(),
  revision: z.number(),
  status: z.enum(["running", "paused", "done", "cancelled", "error"]),
  generation: z.number(),
  generations: z.number(),
  score: z.number(),
  metrics: z.record(z.string(), z.number()),
  strokes: z.array(Stroke),
  activity: z.number(),
  elapsed_seconds: z.number(),
  held_out: z.unknown().nullable(),
  error: z.string().nullable(),
});
type Progress = z.infer<typeof Event>;
type Target = {
  target_id: string;
  original: string;
  target: string;
  size: number;
  foreground: number;
  blank: boolean;
  free: boolean;
  message: string | null;
};
type Graph = {
  label: string;
  nodes: number;
  edges: number;
  readout_parameters: number;
};
const paper = "#f6f1e1";
async function api(path: string, body?: BodyInit) {
  const response = await fetch(path, {
    method: "POST",
    body,
    headers:
      typeof body === "string" ? { "Content-Type": "application/json" } : {},
  });
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : "Invalid request. Check image and training settings.",
    );
  return result;
}
function download(canvas: HTMLCanvasElement, filename: string) {
  const link = document.createElement("a");
  link.download = filename;
  link.href = canvas.toDataURL("image/png");
  link.click();
}
function App() {
  const [graph, setGraph] = useState<Graph>();
  const [target, setTarget] = useState<Target>();
  const [event, setEvent] = useState<Progress>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [threshold, setThreshold] = useState(35);
  const [size, setSize] = useState(128);
  const [seed, setSeed] = useState(42);
  const [generations, setGenerations] = useState(10);
  const [speed, setSpeed] = useState(2);
  const [replayPaused, setReplayPaused] = useState(false);
  const [drawn, setDrawn] = useState(0);
  const ink = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const best = useRef<HTMLCanvasElement>(null);
  const socket = useRef<WebSocket | null>(null);
  const current = useRef<{ target?: string; run?: string }>({});
  const tick = useRef(0);
  const strokes = useRef<z.infer<typeof Stroke>[]>([]);
  const requestVersion = useRef(0);
  const playing = useRef({ speed, paused: replayPaused });
  playing.current = { speed, paused: replayPaused };
  const clear = (canvas: HTMLCanvasElement | null) => {
    if (canvas) {
      const c = canvas.getContext("2d")!;
      c.fillStyle = paper;
      c.fillRect(0, 0, canvas.width, canvas.height);
    }
  };
  const line = (canvas: HTMLCanvasElement, s: z.infer<typeof Stroke>) => {
    if (s[4] > 0.45) {
      const c = canvas.getContext("2d")!;
      c.strokeStyle = "#191c17";
      c.lineWidth = canvas.width / (target?.size ?? 128);
      c.lineCap = "round";
      c.beginPath();
      c.moveTo(s[0] * canvas.width, s[1] * canvas.height);
      c.lineTo(s[2] * canvas.width, s[3] * canvas.height);
      c.stroke();
    }
  };
  const lineRef = useRef(line);
  lineRef.current = line;
  const stop = () => {
    socket.current?.close();
    socket.current = null;
    current.current.run = undefined;
    setEvent(undefined);
    strokes.current = [];
    tick.current = 0;
    setDrawn(0);
    setReplayPaused(false);
    clear(ink.current);
    clear(best.current);
    overlay.current?.getContext("2d")?.clearRect(0, 0, 768, 768);
  };
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((r) => setGraph(r.graph))
      .catch(() =>
        setError("Local trainer is unavailable. Start scripts/run.sh."),
      );
    clear(ink.current);
    clear(best.current);
    return () => socket.current?.close();
  }, []);
  useEffect(() => {
    let frame = 0,
      last = 0,
      credit = 0;
    function animate(now: number) {
      const dt = Math.min(100, now - last);
      last = now;
      if (
        !playing.current.paused &&
        tick.current < strokes.current.length &&
        ink.current &&
        overlay.current
      ) {
        credit += dt * 0.045 * playing.current.speed;
        let count = Math.floor(credit);
        credit -= count;
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches)
          count = strokes.current.length;
        while (count-- > 0 && tick.current < strokes.current.length) {
          lineRef.current(ink.current, strokes.current[tick.current++]);
        }
        const s = strokes.current[Math.max(0, tick.current - 1)];
        if (s) {
          const c = overlay.current.getContext("2d")!;
          c.clearRect(0, 0, 768, 768);
          c.save();
          c.translate(s[2] * 768, s[3] * 768);
          c.rotate(s[5]);
          c.strokeStyle = "#20261f";
          c.lineWidth = 2;
          for (const y of [-1, 1]) {
            for (let i = -1; i <= 1; i++) {
              c.beginPath();
              c.moveTo(i * 5, y * 3);
              c.lineTo(i * 10 - 5, y * 16);
              c.stroke();
            }
            c.fillStyle = "#ffffffb0";
            c.beginPath();
            c.ellipse(-7, y * 9, 13, 6, y * 0.5, 0, Math.PI * 2);
            c.fill();
            c.stroke();
          }
          c.fillStyle = "#20261f";
          c.beginPath();
          c.ellipse(0, 0, 12, 6, 0, 0, Math.PI * 2);
          c.fill();
          c.fillStyle = "#dc5036";
          c.beginPath();
          c.arc(10, -4, 4, 0, 7);
          c.arc(10, 4, 4, 0, 7);
          c.fill();
          c.strokeStyle = "#20261f";
          c.beginPath();
          c.moveTo(7, 7);
          c.lineTo(0, 0);
          c.stroke();
          c.restore();
        }
        setDrawn(tick.current);
      }
      frame = requestAnimationFrame(animate);
    }
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, []);
  async function replace(path: string, body?: BodyInit) {
    const version = ++requestVersion.current;
    const old = current.current.run;
    stop();
    current.current.target = undefined;
    setTarget(undefined);
    setBusy(true);
    setError("");
    try {
      if (old) await api(`/api/runs/${old}/cancel`);
      const result: Target = await api(path, body);
      if (version !== requestVersion.current) return;
      current.current.target = result.target_id;
      setTarget(result);
    } catch (e) {
      if (version === requestVersion.current) setError(String(e));
    } finally {
      if (version === requestVersion.current) setBusy(false);
    }
  }
  const upload = (file?: File) => {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      setError("Choose an image smaller than 10 MB.");
      return;
    }
    void replace(`/api/upload?size=${size}&threshold=${threshold}`, file);
  };
  async function start() {
    if (!target) return;
    setBusy(true);
    setError("");
    setReplayPaused(false);
    try {
      const next = Event.parse(
        await api(
          "/api/start",
          JSON.stringify({
            target_id: target.target_id,
            seed,
            generations,
            population: 16,
            steps: 256,
          }),
        ),
      );
      if (next.target_id !== current.current.target) return;
      socket.current?.close();
      current.current.run = next.run_id;
      setEvent(next);
      const ws = new WebSocket(
        `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/events/${next.run_id}`,
      );
      socket.current = ws;
      ws.onmessage = (message) => {
        try {
          const e = Event.parse(JSON.parse(message.data));
          if (
            e.run_id !== current.current.run ||
            e.target_id !== current.current.target
          )
            return;
          setEvent((previous) => {
            if (previous && previous.revision >= e.revision) return previous;
            return e;
          });
          if (
            e.strokes.length &&
            (strokes.current.length === 0 ||
              e.generation !== Number(ink.current?.dataset.generation))
          ) {
            strokes.current = e.strokes;
            tick.current = 0;
            setDrawn(0);
            clear(ink.current);
            clear(best.current);
            if (ink.current)
              ink.current.dataset.generation = String(e.generation);
            if (best.current)
              e.strokes.forEach((s) => lineRef.current(best.current!, s));
          }
          if (e.error) setError(e.error);
        } catch {
          setError("Trainer sent an invalid progress event.");
        }
      };
      ws.onerror = () =>
        setError("Progress connection failed. Reset and restart the run.");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function pause() {
    const next = !replayPaused;
    setReplayPaused(next);
    if (event?.status === "running" || event?.status === "paused") {
      try {
        await api(`/api/runs/${event.run_id}/${next ? "pause" : "resume"}`);
      } catch (e) {
        setError(String(e));
      }
    }
  }
  async function reset() {
    const run = current.current.run;
    stop();
    if (run) {
      try {
        await api(`/api/runs/${run}/cancel`);
      } catch (e) {
        setError(String(e));
      }
    }
  }
  const running = event?.status === "running" || event?.status === "paused";
  return (
    <main>
      <header>
        <a className="brand" href="/">
          FLY GOGH<span>EXPERIMENTAL ART SCHOOL</span>
        </a>
        <div className="edition">
          LOCAL LAB / № 001
          <br />
          <span>One fly. One pen. Your picture.</span>
        </div>
      </header>
      <section className="intro">
        <div>
          <span className="eyebrow">
            A SMALL BRAIN. AN UNREASONABLE ASSIGNMENT.
          </span>
          <h1>
            Send the fly
            <br />
            to art school<span className="period">.</span>
          </h1>
        </div>
        <p>
          Give it a picture. Watch a tiny neural controller try to turn movement
          into ink.
          <br />
          <em>Talent is very much under investigation.</em>
        </p>
      </section>
      <div className="workspace">
        <aside>
          <section className="panel">
            <h2>
              <span>01</span> The assignment
            </h2>
            <label
              className={`drop ${busy ? "disabled" : ""}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (!busy) upload(e.dataTransfer.files[0]);
              }}
            >
              <strong>+ Drop your picture here</strong>
              <span>or choose a file · PNG / JPG / WebP</span>
              <small>10 MB · 16 megapixels · stays on your device</small>
              <input
                aria-label="Upload picture"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                disabled={busy}
                onChange={(e) => upload(e.target.files?.[0])}
              />
            </label>
            <div className="presets">
              {["cat", "leaf", "circle", "square", "spiral", "free"].map(
                (name) => (
                  <button
                    disabled={busy}
                    key={name}
                    onClick={() =>
                      void replace(
                        `/api/preset/${name}?size=${size}&threshold=${threshold}`,
                      )
                    }
                  >
                    {name}
                  </button>
                ),
              )}
            </div>
            <label className="range">
              Contour threshold <output>{threshold}</output>
              <input
                aria-label="Contour threshold"
                type="range"
                min="5"
                max="150"
                value={threshold}
                onChange={(e) => setThreshold(+e.target.value)}
              />
            </label>
            <div className="row">
              <label>
                Target pixels
                <select
                  aria-label="Target pixels"
                  value={size}
                  onChange={(e) => setSize(+e.target.value)}
                >
                  {[64, 128, 256].map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </select>
              </label>
              <button
                disabled={!target || busy}
                onClick={() =>
                  void replace(
                    `/api/target?size=${size}&threshold=${threshold}`,
                  )
                }
              >
                Apply detail
              </button>
            </div>
            <div className="previews">
              <figure>
                {target ? (
                  <img src={target.original} alt="Original uploaded image" />
                ) : (
                  <div className="empty">YOUR IMAGE</div>
                )}
                <figcaption>Original</figcaption>
              </figure>
              <figure>
                {target ? (
                  <img src={target.target} alt="Simplified sketch target" />
                ) : (
                  <div className="empty">INK TARGET</div>
                )}
                <figcaption>Simplified target</figcaption>
              </figure>
            </div>
            <p className="note">
              Sketch mode reduces color and fine detail to contours. Lower
              thresholds keep more detail.
            </p>
          </section>
          <section className="panel">
            <h2>
              <span>02</span> Art school settings
            </h2>
            <div className="row">
              <label>
                Seed
                <input
                  aria-label="Seed"
                  type="number"
                  min="0"
                  max="2147481648"
                  value={seed}
                  onChange={(e) => setSeed(+e.target.value)}
                />
              </label>
              <label>
                Generations
                <input
                  aria-label="Generations"
                  type="number"
                  min="1"
                  max="100"
                  value={generations}
                  onChange={(e) => setGenerations(+e.target.value)}
                />
              </label>
            </div>
            <p className="note">
              16 candidates · 256 steps · one CPU worker
              <br />
              Per-image training. The reservoir stays fixed.
            </p>
            <button
              className="start"
              disabled={!target || target.blank || busy || running}
              onClick={() => void start()}
            >
              {busy
                ? "Preparing…"
                : target?.free
                  ? "Let the fly wander ↗"
                  : "Send to art school ↗"}
            </button>
            {target?.message && <p role="status">{target.message}</p>}
          </section>
        </aside>
        <section className="drawing-area">
          <div className="canvas-header">
            <span>
              <i /> {event ? event.status.toUpperCase() : "AWAITING ASSIGNMENT"}
            </span>
            <span>
              {target?.free ? "UNTRAINED EXPLORATION" : "BLACK INK / PAPER"}
            </span>
          </div>
          <div className="paper">
            <canvas
              ref={ink}
              width="768"
              height="768"
              aria-label="Live fly drawing"
            />
            <canvas
              className="fly"
              ref={overlay}
              width="768"
              height="768"
              aria-hidden="true"
            />
            {!target && (
              <div className="paper-message">
                <span>↖</span>
                <h2>Every artist starts somewhere.</h2>
                <p>Upload a picture or try the cat.</p>
              </div>
            )}
            <span className="paper-label">FLY GOGH — FIELD NOTES</span>
          </div>
          <div className="transport">
            <button disabled={!event} onClick={() => void pause()}>
              {replayPaused ? "Resume" : "Pause"}
            </button>
            <button disabled={!event} onClick={() => void reset()}>
              Reset
            </button>
            <label>
              Replay speed{" "}
              <select
                aria-label="Replay speed"
                value={speed}
                onChange={(e) => setSpeed(+e.target.value)}
              >
                {[1, 2, 4, 8].map((s) => (
                  <option key={s} value={s}>
                    {s}×
                  </option>
                ))}
              </select>
            </label>
            <button
              disabled={!event?.strokes.length}
              onClick={() =>
                ink.current && download(ink.current, "fly-gogh-drawing.png")
              }
            >
              Export PNG ↓
            </button>
          </div>
          <p className="caption">
            {drawn} / {strokes.current.length} recorded movements · replay of
            the best evaluated attempt. Playback speed does not change
            simulation or training.
          </p>
          <div className="metrics">
            <div>
              <small>GENERATION</small>
              <strong>
                {event?.generation ?? "—"}
                <sub> / {event?.generations ?? generations}</sub>
              </strong>
            </div>
            <div>
              <small>TRAIN SCORE · MEAN</small>
              <strong>{event ? event.score.toFixed(2) : "—"}</strong>
            </div>
            <div>
              <small>TARGET COVERAGE¹</small>
              <strong>
                {event ? ((event.metrics.recall ?? 0) * 100).toFixed(1) : "—"}
                <sub>%</sub>
              </strong>
            </div>
            <div>
              <small>INK PRECISION¹</small>
              <strong>
                {event
                  ? ((event.metrics.precision ?? 0) * 100).toFixed(1)
                  : "—"}
                <sub>%</sub>
              </strong>
            </div>
          </div>
          <p className="caption">
            ¹ Shown training episode. Held-out evaluation is separate; a higher
            training score does not guarantee a better copy.
          </p>
          <div className="under-canvas">
            <div className="best">
              <canvas
                ref={best}
                width="192"
                height="192"
                aria-label="Best evaluated drawing"
              />
              <span>
                Best evaluated attempt
                <br />
                <small>All recorded strokes</small>
              </span>
            </div>
            <div>
              <h3>The brain on the desk</h3>
              <p>
                {graph?.label ?? "Connecting…"}
                <br />
                {graph
                  ? `${graph.nodes.toLocaleString()} nodes / ${graph.edges.toLocaleString()} directed edges`
                  : ""}
              </p>
              <p className="note">
                Measured reservoir |activity|:{" "}
                {event?.activity.toFixed(3) ?? "—"}
                <br />
                CPU · {graph?.readout_parameters ?? 99} trainable readout
                parameters
              </p>
            </div>
          </div>
          {event?.held_out != null && (
            <details>
              <summary>Held-out results · unseen starting seeds</summary>
              <pre>{JSON.stringify(event.held_out, null, 2)}</pre>
            </details>
          )}
        </section>
      </div>
      {error && (
        <div className="error" role="alert">
          {error}
          <button onClick={() => setError("")} aria-label="Dismiss error">
            ×
          </button>
        </div>
      )}
      <footer>
        <strong>Wiring is not a ready-made brain.</strong>
        <p>
          This is an artificial reservoir model with chosen sensory encoding,
          dynamics and rewards. It learns only a small action readout for this
          picture. A synthetic graph is explicitly labeled; real mode uses an
          extracted Male CNS v1.0 subgraph. No consciousness or biological
          drawing ability is implied. Expect exploratory marks and incomplete
          copies.
        </p>
        <a
          href="https://male-cns.janelia.org/download/"
          target="_blank"
          rel="noreferrer"
        >
          Male CNS dataset & attribution ↗
        </a>
      </footer>
    </main>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
