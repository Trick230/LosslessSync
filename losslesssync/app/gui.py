"""Tkinter front end for the sync pipeline."""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

from ..audio import media
from ..sync.pipeline import Options, fmt_t, sync

# weight of each pipeline stage in the overall progress bar (start fraction, span)
STAGES = {"decode": (0.00, 0.10), "speed": (0.10, 0.05), "anchors": (0.15, 0.30),
          "correct": (0.45, 0.10), "fine": (0.55, 0.08), "dropouts": (0.63, 0.07), "render": (0.70, 0.10),
          "verify": (0.80, 0.10), "encode": (0.90, 0.10), "done": (1.0, 0.0)}
STAGE_NAMES = {"decode": "a descodificar", "speed": "a detetar velocidade",
               "anchors": "a procurar correspondências", "correct": "a corrigir velocidade",
               "fine": "a afinar offsets", "dropouts": "a procurar falhas de sinal", "render": "a montar",
               "verify": "a verificar",
               "encode": "a codificar", "done": "concluído"}
AUTO = "auto"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LosslessSync")
        self.minsize(820, 600)
        self.q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.streams: dict[str, list[media.AudioStream]] = {"ref": [], "src": []}
        self._build()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self):
        pad = {"padx": 6, "pady": 3}
        files = ttk.LabelFrame(self, text="Ficheiros")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)
        self.var = {k: tk.StringVar() for k in ("ref", "src", "out", "mux")}
        self.track = {k: tk.StringVar(value=AUTO) for k in ("ref", "src")}
        self.track_box = {}
        rows = [("ref", "Referência (Blu-ray, EN/JP)"), ("src", "Áudio a sincronizar (PT)")]
        for r, (key, label) in enumerate(rows):
            ttk.Label(files, text=label).grid(row=2 * r, column=0, sticky="w", **pad)
            ttk.Entry(files, textvariable=self.var[key]).grid(row=2 * r, column=1, sticky="ew", **pad)
            ttk.Button(files, text="…", width=3,
                       command=lambda k=key: self._pick_in(k)).grid(row=2 * r, column=2, **pad)
            ttk.Label(files, text="faixa").grid(row=2 * r + 1, column=0, sticky="e", **pad)
            cb = ttk.Combobox(files, textvariable=self.track[key], state="readonly", values=[AUTO])
            cb.grid(row=2 * r + 1, column=1, columnspan=2, sticky="ew", **pad)
            self.track_box[key] = cb
        ttk.Label(files, text="Saída").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.var["out"]).grid(row=4, column=1, sticky="ew", **pad)
        ttk.Button(files, text="…", width=3, command=self._pick_out).grid(row=4, column=2, **pad)
        self.do_mux = tk.BooleanVar(value=False)
        ttk.Checkbutton(files, text="Criar MKV (referência + faixa PT)", variable=self.do_mux,
                        command=self._mux_default).grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.var["mux"]).grid(row=5, column=1, sticky="ew", **pad)
        ttk.Button(files, text="…", width=3, command=self._pick_mux).grid(row=5, column=2, **pad)

        opts = ttk.LabelFrame(self, text="Opções")
        opts.pack(fill="x", **pad)
        self.speed = tk.StringVar(value="auto")
        self.pitch = tk.StringVar(value="auto")
        self.fill = tk.StringVar(value="silence")
        self.codec = tk.StringVar(value="auto")
        self.bitrate = tk.StringVar(value="")
        self.lang = tk.StringVar(value="por")
        self.fade = tk.StringVar(value="300")
        items = [("Velocidade", self.speed, ["auto", "pal", "pal24", "ntsc", "1", "1001/960"], False),
                 ("Pitch", self.pitch, ["auto", "resample", "tempo", "keep"], True),
                 ("Partes em falta", self.fill, ["silence", "ref"], True),
                 ("Codec", self.codec, ["auto"] + list(media.CODECS), True)]
        for c, (label, v, vals, ro) in enumerate(items):
            ttk.Label(opts, text=label).grid(row=0, column=2 * c, sticky="e", **pad)
            ttk.Combobox(opts, textvariable=v, values=vals, width=10,
                         state="readonly" if ro else "normal").grid(row=0, column=2 * c + 1, **pad)
        ttk.Label(opts, text="Bitrate").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(opts, textvariable=self.bitrate, width=12).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(opts, text="Idioma").grid(row=1, column=2, sticky="e", **pad)
        ttk.Entry(opts, textvariable=self.lang, width=12).grid(row=1, column=3, sticky="w", **pad)
        ttk.Label(opts, text="Fade (ms)").grid(row=1, column=4, sticky="e", **pad)
        ttk.Entry(opts, textvariable=self.fade, width=12).grid(row=1, column=5, sticky="w", **pad)
        self.analyze_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Só analisar", variable=self.analyze_only).grid(
            row=1, column=6, columnspan=2, sticky="w", **pad)

        run = ttk.Frame(self)
        run.pack(fill="x", **pad)
        self.btn = ttk.Button(run, text="Sincronizar", command=self._start)
        self.btn.pack(side="left", **pad)
        self.pbar = ttk.Progressbar(run, maximum=1.0)
        self.pbar.pack(side="left", fill="x", expand=True, **pad)
        self.status = ttk.Label(run, text="pronto", width=30)
        self.status.pack(side="left", **pad)

        pane = ttk.PanedWindow(self, orient="vertical")
        pane.pack(fill="both", expand=True, **pad)
        tf = ttk.LabelFrame(pane, text="Segmentos")
        cols = ("ref_start", "ref_end", "src_start", "src_end", "offset", "method", "spread")
        heads = ("ref início", "ref fim", "fonte início", "fonte fim", "offset (ms)", "método",
                 "dispersão (ms)")
        self.table = ttk.Treeview(tf, columns=cols, show="headings", height=7)
        for c, h in zip(cols, heads):
            self.table.heading(c, text=h)
            self.table.column(c, width=105, anchor="e")
        self.table.pack(fill="both", expand=True)
        self.summary = ttk.Label(tf, text="")
        self.summary.pack(anchor="w")
        pane.add(tf, weight=1)
        lf = ttk.LabelFrame(pane, text="Registo")
        self.log = tk.Text(lf, height=10, wrap="none", font=("Consolas", 9))
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        pane.add(lf, weight=1)

    # ------------------------------------------------------------------ files
    def _pick_in(self, key):
        p = filedialog.askopenfilename(title="Escolher ficheiro", filetypes=[
            ("Média", "*.mkv *.m2ts *.mts *.ts *.mka *.mp4 *.flac *.wav *.ac3 *.eac3 *.dts *.mp2 *.m4a"),
            ("Todos", "*.*")])
        if not p:
            return
        self.var[key].set(p)
        self._load_tracks(key, p)
        if key == "src" and not self.var["out"].get():
            self.var["out"].set(os.path.splitext(p)[0] + "_synced.flac")
        self._mux_default()

    def _load_tracks(self, key, path):
        try:
            st = media.audio_streams(path)
        except media.MediaError as e:
            messagebox.showerror("LosslessSync", str(e))
            st = []
        self.streams[key] = st
        self.track_box[key]["values"] = [AUTO] + [s.describe() for s in st]
        self.track[key].set(AUTO)

    def _pick_out(self):
        p = filedialog.asksaveasfilename(title="Saída", defaultextension=".flac", filetypes=[
            ("FLAC", "*.flac"), ("WAV", "*.wav"), ("AC3", "*.ac3"), ("E-AC3", "*.eac3"),
            ("AAC", "*.m4a"), ("MKA", "*.mka")])
        if p:
            self.var["out"].set(p)

    def _pick_mux(self):
        p = filedialog.asksaveasfilename(title="MKV", defaultextension=".mkv",
                                         filetypes=[("Matroska", "*.mkv")])
        if p:
            self.var["mux"].set(p)
            self.do_mux.set(True)

    def _mux_default(self):
        if self.do_mux.get() and not self.var["mux"].get() and self.var["ref"].get():
            self.var["mux"].set(os.path.splitext(self.var["ref"].get())[0] + "_PT.mkv")

    def _fade_ms(self) -> float:
        try:
            return max(0.0, float(self.fade.get().replace(",", ".")))
        except ValueError:
            return 300.0

    def _track_index(self, key):
        v = self.track[key].get()
        if v == AUTO:
            return None
        for s in self.streams[key]:
            if s.describe() == v:
                return s.aidx
        return None

    # ------------------------------------------------------------------ run
    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        ref, src, out = self.var["ref"].get(), self.var["src"].get(), self.var["out"].get()
        if not ref or not src:
            messagebox.showwarning("LosslessSync", "Escolhe a referência e o áudio a sincronizar.")
            return
        if not out and not self.analyze_only.get():
            messagebox.showwarning("LosslessSync", "Escolhe o ficheiro de saída.")
            return
        report = (os.path.splitext(out)[0] + "_report.json") if out else None
        opt = Options(ref_track=self._track_index("ref"), src_track=self._track_index("src"),
                      speed=self.speed.get().strip() or "auto", pitch=self.pitch.get(),
                      fill=self.fill.get(), fade_ms=self._fade_ms(),
                      codec=None if self.codec.get() == "auto" else self.codec.get(),
                      bitrate=self.bitrate.get().strip() or None,
                      language=self.lang.get().strip() or "por",
                      mux=self.var["mux"].get() if self.do_mux.get() else None,
                      report=report, analyze_only=self.analyze_only.get())
        self.table.delete(*self.table.get_children())
        self.summary.configure(text="")
        self.log.delete("1.0", "end")
        self.pbar["value"] = 0
        self.btn.state(["disabled"])
        self.worker = threading.Thread(target=self._run, args=(ref, src, out or None, opt),
                                       daemon=True)
        self.worker.start()

    def _run(self, ref, src, out, opt):
        try:
            rep = sync(ref, src, out, opt, log_fn=lambda m: self.q.put(("log", m)),
                       progress=lambda st, f: self.q.put(("prog", (st, f))))
            self.q.put(("done", rep))
        except media.MediaError as e:
            self.q.put(("error", str(e)))
        except Exception:  # noqa: BLE001 - show anything to the user
            self.q.put(("error", traceback.format_exc()))

    def _poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", data + "\n")
                    self.log.see("end")
                elif kind == "prog":
                    st, f = data
                    a, span = STAGES.get(st, (self.pbar["value"], 0))
                    self.pbar["value"] = a + span * min(1.0, max(0.0, f))
                    self.status.configure(text=STAGE_NAMES.get(st, st))
                elif kind == "done":
                    self._finish(data)
                elif kind == "error":
                    self.btn.state(["!disabled"])
                    self.status.configure(text="erro")
                    self.log.insert("end", "ERRO: " + data + "\n")
                    self.log.see("end")
                    messagebox.showerror("LosslessSync", data.strip().splitlines()[-1])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finish(self, rep):
        self.btn.state(["!disabled"])
        self.pbar["value"] = 1.0
        self.status.configure(text="concluído")
        for s in rep["segments"]:
            sp = f"{s['spread_ms']:.3f}" if s["spread_ms"] is not None else "-"
            self.table.insert("", "end", values=(
                fmt_t(s["ref_start"]), fmt_t(s["ref_end"]), fmt_t(s["src_start"]),
                fmt_t(s["src_end"]), f"{s['offset_s'] * 1000:.3f}", s["method"], sp))
        txt = (f"velocidade x{rep['speed']:.7f} ({rep['speed_fraction']}), modo {rep['stretch_mode']}, "
               f"cobertura {rep['coverage_pct']:.1f}%, {len(rep['gaps'])} lacuna(s)")
        v = rep.get("verify")
        if v and v.get("measured"):
            txt += (f"  |  verificação: mediana {v['median_abs_ms']:.3f} ms, "
                    f"máx {v['max_abs_ms']:.3f} ms, {v['within_1ms_pct']:.0f}% < 1 ms")
        self.summary.configure(text=txt)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
