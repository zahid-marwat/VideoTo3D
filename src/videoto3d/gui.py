"""Tkinter-based GUI for the VideoTo3D pipeline."""

from __future__ import annotations

import contextlib
import io
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

from .pipeline import PipelineConfig, PipelineOutputs, run_pipeline
from .panorama import is_gpu_available as pano_gpu_available


def _open_path(path: Path) -> None:
    """Open a file or directory with the system default handler."""

    if not path.exists():
        raise FileNotFoundError(path)
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class VideoTo3DApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VideoTo3D")
        self.geometry("960x720")

        self.video_path_var = tk.StringVar()
        self.workspace_var = tk.StringVar(value="workspace")
        self.fps_var = tk.DoubleVar(value=2.0)
        self.max_frames_var = tk.IntVar(value=150)
        self.pano_width_var = tk.IntVar(value=1600)
        self.pano_frames_cap_var = tk.IntVar(value=60)
        self.crop_var = tk.BooleanVar(value=True)
        self.colmap_var = tk.BooleanVar(value=True)
        self.gpu_var = tk.BooleanVar(value=True)
        self._pano_gpu_available = pano_gpu_available()
        self.pano_gpu_var = tk.BooleanVar(value=self._pano_gpu_available)
        self.use_existing_pano_var = tk.BooleanVar(value=False)
        self.existing_pano_path_var = tk.StringVar()
        self.quality_var = tk.StringVar(value="medium")
        self.status_var = tk.StringVar(value="Idle.")

        self.outputs: PipelineOutputs | None = None
        self.panorama_photo: ImageTk.PhotoImage | None = None

        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._progress_values: dict[str, float] = {stage: 0.0 for stage in ("extract", "panorama", "colmap")}
        self._step_progress_bars: dict[str, ttk.Progressbar] = {}

        self._create_widgets()
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------ UI setup
    def _create_widgets(self) -> None:
        padding = {"padx": 12, "pady": 8}

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", **padding)
        controls.columnconfigure(1, weight=1)

        files_frame = ttk.LabelFrame(controls, text="1. Select input")
        files_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        files_frame.columnconfigure(1, weight=1)

        ttk.Label(files_frame, text="Video file:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        video_entry = ttk.Entry(files_frame, textvariable=self.video_path_var, width=60)
        video_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(files_frame, text="Browse…", command=self._browse_video).grid(row=0, column=2, padx=6, pady=4)

        ttk.Label(files_frame, text="Workspace:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        workspace_entry = ttk.Entry(files_frame, textvariable=self.workspace_var, width=60)
        workspace_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(files_frame, text="Browse…", command=self._browse_workspace).grid(row=1, column=2, padx=6, pady=4)

        ttk.Label(files_frame, text="Existing panorama:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.existing_pano_entry = ttk.Entry(files_frame, textvariable=self.existing_pano_path_var, width=60, state="disabled")
        self.existing_pano_entry.grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        self.existing_pano_button = ttk.Button(files_frame, text="Browse…", command=self._browse_existing_panorama, state="disabled")
        self.existing_pano_button.grid(row=2, column=2, padx=6, pady=4)

        options = ttk.LabelFrame(controls, text="2. Configure options")
        options.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        for col in range(4):
            options.columnconfigure(col, weight=1)

        ttk.Label(options, text="Sampling FPS:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(options, textvariable=self.fps_var, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(options, text="Max frames (0 = auto):").grid(row=0, column=2, sticky="w", padx=6, pady=4)
        ttk.Entry(options, textvariable=self.max_frames_var, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(options, text="Panorama width (px, 0 = full):").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.pano_width_entry = ttk.Entry(options, textvariable=self.pano_width_var, width=8)
        self.pano_width_entry.grid(row=1, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(options, text="Panorama frames (0 = all):").grid(row=1, column=2, sticky="w", padx=6, pady=4)
        self.pano_frames_entry = ttk.Entry(options, textvariable=self.pano_frames_cap_var, width=8)
        self.pano_frames_entry.grid(row=1, column=3, sticky="w", padx=6, pady=4)

        self.crop_check = ttk.Checkbutton(options, text="Auto-crop borders", variable=self.crop_var)
        self.crop_check.grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(options, text="Run COLMAP", variable=self.colmap_var).grid(row=2, column=1, sticky="w", padx=6, pady=4)

        ttk.Checkbutton(options, text="Use GPU (COLMAP)", variable=self.gpu_var).grid(row=3, column=0, sticky="w", padx=6, pady=4)

        self.pano_gpu_check = ttk.Checkbutton(
            options,
            text="Use GPU (Panorama)",
            variable=self.pano_gpu_var,
        )
        self.pano_gpu_check.grid(row=3, column=1, sticky="w", padx=6, pady=4)
        if not self._pano_gpu_available:
            self.pano_gpu_check.configure(state="disabled")
            self.pano_gpu_check.configure(text="Use GPU (Panorama - unavailable)")

        self.use_existing_pano_check = ttk.Checkbutton(
            options,
            text="Use existing panorama (skip stitching)",
            variable=self.use_existing_pano_var,
            command=self._on_use_existing_pano_toggle,
        )
        self.use_existing_pano_check.grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        ttk.Label(options, text="COLMAP quality:").grid(row=3, column=2, sticky="e", padx=6, pady=4)
        quality_combo = ttk.Combobox(options, textvariable=self.quality_var, values=["low", "medium", "high"], state="readonly", width=10)
        quality_combo.grid(row=3, column=3, sticky="w", padx=6, pady=4)
        quality_combo.current(1)

        run_frame = ttk.Frame(controls)
        run_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        run_frame.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(run_frame, text="3. Run pipeline", command=self._run_pipeline_clicked)
        self.run_button.grid(row=0, column=0, sticky="w")

        self.overall_progress = ttk.Progressbar(run_frame, mode="determinate", maximum=100)
        self.overall_progress.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        run_frame.columnconfigure(1, weight=1)

        ttk.Label(run_frame, textvariable=self.status_var).grid(row=0, column=2, sticky="e", padx=(12, 0))

        progress_frame = ttk.LabelFrame(controls, text="Step progress")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        progress_frame.columnconfigure(1, weight=1)

        stage_labels = [
            ("extract", "Frame extraction"),
            ("panorama", "Panorama stitching"),
            ("colmap", "3D reconstruction"),
        ]
        for row_index, (stage, label) in enumerate(stage_labels):
            ttk.Label(progress_frame, text=label + ":").grid(row=row_index, column=0, sticky="w", padx=6, pady=4)
            bar = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
            bar.grid(row=row_index, column=1, sticky="ew", padx=6, pady=4)
            self._step_progress_bars[stage] = bar

        log_frame = ttk.LabelFrame(self, text="4. Status log")
        log_frame.grid(row=1, column=0, sticky="nsew", **padding)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_widget = scrolledtext.ScrolledText(log_frame, wrap="word", height=12, state="disabled")
        self.log_widget.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        preview = ttk.LabelFrame(self, text="5. Outputs")
        preview.grid(row=2, column=0, sticky="nsew", **padding)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview, text="Run the pipeline to preview the panorama.")
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        buttons = ttk.Frame(preview)
        buttons.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)

        self.view_pano_btn = ttk.Button(buttons, text="Open panorama", command=self._open_panorama, state="disabled")
        self.view_pano_btn.grid(row=0, column=0, padx=4)

        self.view_cloud_btn = ttk.Button(buttons, text="View point cloud", command=self._open_point_cloud, state="disabled")
        self.view_cloud_btn.grid(row=0, column=1, padx=4)

        self.open_workspace_btn = ttk.Button(buttons, text="Open workspace", command=self._open_workspace, state="disabled")
        self.open_workspace_btn.grid(row=0, column=2, padx=4)

        for row in range(3):
            self.rowconfigure(row, weight=1 if row >= 1 else 0)
        self.columnconfigure(0, weight=1)

        self._on_use_existing_pano_toggle()

    # ------------------------------------------------------------------ helpers
    def _append_log(self, text: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text)
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.configure(state="disabled")

    def _set_controls_state(self, state: str) -> None:
        widgets = [
            self.run_button,
        ]
        for widget in widgets:
            widget.configure(state=state)

    def _reset_progress_bars(self) -> None:
        for stage in self._progress_values:
            self._progress_values[stage] = 0.0
            bar = self._step_progress_bars.get(stage)
            if bar:
                bar["value"] = 0
        self.overall_progress["value"] = 0

    def _update_stage_progress(self, stage: str, value: float) -> None:
        if stage not in self._progress_values:
            return
        self._progress_values[stage] = max(0.0, min(1.0, value))
        bar = self._step_progress_bars.get(stage)
        if bar:
            bar["value"] = self._progress_values[stage] * 100
        self._update_overall_progress()

    def _update_overall_progress(self) -> None:
        if not self._progress_values:
            return
        average = sum(self._progress_values.values()) / len(self._progress_values)
        self.overall_progress["value"] = average * 100

    # ------------------------------------------------------------------ events
    def _browse_video(self) -> None:
        result = filedialog.askopenfilename(title="Select video", filetypes=[("Video files", "*.mp4;*.mov;*.mkv;*.avi;*.m4v"), ("All files", "*.*")])
        if result:
            self.video_path_var.set(result)

    def _browse_workspace(self) -> None:
        result = filedialog.askdirectory(title="Select workspace directory")
        if result:
            self.workspace_var.set(result)

    def _browse_existing_panorama(self) -> None:
        result = filedialog.askopenfilename(
            title="Select panorama image",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if result:
            self.existing_pano_path_var.set(result)

    def _on_use_existing_pano_toggle(self) -> None:
        use_existing = self.use_existing_pano_var.get()
        entry_state = "normal" if use_existing else "disabled"
        self.existing_pano_entry.configure(state=entry_state)
        self.existing_pano_button.configure(state=entry_state)

        dependent_state = "disabled" if use_existing else "normal"
        self.pano_width_entry.configure(state=dependent_state)
        self.pano_frames_entry.configure(state=dependent_state)
        self.crop_check.configure(state=dependent_state)

        if self._pano_gpu_available:
            self.pano_gpu_check.configure(state=dependent_state)
        else:
            # Preserve disabled state when CUDA is unavailable.
            self.pano_gpu_check.configure(state="disabled")
            self.pano_gpu_check.configure(text="Use GPU (Panorama - unavailable)")

        if use_existing:
            self.pano_gpu_var.set(False)

    def _run_pipeline_clicked(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        video_path = Path(self.video_path_var.get()).expanduser()

        use_existing_pano = self.use_existing_pano_var.get()
        existing_pano_path: Path | None = None
        if use_existing_pano:
            existing_value = self.existing_pano_path_var.get().strip()
            if not existing_value:
                messagebox.showerror("Panorama missing", "Please choose an existing panorama image.")
                return
            existing_pano_path = Path(existing_value).expanduser()
            if not existing_pano_path.exists() or not existing_pano_path.is_file():
                messagebox.showerror("Panorama missing", "The selected panorama file does not exist.")
                return

        if not video_path.exists():
            messagebox.showerror("Video missing", "Please choose a valid video file.")
            return

        workspace = Path(self.workspace_var.get()).expanduser()

        try:
            fps = float(self.fps_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid FPS", "Please enter a numeric sampling FPS.")
            return

        try:
            max_frames_input = int(self.max_frames_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid frame limit", "Max frames must be an integer.")
            return

        try:
            pano_width_input = int(self.pano_width_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid width", "Panorama width must be an integer.")
            return

        try:
            pano_cap_input = int(self.pano_frames_cap_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid panorama frame limit", "Panorama frame cap must be an integer.")
            return

        config = PipelineConfig(
            video_path=video_path,
            workspace=workspace,
            frame_rate=fps if fps > 0 else None,
            panorama_crop=self.crop_var.get(),
            panorama_width=pano_width_input if pano_width_input > 0 else None,
            panorama_max_images=pano_cap_input if pano_cap_input > 0 else None,
            panorama_use_gpu=(self.pano_gpu_var.get() and self._pano_gpu_available) if not use_existing_pano else False,
            existing_panorama=existing_pano_path,
            run_colmap=self.colmap_var.get(),
            colmap_quality=self.quality_var.get(),
            use_gpu=self.gpu_var.get(),
            max_frames=max_frames_input if max_frames_input > 0 else None,
        )

        self.outputs = None
        self.panorama_photo = None
        self.preview_label.configure(image="", text="Processing…")
        self.view_pano_btn.configure(state="disabled")
        self.view_cloud_btn.configure(state="disabled")
        self.open_workspace_btn.configure(state="disabled")

        self._clear_log()
        self.status_var.set("Running…")
        self._reset_progress_bars()
        self._set_controls_state("disabled")

        self._worker = threading.Thread(
            target=self._run_pipeline_thread,
            args=(config,),
            daemon=True,
        )
        self._worker.start()

    def _run_pipeline_thread(self, config: PipelineConfig) -> None:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                outputs = run_pipeline(
                    config,
                    status_callback=lambda msg: self._queue.put(("log", f"{msg}\n")),
                    progress_callback=lambda stage, value: self._queue.put(("progress", (stage, value))),
                )
        except Exception as exc:  # noqa: BLE001
            log_dump = buffer.getvalue()
            self._queue.put(("error", f"{exc}\n{log_dump}"))
        else:
            captured = buffer.getvalue()
            if captured:
                self._queue.put(("log", captured))
            self._queue.put(("done", outputs))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "progress":
                    stage, value = payload if isinstance(payload, tuple) else (None, None)
                    if stage is not None and value is not None:
                        self._update_stage_progress(str(stage), float(value))
                elif kind == "done":
                    self._on_pipeline_complete(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    self._on_pipeline_error(str(payload))
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _on_pipeline_complete(self, outputs: PipelineOutputs) -> None:
        self.outputs = outputs
        self.status_var.set("Completed.")
        self._set_controls_state("normal")
        self.open_workspace_btn.configure(state="normal")
        self._update_overall_progress()
        self.overall_progress["value"] = 100

        pano_path = outputs.panorama_path
        if pano_path.exists():
            self._show_panorama(pano_path)
            self.view_pano_btn.configure(state="normal")
        else:
            self.preview_label.configure(text="Panorama not found.")

        if outputs.point_cloud_path and outputs.point_cloud_path.exists():
            self.view_cloud_btn.configure(state="normal")
        else:
            self._append_log("Point cloud not available.\n")

    def _on_pipeline_error(self, details: str) -> None:
        self.status_var.set("Failed.")
        self._set_controls_state("normal")
        messagebox.showerror("Pipeline failed", details.strip())

    def _show_panorama(self, pano_path: Path) -> None:
        try:
            image = Image.open(pano_path)
        except Exception as exc:  # noqa: BLE001
            self.preview_label.configure(text=f"Failed to load panorama: {exc}")
            return

        max_size = (900, 360)
        image.thumbnail(max_size, Image.LANCZOS)
        self.panorama_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.panorama_photo, text="")

    def _open_panorama(self) -> None:
        if not self.outputs:
            return
        pano_path = self.outputs.panorama_path
        try:
            _open_path(pano_path)
        except FileNotFoundError:
            messagebox.showerror("Missing file", "Panorama image not found on disk.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open failed", str(exc))

    def _open_point_cloud(self) -> None:
        if not self.outputs or not self.outputs.point_cloud_path:
            return
        path = self.outputs.point_cloud_path
        if not path.exists():
            messagebox.showerror("Missing file", "Point cloud file not found.")
            return
        try:
            import open3d as o3d  # type: ignore[import-not-found]

            pcd = o3d.io.read_point_cloud(str(path))
            if pcd.is_empty():
                messagebox.showwarning("Empty cloud", "The point cloud appears to be empty.")
                return
            o3d.visualization.draw_geometries([pcd], window_name="VideoTo3D Point Cloud")
        except ImportError:
            try:
                _open_path(path)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Open failed", str(exc))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Point cloud error", str(exc))

    def _open_workspace(self) -> None:
        if not self.outputs:
            return
        workspace = self.outputs.frames_dir.parent
        try:
            _open_path(workspace)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open failed", str(exc))


def main() -> None:
    app = VideoTo3DApp()
    app.mainloop()


if __name__ == "__main__":
    main()
