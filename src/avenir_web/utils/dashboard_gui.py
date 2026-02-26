import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import queue
import multiprocessing
import sys
import os
import time
from PIL import Image
import traceback

# Setup debug logging
def log_debug(msg):
    with open("/Volumes/Lexar/Avenir-Web/dashboard_debug.log", "a") as f:
        f.write(f"{msg}\n")

class AvenirDashboardProcess(multiprocessing.Process):
    """
    Runs a native customtkinter GUI dashboard in a separate process to avoid blocking the main agent loop.
    Displays real-time status updates from the agent.
    """
    def __init__(self, data_queue, command_queue=None, viewport_height=1000):
        super().__init__()
        self.data_queue = data_queue
        self.command_queue = command_queue
        self.viewport_height = viewport_height
        self.start_time = time.time()
        self.daemon = True # Ensure process dies when parent dies

    def run(self):
        """Entry point for the separate process."""
        try:
            log_debug("Dashboard process starting...")
            self._setup_gui()
            log_debug("GUI setup complete")
            self._start_update_loop()
            log_debug("Update loop started")
            self.root.mainloop()
            log_debug("Mainloop exited")
        except KeyboardInterrupt:
            log_debug("KeyboardInterrupt")
            pass
        except Exception as e:
            err = traceback.format_exc()
            log_debug(f"Dashboard GUI Error: {e}\n{err}")
            print(f"Dashboard GUI Error: {e}", file=sys.stderr)

    def _setup_gui(self):
        log_debug("Setting up GUI...")
        # Initialize CustomTkinter
        ctk.set_appearance_mode("Dark")
        # We'll override colors manually for the "Avenir" look
        
        log_debug("Creating CTk root...")
        self.root = ctk.CTk()
        self.root.title("Avenir-Web Dashboard")
        
        # Avenir / Elegant Monochrome Theme
        self.bg_color = "#050505"      # Deepest Black
        self.fg_color = "#FFFFFF"      # Pure White
        self.accent_color = "#E0E0E0"  # Platinum/Silver
        self.panel_color = "#0F0F0F"   # Soft Black
        self.border_color = "#222222"  # Subtle Border
        self.card_bg_color = "#141414" # Card Background
        
        self.success_color = "#32CD32" # Lime Green (Classic terminal success)
        self.fail_color = "#FF4500"    # Orange Red
        
        # Initial Window Size and Position
        window_height = self.viewport_height
        window_width = 400
        x_pos = 0
        y_pos = 0
        
        self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        self.root.configure(fg_color=self.bg_color)

        # Fonts - Avenir Preference
        # Try to use Avenir, fall back to Helvetica/Arial
        self.font_family = "Avenir"
        
        self.font_main = (self.font_family, 13)
        self.font_header = (self.font_family, 12, "bold")
        self.font_title = (self.font_family, 18, "bold")
        self.font_subtitle = (self.font_family, 10, "normal")
        self.font_status = (self.font_family, 14, "bold")
        self.font_small_bold = (self.font_family, 11, "bold")
        self.font_small = (self.font_family, 11)
        self.font_mono = ("Menlo", 11) # For code/urls

        # Main Container
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # --- Header Section (Logo + Title) ---
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 15))
        
        # Load Logo
        self.logo_image = None
        try:
            possible_paths = [
                os.path.abspath(os.path.join(os.getcwd(), "img", "icon.png")),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../img/icon.png"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "img/icon.png"),
                "/Volumes/Lexar/Avenir-Web/img/icon.png"
            ]
            
            logo_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    logo_path = p
                    break
            
            if logo_path:
                pil_image = Image.open(logo_path)
                self.logo_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(40, 40))
                logo_label = ctk.CTkLabel(header_frame, text="", image=self.logo_image)
                logo_label.pack(side="left", padx=(0, 10))
        except Exception:
            pass

        # Title Column
        title_column = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_column.pack(side="left", fill="y")
        
        # Center vertically in the frame
        title_label = ctk.CTkLabel(title_column, text="Avenir-Web", 
                                 text_color=self.fg_color, font=self.font_title)
        title_label.pack(side="left", anchor="center", pady=0)
        
        control_column = ctk.CTkFrame(header_frame, fg_color="transparent")
        control_column.pack(side="right", fill="y", anchor="e")

        button_row = ctk.CTkFrame(control_column, fg_color="transparent")
        button_row.pack(anchor="e", pady=(0, 4))

        icon_font = ("Arial", 16)
        
        # Timer moved into button_row - now on the other side
        self.timer_label = ctk.CTkLabel(button_row, text="00:00", text_color="#888", font=self.font_small)
        self.timer_label.pack(side="left", padx=(0, 10))

        self.stop_btn = ctk.CTkButton(button_row, text="⏹", command=lambda: self._send_command("terminate"),
                      fg_color="transparent", border_width=1, border_color="#333",
                      hover_color="#222", text_color="#EF5350",  # Red-ish for Stop
                      font=icon_font, width=28, height=28, corner_radius=4)
        self.stop_btn.pack(side="left", padx=(0, 6))

        self.status_label = ctk.CTkLabel(control_column, text="Idle", text_color=self.fg_color, font=self.font_status)
        self.status_label.pack(anchor="e")

        # --- Objective ---
        self._create_header(main_frame, "OBJECTIVE")
        self.task_label = ctk.CTkLabel(main_frame, text="Initializing...", 
                                     text_color=self.fg_color, font=self.font_main, 
                                     wraplength=350, justify="left", anchor="w")
        self.task_label.pack(fill="x", pady=(2, 5))

        # --- Task URL ---
        self.task_url_label = ctk.CTkLabel(main_frame, text="TASK URL", text_color="#555", font=self.font_small_bold, anchor="w")
        self.task_url_label.pack(fill="x", pady=(0, 0))
        
        self.initial_url_entry = ctk.CTkEntry(main_frame, fg_color=self.panel_color, text_color=self.fg_color,
                                            border_width=0, font=self.font_mono, height=24, corner_radius=4)
        self.initial_url_entry.pack(fill="x", pady=(0, 10))
        self.initial_url_entry.insert(0, "N/A")
        self.initial_url_entry.configure(state="readonly")

        # --- Current Operation Box ---
        self.status_frame = ctk.CTkFrame(main_frame, fg_color=self.panel_color, 
                                  border_color=self.border_color, border_width=1, corner_radius=8)
        self.status_frame.pack(fill="x", pady=(0, 10))
        
        status_inner = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        status_inner.pack(fill="both", padx=10, pady=8)
        
        # Action Display
        ctk.CTkLabel(status_inner, text="CURRENT OPERATION", text_color="#666", font=self.font_small_bold).pack(anchor="w", pady=(0, 0))
        
        # Action Container (Inline)
        self.action_row = ctk.CTkFrame(status_inner, fg_color="transparent")
        self.action_row.pack(fill="x", pady=(0, 0))
        
        # Tool Name (Bold)
        self.action_tool_label = ctk.CTkLabel(self.action_row, text="-", text_color=self.accent_color, 
                                            font=self.font_status, anchor="w")
        self.action_tool_label.pack(side="left")
        
        # Description (Normal) - Vertical Stack (Below tool name)
        # Increased wraplength to allow more text before wrapping
        self.action_desc_label = ctk.CTkLabel(self.action_row, text="", text_color="#BBB", 
                                            font=self.font_main, anchor="w", wraplength=350, justify="left")
        self.action_desc_label.pack(side="left", padx=(6, 0), fill="x", expand=True)

        # --- Strategy & Checklist ---
        self._create_header(main_frame, "STRATEGY")
        
        # Strategy
        # Use transparent frame to match Checklist style (clean, no box)
        strategy_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        strategy_frame.pack(fill="x", expand=False, pady=(0, 5)) # Fixed height
        
        # Increased height for strategy text area to show more content
        self.strategy_text = ctk.CTkTextbox(strategy_frame, height=80, fg_color="transparent", text_color="#AAA", 
                                          font=self.font_main, activate_scrollbars=True, border_width=0)
        # Use minimal padding to align with text flow
        self.strategy_text.pack(fill="both", expand=True, padx=0, pady=0)
        self.strategy_text.configure(state="disabled")

        # Checklist Container
        self.checklist_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.checklist_frame.pack(fill="x", pady=(0, 10))

        # --- Execution History ---
        history_header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        history_header_frame.pack(fill="x", pady=(0, 1))
        
        ctk.CTkLabel(history_header_frame, text="ACTION HISTORY", text_color="#555", font=self.font_header).pack(side="left")
        
        self.history_mode = True 
        self.is_full_history = False
        
        self.history_toggle_btn = ctk.CTkButton(history_header_frame, text="VIEW ALL", command=self._toggle_history,
                                              fg_color="transparent", border_width=1, border_color="#333",
                                              hover_color="#222", text_color="#888", 
                                              font=self.font_small, width=80, height=22, corner_radius=4)
        self.history_toggle_btn.pack(side="right")
        
        # Scrollable container
        # Increase padx for left/right borders (tighter content, more margin)
        self.history_container = ctk.CTkScrollableFrame(main_frame, fg_color="transparent", label_text="")
        self.history_container.pack(fill="both", expand=True, padx=0, pady=(0, 0))
        
        self.full_history_data = []
        self.last_rendered_history_len = 0
        self.last_rendered_history_mode = None

    def _create_header(self, parent, text):
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(header_frame, text=text, text_color="#666", font=self.font_header, anchor="w").pack(side="left")
        ctk.CTkFrame(header_frame, height=1, fg_color="#222").pack(side="left", fill="x", expand=True, padx=(10, 0))

    def _toggle_history(self):
        self.is_full_history = not self.is_full_history
        self.history_toggle_btn.configure(text="VIEW STACK" if self.is_full_history else "VIEW ALL")
        self._refresh_history_view(force=True)

    def _start_update_loop(self):
        self._check_queue()
        self._update_timer()

    def _update_timer(self):
        """Updates the elapsed time timer."""
        elapsed = int(time.time() - self.start_time)
        
        # Format duration
        if elapsed < 3600:
            minutes = elapsed // 60
            seconds = elapsed % 60
            time_str = f"{minutes:02d}:{seconds:02d}"
        else:
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
        self.timer_label.configure(text=time_str)
        self.root.after(1000, self._update_timer)

    def _send_command(self, command):
        if not self.command_queue:
            return
        try:
            self.command_queue.put(command)
            if command == "terminate":
                self.stop_btn.configure(fg_color="#442222")
                self.root.after(200, lambda: self.stop_btn.configure(fg_color="transparent"))
        except Exception:
            pass

    def _check_queue(self):
        try:
            last_data = None
            while True:
                last_data = self.data_queue.get_nowait()
        except queue.Empty:
            pass

        if last_data:
            try:
                self._update_widgets(last_data)
            except Exception as e:
                log_debug(f"Widget update error: {e}\n{traceback.format_exc()}")
        
        self.root.after(100, self._check_queue)

    def _update_widgets(self, data):
        # Update Task
        self.task_label.configure(text=data.get('task', 'No task assigned'))
        
        # Update Status
        raw_status = str(data.get('status', 'IDLE'))
        status_text = raw_status.upper()
        if "WAIT" in status_text:
            display_status = "Waiting"
        elif "THINKING" in status_text:
            display_status = "Thinking"
        elif "PLANNING" in status_text:
            display_status = "Planning"
        elif "REASONING" in status_text:
            display_status = "Reasoning"
        elif "ANALYZING" in status_text:
            display_status = "Analyzing"
        elif "CAPTURING" in status_text:
            display_status = "Capturing"
        elif "STRATEGIZING" in status_text:
            display_status = "Strategy"
        elif "EXECUTING" in status_text:
            display_status = "Executing"
        elif "READY" in status_text:
            display_status = "Ready"
        elif "LOADED" in status_text:
            display_status = "Ready"
        elif "FALLBACK" in status_text:
            display_status = "Fallback"
        elif "ERROR" in status_text:
            display_status = "Error"
        else:
            display_status = "Idle"
        self.status_label.configure(text=display_status)
        
        # Elegant Status Coloring
        if "THINKING" in status_text or "PLANNING" in status_text or "REASONING" in status_text or "ANALYZING" in status_text or "CAPTURING" in status_text or "STRATEGIZING" in status_text:
            self.status_label.configure(text_color="#B8860B") # Dark Goldenrod
        elif "EXECUTING" in status_text or "READY" in status_text or "LOADED" in status_text:
            self.status_label.configure(text_color=self.success_color)
        elif "FALLBACK" in status_text or "ERROR" in status_text:
            self.status_label.configure(text_color=self.fail_color)
        else:
            self.status_label.configure(text_color="#888")

        # Action Display - Split
        action_disp = str(data.get('action_display', 'Waiting...'))
        if not action_disp or action_disp.strip() == "..." or action_disp.lower() == "none":
             action_disp = "Analyzing State..."

        # Attempt to split into Tool + Desc
        parts = action_disp.split(' ', 1)
        tool_name = parts[0]
        desc = parts[1] if len(parts) > 1 else ""
        
        self.action_tool_label.configure(text=tool_name)
        self.action_desc_label.configure(text=desc)
        
        # Update URLs
        self._update_entry(self.initial_url_entry, data.get('initial_url', ''))
        
        # Update Strategy
        self._update_text_area(self.strategy_text, data.get('strategy', ''))
        
        # Update Checklist
        new_checklist = data.get('checklist', [])
        if new_checklist != getattr(self, 'last_checklist_data', None):
            self._render_checklist(new_checklist)
            self.last_checklist_data = new_checklist
        
        # Update History
        self.full_history_data = data.get('history', [])
        self.history_summary = data.get('history_summary') # Capture summary
        self._refresh_history_view()

    def _refresh_history_view(self, force=False):
        current_len = len(self.full_history_data)
        if not force and current_len == self.last_rendered_history_len and self.is_full_history == self.last_rendered_history_mode:
            return

        # Determine what to display
        if self.is_full_history:
            display_list = list(reversed(self.full_history_data)) # Show all, newest first
            start_index = 0
        else:
            # Show last 5, but reversed (newest at top)
            # Original: [1, 2, 3, 4, 5, 6, 7] -> last 5: [3, 4, 5, 6, 7]
            # Reversed: [7, 6, 5, 4, 3]
            if self.full_history_data:
                display_list = list(reversed(self.full_history_data))[:5]
            else:
                display_list = []
            start_index = max(0, current_len - 5)
            
        # Clear container
        for widget in self.history_container.winfo_children():
            widget.destroy()
            
        # Build cards
        for i, item in enumerate(display_list):
            # Calculate correct step number for reversed list
            # If full history: len - i
            # If collapsed (showing last 5): len - i
            step_num = current_len - i
            
            if isinstance(item, dict):
                description = item.get('description', 'Unknown Action')
                success = item.get('success', True)
                error = item.get('error')
            else:
                description = str(item)
                success = True
                error = None
            
            # Visuals
            is_latest = i == 0
            border_col = "#333"
            card_bg = self.card_bg_color
            if not success:
                border_col = "#500"
            elif is_latest:
                border_col = "#777"
                card_bg = "#1A1A1A"
            
            card = ctk.CTkFrame(self.history_container, fg_color=card_bg, 
                              border_color=border_col, border_width=1, corner_radius=4)
            card.pack(fill="x", pady=6, padx=0)
            
            status_dot = "●" if success else "×"
            status_col = "#444" if success else self.fail_color
            if success: status_col = "#2E8B57"
            
            step_col = "#888" if is_latest else "#555"
            
            content_text = description
            if error:
                content_text += f"\nFAILED: {error}"
                
            content_color = "#EEE" if is_latest else "#CCC"
            content_font = self.font_small_bold if is_latest else self.font_small
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=(6, 6))
            ctk.CTkLabel(row, text=f"{step_num:02d}", text_color=step_col, font=self.font_small_bold).pack(side="left")
            ctk.CTkLabel(row, text=status_dot, text_color=status_col, font=self.font_small_bold).pack(side="right")
            ctk.CTkLabel(row, text=content_text, text_color=content_color, 
                       font=content_font, wraplength=280, justify="left", anchor="w").pack(side="left", fill="x", expand=True, padx=(10, 10))

        if not self.is_full_history and current_len > 5:
            ctk.CTkLabel(self.history_container, text=f"• {current_len - 5} previous steps summarized below •", 
                       text_color="#666", font=(self.font_family, 10)).pack(pady=2)

        if hasattr(self, 'history_summary') and self.history_summary:
            summary_card = ctk.CTkFrame(self.history_container, fg_color="#1A1A1A", 
                              border_color="#444", border_width=1, corner_radius=4)
            summary_card.pack(fill="x", pady=6, padx=0)
            
            header = ctk.CTkFrame(summary_card, fg_color="transparent", height=14)
            header.pack(fill="x", padx=10, pady=(2, 0))
            ctk.CTkLabel(header, text="HISTORY SUMMARY", text_color="#AAA", font=self.font_small_bold).pack(side="left")
            
            ctk.CTkLabel(summary_card, text=self.history_summary, text_color="#EEE", 
                       font=self.font_small, wraplength=330, justify="left", anchor="w").pack(fill="x", padx=10, pady=(1, 4))

        self.last_rendered_history_len = current_len
        self.last_rendered_history_mode = self.is_full_history

    def _update_entry(self, entry, text):
        entry.configure(state="normal")
        entry.delete(0, "end")
        entry.insert(0, str(text) if text is not None else "")
        entry.configure(state="readonly")

    def _update_text_area(self, text_widget, text):
        text_widget.configure(state="normal")
        text_widget.delete("1.0", "end")
        text_widget.insert("1.0", str(text) if text is not None else "")
        text_widget.configure(state="disabled")

    def _render_checklist(self, items):
        # Clear
        for widget in self.checklist_frame.winfo_children():
            widget.destroy()
            
        if not items:
            return

        ctk.CTkLabel(self.checklist_frame, text="CHECKLIST", text_color="#666", 
                   font=self.font_small_bold, anchor="w").pack(fill="x", pady=(2, 2))

        for item in items:
            # Handle both old format (string) and new format (dict)
            status = 'pending'
            text = ""
            
            if isinstance(item, dict):
                text = item.get('description', '')
                status = item.get('status', 'pending')
            else:
                # Fallback for old string format
                if "✅" in item:
                    status = 'completed'
                    text = item.replace("✅", "").strip()
                else:
                    status = 'pending'
                    text = item.replace("⬜", "").strip()
            
            row = ctk.CTkFrame(self.checklist_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            
            # Icons and colors based on status
            if status == 'completed' or status == 'success':
                icon = "✓"
                col = self.success_color
                text_col = "#888"
            elif status == 'in_progress' or status == 'inprogress':
                icon = "▶"
                col = "#4FC3F7"  # Light Blue for active
                text_col = "#FFF"  # Highlight active text
            else:
                icon = "○"
                col = "#666"
                text_col = "#CCC"
            
            ctk.CTkLabel(row, text=icon, text_color=col, font=self.font_small_bold, width=18).pack(side="left")
            
            ctk.CTkLabel(row, text=text, text_color=text_col, font=self.font_main, anchor="w").pack(side="left", fill="x")
