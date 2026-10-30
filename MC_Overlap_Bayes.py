'''
Created on Oct 19, 2026

@author: Markus Pfenninger
'''


import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import random
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class RandomisationTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Randomisation Test Simulator")
        self.root.geometry("1000x720")
        self.root.resizable(True, True)
        self.last_simval = None 
        
        # Apply a modern, clean theme
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        self.create_widgets()

    def create_widgets(self):
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=15, pady=15)

        # --- LEFT PANEL (Controls & Text Output) ---
        left_frame = ttk.Frame(main_pane, width=420)
        main_pane.add(left_frame, weight=1)

        # Input Frame
        input_frame = ttk.LabelFrame(left_frame, text="Simulation Parameters", padding=(15, 10))
        input_frame.pack(fill="x", pady=(0, 10))

        self.var_basepop = tk.StringVar(value="17116")
        self.var_draw1 = tk.StringVar(value="358")
        self.var_draw2 = tk.StringVar(value="341")
        self.var_iter = tk.StringVar(value="10000")
        self.var_obs = tk.StringVar(value="15")

        fields = [
            ("Base Population Size:", self.var_basepop),
            ("Draws in Group 1:", self.var_draw1),
            ("Draws in Group 2:", self.var_draw2),
            ("Number of Simulations:", self.var_iter),
            ("Observed Intersection:", self.var_obs)
        ]

        for i, (label_text, var) in enumerate(fields):
            ttk.Label(input_frame, text=label_text).grid(row=i, column=0, sticky="w", pady=5)
            ttk.Entry(input_frame, textvariable=var, width=15).grid(row=i, column=1, sticky="e", pady=5)
            
        input_frame.columnconfigure(1, weight=1)

        # Controls Frame
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill="x", pady=5)

        self.var_bayes = tk.BooleanVar(value=SCIPY_AVAILABLE)
        self.bayes_check = ttk.Checkbutton(
            control_frame, 
            text="Include Bayesian Posterior Estimates (Beta-Binomial)", 
            variable=self.var_bayes,
            state="normal" if SCIPY_AVAILABLE else "disabled"
        )
        self.bayes_check.pack(fill="x", pady=(0, 5))

        self.run_button = ttk.Button(control_frame, text="Run Simulation", command=self.start_simulation)
        self.run_button.pack(fill="x", pady=5)

        self.progress = ttk.Progressbar(control_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=5)

        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(anchor="w")

        # Output Text Frame
        output_frame = ttk.LabelFrame(left_frame, text="Results", padding=(10, 10))
        output_frame.pack(fill="both", expand=True, pady=(10, 0))

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap="word", font=("Consolas", 12), height=15)
        self.output_text.pack(fill="both", expand=True)
        self.output_text.config(state="disabled")

        # Results Export Button
        self.export_txt_button = ttk.Button(output_frame, text="Export Results (TXT/CSV)", command=self.export_results, state="disabled")
        self.export_txt_button.pack(fill="x", pady=(10, 0))

        # --- RIGHT PANEL (Plot) ---
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=2)

        plot_frame = ttk.LabelFrame(right_frame, text="Null Distribution Visualization", padding=(10, 10))
        plot_frame.pack(fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.set_facecolor('#ffffff')
        self.ax.set_title("Simulation Results will appear here", color='gray')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.export_plot_button = ttk.Button(plot_frame, text="Export Plot (PNG/PDF)", command=self.export_plot, state="disabled")
        self.export_plot_button.pack(pady=(10, 0))
        
        self.canvas.draw()

    def log_output(self, message):
        self.output_text.config(state="normal")
        self.output_text.insert("end", message + "\n")
        self.output_text.see("end")
        self.output_text.config(state="disabled")

    def start_simulation(self):
        try:
            params = {
                'basepop': int(self.var_basepop.get()),
                'draw1': int(self.var_draw1.get()),
                'draw2': int(self.var_draw2.get()),
                'iterations': int(self.var_iter.get()),
                'obs': int(self.var_obs.get()),
                'calc_bayes': self.var_bayes.get()
            }
            if params['draw1'] > params['basepop'] or params['draw2'] > params['basepop']:
                raise ValueError("Draw sizes cannot exceed the base population.")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e) if str(e) else "Please enter valid integers.")
            return

        self.run_button.config(state="disabled")
        self.export_plot_button.config(state="disabled")
        self.export_txt_button.config(state="disabled")
        
        self.progress["value"] = 0
        self.progress["maximum"] = params['iterations']
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, "end")
        self.output_text.config(state="disabled")
        
        if not SCIPY_AVAILABLE and params['calc_bayes']:
            self.log_output("Warning: 'scipy' is not installed. Bayesian estimates will be skipped.\n")
            
        self.status_label.config(text="Running simulation...")

        self.ax.clear()
        self.ax.set_title("Running simulation...", color='gray')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

        thread = threading.Thread(target=self.run_calculation, args=(params,), daemon=True)
        thread.start()

    def run_calculation(self, params):
        basepop = params['basepop']
        draw1 = params['draw1']
        draw2 = params['draw2']
        iterations = params['iterations']
        obs = params['obs']

        larger = 0
        smaller = 0
        simval = []

        base_range = range(1, basepop + 1)

        for y in range(iterations):
            set1 = set(random.sample(base_range, draw1))
            set2 = set(random.sample(base_range, draw2))
            
            same = len(set1.intersection(set2))
            simval.append(same)

            if same > obs:
                larger += 1
            else:
                smaller += 1

            if y % 100 == 0:
                self.root.after(0, self.update_progress, y)

        probl = smaller / iterations
        probs = larger / iterations
        mean = sum(simval) / len(simval)
        max_val = max(simval)
        min_val = min(simval)

        results_text = (
            f"--- Input Parameters ---\n"
            f"Size of base population:  {basepop}\n"
            f"Draws in group 1:         {draw1}\n"
            f"Draws in group 2:         {draw2}\n"
            f"Random simulations run:   {iterations}\n"
            f"Observed intersection:    {obs}\n\n"
            f"--- Distribution Statistics ---\n"
            f"Mean of null distribution: {mean:.4f}\n"
            f"Maximum value:             {max_val}\n"
            f"Minimum value:             {min_val}\n\n"
            f"--- Frequentist Probabilities ---\n"
            f"Observed value was {smaller} times larger than simulated values.\n"
            f"Associated probability (p): {probl:.4f}\n\n"
            f"Observed value was {larger} times smaller than or equal to simulated values.\n"
            f"Associated probability (p): {probs:.4f}\n"
        )

        if params['calc_bayes'] and SCIPY_AVAILABLE:
            # Beta(1,1) prior (Uniform) updates to Beta(successes + 1, trials - successes + 1)
            # Probability that null draws are <= observed
            alpha_s = smaller + 1
            beta_s = iterations - smaller + 1
            post_mean_s = alpha_s / (alpha_s + beta_s)
            ci_lower_s = stats.beta.ppf(0.025, alpha_s, beta_s)
            ci_upper_s = stats.beta.ppf(0.975, alpha_s, beta_s)

            # Probability that null draws are > observed
            alpha_l = larger + 1
            beta_l = iterations - larger + 1
            post_mean_l = alpha_l / (alpha_l + beta_l)
            ci_lower_l = stats.beta.ppf(0.025, alpha_l, beta_l)
            ci_upper_l = stats.beta.ppf(0.975, alpha_l, beta_l)

            results_text += (
                f"\n--- Bayesian Posterior Estimates ---\n"
                f"[Assuming uniform Beta(1,1) prior]\n"
                f"Posterior probability (simulated <= observed):\n"
                f"  Mean Estimate: {post_mean_s:.4f}\n"
                f"  95% Credible Interval: [{ci_lower_s:.4f}, {ci_upper_s:.4f}]\n\n"
                f"Posterior probability (simulated > observed):\n"
                f"  Mean Estimate: {post_mean_l:.4f}\n"
                f"  95% Credible Interval: [{ci_lower_l:.4f}, {ci_upper_l:.4f}]\n"
            )

        self.root.after(0, self.finish_simulation, results_text, simval, obs)

    def update_progress(self, current_iter):
        self.progress["value"] = current_iter

    def finish_simulation(self, results_text, simval, obs):
        self.last_simval = simval
        self.progress["value"] = self.progress["maximum"]
        self.log_output(results_text)
        self.status_label.config(text="Simulation complete.")
        
        self.run_button.config(state="normal")
        self.export_plot_button.config(state="normal")
        self.export_txt_button.config(state="normal")
        
        self.draw_plot(simval, obs)

    def draw_plot(self, simval, obs):
        self.ax.clear()
        
        min_val = min(simval)
        max_val = max(simval)
        bins = range(min_val, max(max_val, obs) + 2)

        self.ax.hist(
            simval, 
            bins=bins, 
            align='left', 
            color='#4C72B0', 
            edgecolor='white', 
            alpha=0.8,
            label='Simulated Null Distribution'
        )
        
        self.ax.axvline(
            x=obs, 
            color='#C44E52', 
            linestyle='dashed', 
            linewidth=2, 
            label=f'Observed Value ({obs})'
        )

        self.ax.set_title("Null Distribution vs Observed Intersection", pad=15, fontweight='bold')
        self.ax.set_xlabel("Number of Intersecting Draws")
        self.ax.set_ylabel("Frequency")
        self.ax.grid(axis='y', linestyle='--', alpha=0.7)
        self.ax.legend(loc="upper right")
        
        self.ax.set_xlim(left=max(0, min_val - 1))
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        self.canvas.draw()
        
    def export_plot(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("All Files", "*.*")],
            title="Save Plot As..."
        )
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
                messagebox.showinfo("Export Successful", f"Plot saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save plot:\n{str(e)}")

    def export_results(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Summary Text File", "*.txt"), ("Raw Simulation Data (CSV)", "*.csv"), ("All Files", "*.*")],
            title="Save Results As..."
        )
        if file_path:
            try:
                if file_path.endswith('.csv') and self.last_simval is not None:
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Iteration", "Intersecting_Draws"])
                        for i, val in enumerate(self.last_simval, 1):
                            writer.writerow([i, val])
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.output_text.get(1.0, tk.END).strip())
                messagebox.showinfo("Export Successful", f"Results saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save results:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RandomisationTestApp(root)
    root.mainloop()