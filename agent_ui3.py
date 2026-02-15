"""
AI Research Agent - Windows Desktop Application
MODERN UI VERSION 3.2 (Precision Link Extraction)
"""
import sys
import os
import json
import threading
import queue
import asyncio
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx 

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from playwright.async_api import async_playwright

# Try importing the backend logic
try:
    from agent import SmartResearchAgent
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agent module: {e}")
    AGENT_AVAILABLE = False

# --- THEME CONFIGURATION ---
COLORS = {
    'bg': '#1e1e1e',          
    'sidebar': '#252526',     
    'card': '#333333',        
    'text': '#ffffff',        
    'subtext': '#cccccc',     
    'accent': '#007acc',      
    'accent_hover': '#0098ff',
    'success': '#4ec9b0',     
    'warning': '#cca700',     
    'error': '#f48771',       
    'border': '#3e3e42',      
    'select': '#264f78'       
}

FONTS = {
    'header': ('Segoe UI', 16, 'bold'),
    'subheader': ('Segoe UI', 12, 'bold'),
    'normal': ('Segoe UI', 10),
    'small': ('Segoe UI', 9),
    'code': ('Consolas', 10)
}

# --- DATA MANAGER ---
class DataManager:
    """Manages saved topics and settings"""
    def __init__(self):
        self.filename = "agent_data.json"
        self.data = self.load()
    
    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except: pass
        return {
            "topics": ["Artificial Intelligence", "Python Automation", "Climate Change"],
            "urls": ["https://news.ycombinator.com", "https://arxiv.org", "https://reddit.com/r/technology"],
            "presets": {
                "General Research": {"look": "Articles, papers, tutorials", "ignore": "Ads, login pages"},
                "News Scan": {"look": "Breaking news, headlines", "ignore": "Opinion, archive, navigation"}
            }
        }

    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e: print(f"Error saving: {e}")

data_manager = DataManager()

# --- CUSTOM WIDGETS ---
class ModernButton(tk.Label):
    """Custom flat button widget"""
    def __init__(self, parent, text, command, bg=COLORS['accent'], fg='white', width=None):
        super().__init__(parent, text=text, bg=bg, fg=fg, 
                         font=('Segoe UI', 10, 'bold'), cursor='hand2', padx=15, pady=8)
        self.command = command
        self.default_bg = bg
        self.hover_bg = COLORS['accent_hover'] if bg == COLORS['accent'] else '#555555'
        if width: self.configure(width=width)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)
        
    def on_enter(self, e): 
        if self.cget('state') != 'disabled': self.configure(bg=self.hover_bg)
    def on_leave(self, e): 
        if self.cget('state') != 'disabled': self.configure(bg=self.default_bg)
    def on_click(self, e): 
        if self.cget('state') != 'disabled': self.command()
    def set_state(self, state):
        self.configure(state=state)
        if state == 'disabled': self.configure(bg='#444444', fg='#888888', cursor='arrow')
        else: self.configure(bg=self.default_bg, fg='white', cursor='hand2')

# --- DIALOGS ---
class ManagerDialog(tk.Toplevel):
    """Dialog to Add/Edit/Delete List Items"""
    def __init__(self, parent, title, items, on_save):
        super().__init__(parent)
        self.title(title)
        self.geometry("500x400")
        self.configure(bg=COLORS['bg'])
        self.items = list(items)
        self.on_save = on_save
        self.transient(parent)
        self.grab_set()
        
        main_frame = tk.Frame(self, bg=COLORS['bg'], padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        list_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        list_frame.pack(fill='both', expand=True)
        
        self.listbox = tk.Listbox(list_frame, bg=COLORS['card'], fg=COLORS['text'],
                                 selectbackground=COLORS['select'], font=FONTS['normal'],
                                 borderwidth=0, highlightthickness=1, highlightbackground=COLORS['border'],
                                 selectmode='extended')
        self.listbox.pack(side='left', fill='both', expand=True)
        
        scroll = ttk.Scrollbar(list_frame, command=self.listbox.yview)
        scroll.pack(side='right', fill='y')
        self.listbox.config(yscrollcommand=scroll.set)
        self.refresh()
            
        btn_frame = tk.Frame(main_frame, bg=COLORS['bg'], pady=10)
        btn_frame.pack(fill='x')
        ModernButton(btn_frame, "Add", self.add_item, bg=COLORS['success']).pack(side='left', padx=2)
        ModernButton(btn_frame, "Delete", self.delete_items, bg=COLORS['error']).pack(side='left', padx=2)
        ModernButton(btn_frame, "Save", self.save_close, bg=COLORS['sidebar']).pack(side='right', padx=2)

    def add_item(self):
        val = simpledialog.askstring("Add", "New item:", parent=self)
        if val and val.strip():
            self.items.append(val.strip())
            self.refresh()

    def delete_items(self):
        sel = self.listbox.curselection()
        for i in reversed(sel): del self.items[i]
        self.refresh()

    def refresh(self):
        self.listbox.delete(0, 'end')
        for item in self.items: self.listbox.insert('end', item)

    def save_close(self):
        self.on_save(self.items)
        self.destroy()

class LinkSelectorDialog(tk.Toplevel):
    """Dialog to manually select which links to process with counter"""
    def __init__(self, parent, links):
        super().__init__(parent)
        self.title(f"Select Links to Analyze ({len(links)} found)")
        self.geometry("950x600")
        self.configure(bg=COLORS['bg'])
        self.result = None
        self.links = links
        self.transient(parent)
        self.grab_set()
        
        main_frame = tk.Frame(self, bg=COLORS['bg'], padx=10, pady=10)
        main_frame.pack(fill='both', expand=True)
        
        # Counter label
        self.counter_var = tk.StringVar(value=f"0 selected / {len(links)} total")
        counter_label = tk.Label(main_frame, textvariable=self.counter_var, 
                               bg=COLORS['bg'], fg=COLORS['accent'], font=('Segoe UI', 11, 'bold'))
        counter_label.pack(anchor='w', pady=(0, 10))
        
        # Treeview
        cols = ("Text", "URL")
        self.tree = ttk.Treeview(main_frame, columns=cols, show='headings', selectmode='extended')
        self.tree.heading("Text", text="Link Text")
        self.tree.heading("URL", text="URL")
        self.tree.column("Text", width=600)
        self.tree.column("URL", width=300)
        self.tree.pack(side='left', fill='both', expand=True)
        
        vsb = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        
        for i, link in enumerate(links):
            self.tree.insert("", "end", iid=str(i), values=(link['text'][:100], link['url'][:80]))
        
        # Bind selection change to update counter
        self.tree.bind('<<TreeviewSelect>>', self.update_counter)
        
        # Buttons
        btn_frame = tk.Frame(self, bg=COLORS['sidebar'], pady=10, padx=10)
        btn_frame.pack(fill='x')
        
        ModernButton(btn_frame, "Select All", self.select_all, bg=COLORS['card']).pack(side='left', padx=5)
        ModernButton(btn_frame, "Clear Selection", self.clear_selection, bg=COLORS['warning']).pack(side='left', padx=5)
        ModernButton(btn_frame, "Start Research", self.on_confirm, bg=COLORS['success']).pack(side='right', padx=5)
        self.select_all()

    def update_counter(self, event=None):
        selected = len(self.tree.selection())
        total = len(self.links)
        self.counter_var.set(f"{selected} selected / {total} total")
        self.start_btn_state(selected > 0)

    def start_btn_state(self, enabled):
        # Find the Start Research button and enable/disable
        pass  # Optional enhancement

    def select_all(self):
        self.tree.selection_set(self.tree.get_children())
        self.update_counter()

    def clear_selection(self):
        self.tree.selection_remove(self.tree.selection())
        self.update_counter()

    def on_confirm(self):
        selected_ids = self.tree.selection()
        if not selected_ids:
            messagebox.showwarning("No Selection", "Please select at least one link.")
            return
        self.result = [self.links[int(i)] for i in selected_ids]
        self.destroy()

class ResultsDialog(tk.Toplevel):
    """Dialog to view, select, and export research results"""
    def __init__(self, parent, results):
        super().__init__(parent)
        self.title(f"Research Results ({len(results)} found)")
        self.geometry("1100x700")
        self.configure(bg=COLORS['bg'])
        self.results = results
        self.selected_indices = set()
        self.transient(parent)
        self.grab_set()
        
        main_frame = tk.Frame(self, bg=COLORS['bg'], padx=15, pady=15)
        main_frame.pack(fill='both', expand=True)
        
        # Header with counter and export button
        header = tk.Frame(main_frame, bg=COLORS['bg'])
        header.pack(fill='x', pady=(0, 15))
        
        self.counter_var = tk.StringVar(value=f"0 selected / {len(results)} total")
        counter_label = tk.Label(header, textvariable=self.counter_var, 
                               bg=COLORS['bg'], fg=COLORS['accent'], font=('Segoe UI', 12, 'bold'))
        counter_label.pack(side='left')
        
        ModernButton(header, "Export Selected", self.export_results, 
                    bg=COLORS['success']).pack(side='right', padx=(10, 0))
        ModernButton(header, "Select All", self.select_all, 
                    bg=COLORS['card']).pack(side='right', padx=(5, 10))
        ModernButton(header, "Clear Selection", self.clear_selection, 
                    bg=COLORS['warning']).pack(side='right', padx=(5, 10))
        
        # Results table
        cols = ("Score", "Title", "URL")
        self.tree = ttk.Treeview(main_frame, columns=cols, show='headings', selectmode='none')
        self.tree.heading("Score", text="Relevance")
        self.tree.heading("Title", text="Title & Summary")
        self.tree.heading("URL", text="Source URL")
        self.tree.column("Score", width=100, anchor='center')
        self.tree.column("Title", width=600)
        self.tree.column("URL", width=300)
        
        vsb = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        
        # Add checkbox functionality
        self.checkboxes = {}
        for i, res in enumerate(results):
            score = f"{res.get('relevance_score', 0):.2f}"
            title = f"{res.get('title', 'Untitled')}\n  → {res.get('summary', '')[:80]}..."
            url = res.get('url', '')[:60]
            
            item_id = self.tree.insert("", "end", values=(score, title, url))
            self.checkboxes[item_id] = i
            self.tree.tag_bind(item_id, '<Button-1>', lambda e, idx=i, iid=item_id: self.toggle_selection(idx, iid))
        
        # Footer with details view
        detail_frame = tk.LabelFrame(main_frame, text="Details", bg=COLORS['bg'], 
                                   fg=COLORS['subtext'], font=FONTS['subheader'], pady=10)
        detail_frame.pack(fill='x', pady=(15, 0))
        
        self.detail_text = scrolledtext.ScrolledText(detail_frame, height=6, 
                                                   bg=COLORS['card'], fg='white',
                                                   font=FONTS['normal'], wrap='word')
        self.detail_text.pack(fill='x')
        self.tree.bind('<<TreeviewSelect>>', self.show_details)
        
        # Double-click to open URL
        self.tree.bind('<Double-1>', self.open_selected_url)

    def toggle_selection(self, idx, item_id):
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
            self.tree.item(item_id, tags=())
        else:
            self.selected_indices.add(idx)
            self.tree.item(item_id, tags=('selected',))
        
        self.tree.tag_configure('selected', background=COLORS['select'])
        self.update_counter()

    def update_counter(self):
        self.counter_var.set(f"{len(self.selected_indices)} selected / {len(self.results)} total")

    def select_all(self):
        self.selected_indices = set(range(len(self.results)))
        for item_id, idx in self.checkboxes.items():
            self.tree.item(item_id, tags=('selected',))
        self.tree.tag_configure('selected', background=COLORS['select'])
        self.update_counter()

    def clear_selection(self):
        self.selected_indices.clear()
        for item_id in self.checkboxes.keys():
            self.tree.item(item_id, tags=())
        self.update_counter()

    def show_details(self, event=None):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            idx = self.checkboxes.get(selection[0])
            if idx is not None and idx < len(self.results):
                res = self.results[idx]
                details = (
                    f"Title: {res.get('title', 'N/A')}\n"
                    f"URL: {res.get('url', 'N/A')}\n"
                    f"Relevance Score: {res.get('relevance_score', 0):.3f}\n"
                    f"Reason: {res.get('reason', 'N/A')}\n"
                    f"Summary: {res.get('summary', 'N/A')}"
                )
                self.detail_text.delete('1.0', 'end')
                self.detail_text.insert('1.0', details)

    def open_selected_url(self, event=None):
        selection = self.tree.selection()
        if selection:
            idx = self.checkboxes.get(selection[0])
            if idx is not None and idx < len(self.results):
                url = self.results[idx].get('url')
                if url:
                    import webbrowser
                    webbrowser.open_new_tab(url)

    def export_results(self):
        if not self.selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one result to export.")
            return
        
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Results"
        )
        
        if not filepath:
            return
        
        selected_results = [self.results[i] for i in sorted(self.selected_indices)]
        
        try:
            if filepath.endswith('.json'):
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(selected_results, f, indent=2, ensure_ascii=False)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    for i, res in enumerate(selected_results, 1):
                        f.write(f"RESULT #{i}\n")
                        f.write(f"Title: {res.get('title', 'N/A')}\n")
                        f.write(f"URL: {res.get('url', 'N/A')}\n")
                        f.write(f"Score: {res.get('relevance_score', 0):.3f}\n")
                        f.write(f"Summary: {res.get('summary', 'N/A')}\n")
                        f.write(f"Reason: {res.get('reason', 'N/A')}\n")
                        f.write("-" * 80 + "\n\n")
            
            messagebox.showinfo("Export Successful", 
                              f"Exported {len(selected_results)} results to:\n{filepath}")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

# --- AGENT LOGIC ---
class ModernUIAgent(SmartResearchAgent):
    def __init__(self, model_name, ui_callback, ignore_terms=None):
        super().__init__(model_name)
        self.ui_callback = ui_callback
        self.answers = queue.Queue()
        self.stop_flag = False
        self.ignore_terms = ignore_terms or []

    async def _extract_links_smart(self, page, topic: str, start_url: str) -> List[Dict]:
        """
        PRECISION LINK EXTRACTOR (Cluster & Heuristic Based)
        Finds the 'signal' (articles) and drops the 'noise' (nav, footer).
        """
        if self.ui_callback:
            self.ui_callback('log', {'type': 'action', 'message': '🔍 Scanning for content links...'})
        
        try:
            links = await page.evaluate("""
                () => {
                    const host = window.location.hostname;
                    const links = [];
                    
                    // --- STRATEGY 1: Site Specific (High Precision) ---
                    if (host.includes('news.ycombinator.com')) {
                        // Hacker News Main Links
                        const anchors = document.querySelectorAll('.titleline > a');
                        for (const a of anchors) links.push({text: a.textContent, url: a.href});
                        return links;
                    }
                    if (host.includes('reddit.com')) {
                        // Reddit Post Links
                        const anchors = document.querySelectorAll('a[data-click-id="body"], a.title');
                        for (const a of anchors) {
                            if (a.textContent.length > 10) links.push({text: a.textContent, url: a.href});
                        }
                        return links;
                    }

                    // --- STRATEGY 2: Cluster Analysis (Generic) ---
                    const allAnchors = Array.from(document.querySelectorAll('a[href]'));
                    const candidates = [];
                    
                    const junkTerms = ['login', 'signup', 'register', 'home', 'about', 'contact', 'privacy', 
                                     'terms', 'comment', 'share', 'reply', 'hide', 'report', 'save', 
                                     'forgot', 'password', 'user', 'profile', 'admin'];

                    for (const a of allAnchors) {
                        const text = a.textContent.trim();
                        const url = a.href;
                        
                        // 1. Basic Filters
                        if (!url || !url.startsWith('http')) continue;
                        if (text.length < 15) continue; // Articles usually have longer titles (>15 chars)
                        
                        // 2. Junk Word Filter
                        const lowerText = text.toLowerCase();
                        if (junkTerms.some(term => lowerText.includes(term))) continue;
                        
                        // 3. Location Filter (Ignore Nav/Footer/Sidebar)
                        const navCheck = a.closest('nav, header, footer, .sidebar, .menu, .navigation');
                        if (navCheck) continue;

                        candidates.push({ text: text, url: url });
                    }
                    
                    return candidates;
                }
            """)
            
            # Deduplicate
            seen = set()
            unique_links = []
            for link in links:
                # Remove query params for deduplication mostly
                clean_url = link['url'].split('#')[0]
                if clean_url not in seen:
                    seen.add(clean_url)
                    unique_links.append(link)
            
            if self.ui_callback:
                self.ui_callback('log', {'type': 'success', 'message': f'📊 Found {len(unique_links)} relevant links.'})
                
            return unique_links
            
        except Exception as e:
            if self.ui_callback:
                self.ui_callback('log', {'type': 'error', 'message': f'Extraction error: {e}'})
            return []

    async def research_ui(self, topic, start_url, countries, strict_countries, max_batches=0):
        """Main Loop with batch limiting capability"""
        if self.ui_callback:
            self.ui_callback('log', {'type': 'info', 'message': f'🚀 Starting research: {topic}'})

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                self.ui_callback('log', {'type': 'action', 'message': f'🌐 Loading {start_url}...'})
                await page.goto(start_url, wait_until='domcontentloaded')
                await asyncio.sleep(2)
                
                # 1. Extract
                all_links = await self._extract_links_smart(page, topic, start_url)
                if not all_links:
                    self.ui_callback('log', {'type': 'error', 'message': 'No links found.'})
                    return []

                # 2. Select
                self.ui_callback('request_selection', all_links)
                loop = asyncio.get_event_loop()
                selected_links = await loop.run_in_executor(None, self.answers.get)
                
                if not selected_links or selected_links == "STOPPED":
                    self.ui_callback('log', {'type': 'warning', 'message': 'Research cancelled.'})
                    return []
                
                self.ui_callback('log', {'type': 'success', 'message': f'✅ Analyzing {len(selected_links)} selected links...'})

                # 3. Evaluate with batch limiting
                chunk_size = 10
                promising_links = []
                total_batches = (len(selected_links) + chunk_size - 1) // chunk_size
                batches_to_process = min(total_batches, max_batches) if max_batches > 0 else total_batches
                
                if max_batches > 0 and self.ui_callback:
                    self.ui_callback('log', {'type': 'info', 
                        'message': f'⚠️ Processing {batches_to_process}/{total_batches} batches (limit: {max_batches})'})
                
                for i in range(batches_to_process):
                    if self.stop_flag: break
                    chunk = selected_links[i*chunk_size:(i+1)*chunk_size]
                    results = await self.evaluator.evaluate_batch(topic, chunk)
                    for link, res in zip(chunk, results):
                        if res['should_click']:
                            promising_links.append({**link, **res})
                            self.ui_callback('log', {'type': 'decision', 'message': f"✓ FOUND: {link['text'][:40]}..."})

                # 4. Visit
                if not promising_links:
                    self.ui_callback('log', {'type': 'warning', 'message': 'No relevant links found.'})
                    return []

                final_results = []
                for link_info in promising_links:
                    if self.stop_flag: break
                    url = link_info['url']
                    self.ui_callback('log', {'type': 'action', 'message': f"🔗 Visiting: {link_info['text'][:50]}..."})
                    try:
                        new_page = await context.new_page()
                        await new_page.goto(url, wait_until='domcontentloaded')
                        await asyncio.sleep(2)
                        title, content, _ = await self.evaluator.extract_complete_page_content(new_page, url)
                        evaluation = await self.evaluator.evaluate_after_click(topic, url, title, content)
                        if evaluation.get('is_relevant', False):
                            self.ui_callback('log', {'type': 'success', 'message': f"✅ RELEVANT! Score: {evaluation.get('relevance_score',0)}"})
                            final_results.append({
                                'title': title,
                                'url': url,
                                'relevance_score': evaluation.get('relevance_score', 0),
                                'summary': evaluation.get('summary', ''),
                                'reason': evaluation.get('reason', '')
                            })
                        await new_page.close()
                    except Exception as e:
                        self.ui_callback('log', {'type': 'error', 'message': f"Error visiting {url}: {e}"})

                self.ui_callback('log', {'type': 'success', 'message': f'🎉 Done. {len(final_results)} results.'})
                return final_results

            except Exception as e: 
                traceback.print_exc()
                self.ui_callback('log', {'type': 'error', 'message': str(e)})
            finally:
                await browser.close()


# --- MAIN APP ---
class ResearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Research Agent Pro")
        self.geometry("1400x900")
        self.configure(bg=COLORS['bg'])
        self.setup_styles()
        
        self.agent = None
        self.agent_running = False
        
        self.build_ui()
        self.load_presets()
        
        # Check connection immediately
        self.after(100, self.check_ollama)

    def show_results_window(self, results):
        if results:
            ResultsDialog(self, results)
        else:
            messagebox.showinfo("No Results", "No relevant results were found during research.")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox", fieldbackground=COLORS['card'], background=COLORS['sidebar'], 
                        foreground='white', arrowcolor='white', borderwidth=0)
        style.map("TCombobox", fieldbackground=[('readonly', COLORS['card'])])
        style.configure("Vertical.TScrollbar", background=COLORS['card'], troughcolor=COLORS['bg'], borderwidth=0)
        style.configure("Treeview", background=COLORS['card'], foreground=COLORS['text'], 
                        fieldbackground=COLORS['card'], borderwidth=0)
        style.configure("Treeview.Heading", background=COLORS['sidebar'], foreground='white', borderwidth=0)

    def build_ui(self):
        # HEADER
        header = tk.Frame(self, bg=COLORS['sidebar'], height=60, pady=10, padx=20)
        header.pack(fill='x')
        
        tk.Label(header, text="AI RESEARCH AGENT", font=FONTS['header'], bg=COLORS['sidebar'], fg='white').pack(side='left')
        
        # Right Header: Status + Model
        right_header = tk.Frame(header, bg=COLORS['sidebar'])
        right_header.pack(side='right')
        
        # Status Light
        self.status_canvas = tk.Canvas(right_header, width=15, height=15, bg=COLORS['sidebar'], highlightthickness=0)
        self.status_light = self.status_canvas.create_oval(2, 2, 13, 13, fill=COLORS['error'], outline="")
        self.status_canvas.pack(side='left', padx=(0, 10))
        
        # Model Dropdown
        tk.Label(right_header, text="Model:", bg=COLORS['sidebar'], fg='white').pack(side='left', padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(right_header, textvariable=self.model_var, state="readonly", width=20)
        self.model_combo.pack(side='left')

        # CONTENT
        content = tk.Frame(self, bg=COLORS['bg'])
        content.pack(fill='both', expand=True, padx=20, pady=20)
        
        left_panel = tk.Frame(content, bg=COLORS['bg'], width=450)
        left_panel.pack(side='left', fill='y', padx=(0, 20))
        left_panel.pack_propagate(False)
        self.build_controls(left_panel)
        
        right_panel = tk.Frame(content, bg=COLORS['bg'])
        right_panel.pack(side='right', fill='both', expand=True)
        self.build_logs(right_panel)

    def build_controls(self, parent):
        # Topic
        self.create_label(parent, "Research Topic")
        topic_frame = tk.Frame(parent, bg=COLORS['bg'])
        topic_frame.pack(fill='x', pady=(0, 15))
        self.topic_var = tk.StringVar()
        self.topic_combo = ttk.Combobox(topic_frame, textvariable=self.topic_var, font=FONTS['normal'])
        self.topic_combo['values'] = data_manager.data['topics']
        if self.topic_combo['values']: self.topic_combo.current(0)
        self.topic_combo.pack(side='left', fill='x', expand=True)
        ModernButton(topic_frame, "⚙", lambda: self.manage_list("topics", self.topic_combo), bg=COLORS['card'], width=3).pack(side='right', padx=(5,0))

        # URL
        self.create_label(parent, "Start URL")
        url_frame = tk.Frame(parent, bg=COLORS['bg'])
        url_frame.pack(fill='x', pady=(0, 15))
        self.url_var = tk.StringVar()
        self.url_combo = ttk.Combobox(url_frame, textvariable=self.url_var, font=FONTS['normal'])
        self.url_combo['values'] = data_manager.data['urls']
        if self.url_combo['values']: self.url_combo.current(0)
        self.url_combo.pack(side='left', fill='x', expand=True)
        ModernButton(url_frame, "⚙", lambda: self.manage_list("urls", self.url_combo), bg=COLORS['card'], width=3).pack(side='right', padx=(5,0))

        # Countries
        self.create_label(parent, "Countries")
        self.country_entry = tk.Entry(parent, bg=COLORS['card'], fg='white', insertbackground='white', font=FONTS['normal'], relief='flat')
        self.country_entry.pack(fill='x', ipady=5, pady=(0, 5))
        
        self.strict_var = tk.BooleanVar(value=False)
        radio_frame = tk.Frame(parent, bg=COLORS['bg'])
        radio_frame.pack(fill='x', pady=(0, 15))
        tk.Radiobutton(radio_frame, text="Optional", variable=self.strict_var, value=False, bg=COLORS['bg'], fg='white', selectcolor=COLORS['bg']).pack(side='left')
        tk.Radiobutton(radio_frame, text="Mandatory", variable=self.strict_var, value=True, bg=COLORS['bg'], fg='white', selectcolor=COLORS['bg']).pack(side='left', padx=15)

        # Batch Limit Control
        batch_frame = tk.Frame(parent, bg=COLORS['bg'])
        batch_frame.pack(fill='x', pady=(0, 15))
        self.create_label(batch_frame, "Max Batches (0 = unlimited)")
        self.batch_var = tk.StringVar(value="0")
        batch_entry = tk.Entry(batch_frame, textvariable=self.batch_var, 
                            width=5, bg=COLORS['card'], fg='white', 
                            insertbackground='white', font=FONTS['normal'], 
                            relief='flat')
        batch_entry.pack(side='left')
        tk.Label(batch_frame, text=" (each batch = 10 links)", 
         bg=COLORS['bg'], fg=COLORS['subtext'], font=FONTS['small']).pack(side='left', padx=(10, 0))

        # Filters
        preset_frame = tk.Frame(parent, bg=COLORS['bg'])
        preset_frame.pack(fill='x')
        self.create_label(preset_frame, "Filters").pack(side='left')
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var, values=list(data_manager.data['presets'].keys()), width=15)
        preset_combo.pack(side='right')
        preset_combo.bind('<<ComboboxSelected>>', self.load_preset_selection)
        
        self.look_text = self.create_text_area(parent, "Look For", 4)
        self.ignore_text = self.create_text_area(parent, "Ignore", 4)
        
        # Buttons
        btn_frame = tk.Frame(parent, bg=COLORS['bg'], pady=20)
        btn_frame.pack(fill='x', side='bottom')
        self.start_btn = ModernButton(btn_frame, "▶ START", self.start_research, bg=COLORS['success'])
        self.start_btn.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.stop_btn = ModernButton(btn_frame, "⏹ STOP", self.stop_research, bg=COLORS['error'])
        self.stop_btn.pack(side='right', fill='x', expand=True, padx=(5, 0))
        self.stop_btn.set_state('disabled')

    def build_logs(self, parent):
        log_frame = tk.LabelFrame(parent, text="Log", bg=COLORS['bg'], fg=COLORS['text'], font=FONTS['subheader'], padx=10, pady=10)
        log_frame.pack(fill='both', expand=True)
        self.log_widget = scrolledtext.ScrolledText(log_frame, bg=COLORS['card'], fg='white', font=FONTS['code'], borderwidth=0)
        self.log_widget.pack(fill='both', expand=True)
        for tag, color in [('INFO', '#569cd6'), ('SUCCESS', COLORS['success']), ('ERROR', COLORS['error']), ('WARNING', COLORS['warning']), ('ACTION', COLORS['accent']), ('DECISION', '#c586c0')]:
            self.log_widget.tag_config(tag, foreground=color)

    def create_label(self, parent, text):
        lbl = tk.Label(parent, text=text, bg=COLORS['bg'], fg=COLORS['subtext'], font=FONTS['subheader'])
        lbl.pack(anchor='w', pady=(0, 5))
        return lbl

    def create_text_area(self, parent, title, height):
        self.create_label(parent, title)
        txt = scrolledtext.ScrolledText(parent, height=height, bg=COLORS['card'], fg='white', insertbackground='white', font=FONTS['normal'], borderwidth=0)
        txt.pack(fill='x', pady=(0, 15))
        return txt

    def manage_list(self, key, combo_widget):
        def save_cb(new_list):
            data_manager.data[key] = new_list
            data_manager.save()
            combo_widget['values'] = new_list
            if new_list: combo_widget.current(0)
        ManagerDialog(self, f"Manage {key}", data_manager.data[key], save_cb)

    def load_presets(self):
        presets = data_manager.data.get('presets', {})
        if presets:
            first_key = list(presets.keys())[0]
            self.preset_var.set(first_key)
            self.load_preset_selection(None)

    def load_preset_selection(self, event):
        name = self.preset_var.get()
        if name in data_manager.data['presets']:
            p = data_manager.data['presets'][name]
            self.look_text.delete('1.0', 'end')
            self.look_text.insert('1.0', p.get('look', ''))
            self.ignore_text.delete('1.0', 'end')
            self.ignore_text.insert('1.0', p.get('ignore', ''))

    def log(self, type_, message):
        self.log_widget.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] ", 'TIME')
        self.log_widget.insert('end', f"{type_.upper()}: {message}\n", type_.upper())
        self.log_widget.see('end')

    def agent_callback(self, event_type, data):
        self.after(0, lambda: self._handle_agent_event(event_type, data))

    def _handle_agent_event(self, event_type, data):
        if event_type == 'log': self.log(data['type'], data['message'])
        elif event_type == 'request_selection':
            dialog = LinkSelectorDialog(self, data)
            self.wait_window(dialog)
            if dialog.result: self.agent.answers.put(dialog.result)
            else: self.agent.answers.put("STOPPED")

    def start_research(self):
        topic = self.topic_var.get()
        url = self.url_var.get()
        if not topic: return messagebox.showerror("Error", "Enter a topic")
        
        self.agent_running = True
        self.start_btn.set_state('disabled')
        self.stop_btn.set_state('normal')
        self.log_widget.delete('1.0', 'end')
        
        t = threading.Thread(target=self.run_agent_thread, args=(topic, url), daemon=True)
        t.start()

    def stop_research(self):
        if self.agent: 
            self.agent.stop_flag = True
            try: self.agent.answers.put_nowait("STOPPED")
            except: pass
        self.log("WARNING", "Stopping...")
        self.start_btn.set_state('normal')
        self.stop_btn.set_state('disabled')

    def run_agent_thread(self, topic, url):
        try:
            model = self.model_var.get() or "qwen2.5:3b"
            ignore = self.ignore_text.get('1.0', 'end').strip().split('\n')
            max_batches = int(self.batch_var.get() or "0")  # <-- ADD THIS
            
            self.agent = ModernUIAgent(model, self.agent_callback, ignore)
            countries = self.country_entry.get().split(',') if self.country_entry.get() else []
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Pass max_batches to research_ui
            results = loop.run_until_complete(
                self.agent.research_ui(topic, url, countries, self.strict_var.get(), max_batches)
            )
            # Show results window when done
            self.after(0, lambda: self.show_results_window(results or []))
        except ValueError:
            self.agent_callback('log', {'type': 'error', 'message': 'Invalid batch number. Enter 0 for unlimited or a positive integer.'})
        except Exception as e:
            self.agent_callback('log', {'type': 'error', 'message': str(e)})
        finally:
            self.after(0, lambda: self.start_btn.set_state('normal'))
            self.after(0, lambda: self.stop_btn.set_state('disabled'))



    def check_ollama(self):
        """Fetches real models from Ollama API"""
        def _check():
            try:
                response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
                if response.status_code == 200:
                    models = [m['name'] for m in response.json().get('models', [])]
                    self.after(0, lambda: self.update_ollama_status(True, models))
                else:
                    self.after(0, lambda: self.update_ollama_status(False, []))
            except:
                self.after(0, lambda: self.update_ollama_status(False, []))
        
        threading.Thread(target=_check, daemon=True).start()

    def update_ollama_status(self, connected, models):
        color = COLORS['success'] if connected else COLORS['error']
        self.status_canvas.itemconfig(self.status_light, fill=color)
        
        if connected and models:
            self.model_combo['values'] = models
            current = self.model_var.get()
            if not current or current not in models:
                self.model_combo.current(0)
            self.log("SUCCESS", f"Connected to Ollama. Found {len(models)} models.")
        else:
            self.log("ERROR", "Ollama not reachable.")
            self.model_combo['values'] = ["qwen2.5:3b", "hermes-3", "mistral"]
            self.model_combo.current(0)

if __name__ == "__main__":
    app = ResearchApp()
    app.mainloop()