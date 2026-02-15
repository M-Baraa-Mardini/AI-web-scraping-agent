"""
AI Research Agent - Windows Desktop Application
MODERN UI VERSION 3.3 (Countries Manager + Progress Bar)
"""
import sys
import os
import json
import threading
import queue
import asyncio
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog, filedialog
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

import pycountry
from typing import List

def normalize_countries(raw_names: List[str]) -> List[str]:
    """
    Convert a list of country names (any language) to standard English names.
    Uses pycountry for lookup; falls back to a custom mapping.
    """
    FALLBACK_MAP = {
        "italia": "Italy",
        "deutschland": "Germany",
        "españa": "Spain",
        "francia": "France",
        "france": "France",
        "uk": "United Kingdom",
        "usa": "United States",
        "america": "United States",
        "brasil": "Brazil",
        "canadá": "Canada",
        "méxico": "Mexico",
        "nederland": "Netherlands",
        "schweiz": "Switzerland",
        "suisse": "Switzerland",
        "svizzera": "Switzerland",
        "österreich": "Austria",
        "polska": "Poland",
        "türkiye": "Turkey",
        "中国": "China",
        "日本": "Japan",
        "한국": "South Korea",
        "Россия": "Russia",
    }

    normalized = []
    for name in raw_names:
        if not name or not isinstance(name, str):
            continue
        cleaned = name.strip().lower()
        # Try pycountry first
        try:
            country = pycountry.countries.lookup(cleaned)
            normalized.append(country.name)
            continue
        except (LookupError, AttributeError):
            pass
        # Fallback to custom mapping
        if cleaned in FALLBACK_MAP:
            normalized.append(FALLBACK_MAP[cleaned])
        else:
            normalized.append(name.strip())  # keep original
    return normalized


# --- INTELLIGENT REGION MAPPING ---
REGION_HIERARCHY = {
    "europe": [
        "italy", "france", "germany", "spain", "uk", "united kingdom", "netherlands", 
        "belgium", "switzerland", "sweden", "norway", "denmark", "finland", "poland", 
        "austria", "greece", "portugal", "ireland", "czech republic", "hungary", "romania"
    ],
    "eu": [
        "italy", "france", "germany", "spain", "netherlands", "belgium", "sweden", 
        "poland", "austria", "greece", "portugal", "ireland"
    ],
    "north america": ["usa", "united states", "canada", "mexico"],
    "asia": ["china", "japan", "india", "south korea", "vietnam", "thailand", "indonesia"],
    "uk": ["united kingdom", "england", "scotland", "wales", "london"]
}

def are_locations_related(user_loc, found_loc):
    """
    Returns True if locations are related hierarchically (bidirectional).
    Example: 'Italy' matches 'Europe' AND 'Europe' matches 'Italy'.
    """
    u = user_loc.lower().strip()
    f = found_loc.lower().strip()
    
    # 1. Direct Match
    if u == f or u in f or f in u:
        return True
        
    # 2. Hierarchy Check (Is one inside the other?)
    # Check if User Input is inside the Found Region (Search: Italy, Found: Europe)
    if f in REGION_HIERARCHY and u in REGION_HIERARCHY[f]:
        return True
        
    # Check if Found Location is inside User Region (Search: Europe, Found: Italy)
    if u in REGION_HIERARCHY and f in REGION_HIERARCHY[u]:
        return True
        
    return False


# --- DATA MANAGER (now includes countries) ---
class DataManager:
    """Manages saved topics, URLs, countries, and settings"""
    def __init__(self):
        self.filename = "agent_data.json"
        self.data = self.load()

    def load(self):
        default_data = {
            "topics": ["Artificial Intelligence", "Python Automation", "Climate Change"],
            "urls": ["https://news.ycombinator.com", "https://arxiv.org", "https://reddit.com/r/technology"],
            "countries": ["USA", "UK", "Germany", "Canada", "Australia"],
            "presets": {
                "General Research": {"look": "Articles, papers, tutorials", "ignore": "Ads, login pages"},
                "News Scan": {"look": "Breaking news, headlines", "ignore": "Opinion, archive, navigation"}
            }
        }

        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    loaded = json.load(f)
                # Merge loaded data with defaults to ensure all keys exist
                for key in default_data:
                    if key not in loaded:
                        loaded[key] = default_data[key]
                return loaded
            except Exception:
                # If file is corrupt, fall back to defaults
                return default_data
        return default_data

    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving: {e}")
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
        if width:
            self.configure(width=width)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)

    def on_enter(self, e):
        if self.cget('state') != 'disabled':
            self.configure(bg=self.hover_bg)

    def on_leave(self, e):
        if self.cget('state') != 'disabled':
            self.configure(bg=self.default_bg)

    def on_click(self, e):
        if self.cget('state') != 'disabled':
            self.command()

    def set_state(self, state):
        self.configure(state=state)
        if state == 'disabled':
            self.configure(bg='#444444', fg='#888888', cursor='arrow')
        else:
            self.configure(bg=self.default_bg, fg='white', cursor='hand2')

# --- DIALOGS (Manager now supports comma‑separated entry) ---
class ManagerDialog(tk.Toplevel):
    """Dialog to Add/Edit/Delete List Items (supports comma‑separated input)"""
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
        """Add item(s) – split by commas if present."""
        val = simpledialog.askstring("Add", "Enter item(s) separated by commas:", parent=self)
        if val and val.strip():
            parts = [part.strip() for part in val.split(',') if part.strip()]
            self.items.extend(parts)
            self.refresh()

    def delete_items(self):
        sel = self.listbox.curselection()
        for i in reversed(sel):
            del self.items[i]
        self.refresh()

    def refresh(self):
        self.listbox.delete(0, 'end')
        for item in self.items:
            self.listbox.insert('end', item)

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

# --- AGENT LOGIC (with progress updates) ---
class ModernUIAgent(SmartResearchAgent):
    def __init__(self, model_name, ui_callback, ignore_terms=None):
        super().__init__(model_name)
        self.ui_callback = ui_callback
        self.answers = queue.Queue()
        self.stop_flag = False
        self.ignore_terms = ignore_terms or []

    async def _extract_links_smart(self, page, topic: str, start_url: str) -> List[Dict]:
        """PRECISION LINK EXTRACTOR (Cluster & Heuristic Based)"""
        if self.ui_callback:
            self.ui_callback('log', {'type': 'action', 'message': '🔍 Scanning for content links...'})
            self.ui_callback('progress', {'stage': 'Extracting links...', 'fraction': 0.0})

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
            """Main Loop with Intelligent Matching & Filtered Results."""
            if self.ui_callback:
                self.ui_callback('log', {'type': 'info', 'message': f'🚀 Starting research: {topic}'})
                self.ui_callback('progress', {'stage': 'Starting...', 'fraction': 0.0})

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
                    self.ui_callback('progress', {'stage': 'Waiting for link selection...', 'fraction': 0.1})
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
                    total_selected = len(selected_links)
                    total_batches = (total_selected + chunk_size - 1) // chunk_size
                    batches_to_process = min(total_batches, max_batches) if max_batches > 0 else total_batches

                    if max_batches > 0 and self.ui_callback:
                        self.ui_callback('log', {'type': 'info',
                            'message': f'⚠️ Processing {batches_to_process}/{total_batches} batches (limit: {max_batches})'})

                    links_evaluated = 0
                    for i in range(batches_to_process):
                        if self.stop_flag: break
                        chunk = selected_links[i*chunk_size:(i+1)*chunk_size]
                        results = await self.evaluator.evaluate_batch(topic, chunk)
                        links_evaluated += len(chunk)

                        # Progress update
                        fraction = 0.1 + 0.4 * (links_evaluated / total_selected)
                        self.ui_callback('progress', {'stage': f'Evaluating: {links_evaluated}/{total_selected}', 'fraction': fraction})

                        for link, res in zip(chunk, results):
                            if res['should_click']:
                                promising_links.append({**link, **res})
                                self.ui_callback('log', {'type': 'decision', 'message': f"✓ FOUND: {link['text'][:40]}..."})

                    # 4. Visit
                    if not promising_links:
                        self.ui_callback('log', {'type': 'warning', 'message': 'No relevant links found.'})
                        self.ui_callback('progress', {'stage': 'No relevant links', 'fraction': 1.0})
                        return []

                    self.ui_callback('progress', {'stage': f'Visiting {len(promising_links)} pages...', 'fraction': 0.6})

                    final_results = []
                    visited = 0
                    for link_info in promising_links:
                        if self.stop_flag: break
                        visited += 1
                        fraction = 0.6 + 0.3 * (visited / len(promising_links))
                        self.ui_callback('progress', {'stage': f'Visiting: {visited}/{len(promising_links)}', 'fraction': fraction})

                        url = link_info['url']
                        self.ui_callback('log', {'type': 'action', 'message': f"🔗 Visiting: {link_info['text'][:50]}..."})

                        new_page = None
                        evaluation = None
                        try:
                            new_page = await context.new_page()
                            await new_page.goto(url, wait_until='domcontentloaded')
                            await asyncio.sleep(2)
                            title, content, _ = await self.evaluator.extract_complete_page_content(new_page, url)
                            evaluation = await self.evaluator.evaluate_after_click(topic, url, title, content)
                        except Exception as e:
                            self.ui_callback('log', {'type': 'error', 'message': f"Error visiting {url}: {e}"})
                        finally:
                            if new_page: await new_page.close()

                        # --- INTELLIGENT MATCHING & FILTERING ---
                        if evaluation and evaluation.get('is_relevant', False):
                            
                            # A. Prepare Lists
                            # Normalize found items
                            found_countries_raw = evaluation.get('countries', [])
                            found_countries_norm = normalize_countries(found_countries_raw) # e.g. "Italy"
                            found_topics = evaluation.get('topics', [])
                            
                            # Normalize user items
                            user_countries_norm = normalize_countries(countries) # e.g. "Italy"
                            user_countries_clean = [c.strip() for c in user_countries_norm if c.strip()]
                            
                            user_topics_clean = [t.strip() for t in topic.split(',') if t.strip()]

                            # B. Identify MATCHED Items (for display)
                            matched_countries = []
                            matched_topics = []

                            # 1. Match Countries (Bi-directional Hierarchy)
                            country_match = False
                            if not user_countries_clean:
                                country_match = True
                                matched_countries = found_countries_norm # If user asked for nothing, show all found
                            else:
                                for uc in user_countries_clean:
                                    for fc in found_countries_norm:
                                        if are_locations_related(uc, fc):
                                            country_match = True
                                            if fc not in matched_countries: matched_countries.append(fc)
                                
                                # If we matched based on hierarchy (e.g. User: Italy, Found: Europe), 
                                # we should probably show 'Europe' in the results so the user knows why.
                                # The loop above adds 'fc' (found country) to the list.

                            # 2. Match Topics (Substring/Fuzzy)
                            topic_match = False
                            if not user_topics_clean:
                                topic_match = True
                                matched_topics = found_topics
                            else:
                                for ut in user_topics_clean:
                                    for ft in found_topics:
                                        if ut.lower() in ft.lower() or ft.lower() in ut.lower():
                                            topic_match = True
                                            if ft not in matched_topics: matched_topics.append(ft)

                            # C. Strict Validation
                            is_valid = True
                            rejection_reason = ""

                            if strict_countries:
                                if not topic_match:
                                    is_valid = False
                                    rejection_reason = f"Strict: Topic mismatch (Expected {user_topics_clean}, Found {found_topics})"
                                elif not country_match:
                                    is_valid = False
                                    rejection_reason = f"Strict: Country mismatch (Expected {user_countries_clean}, Found {found_countries_norm})"

                            # D. Save Result (Saving ONLY matched items if filtered, or all if no filter)
                            if is_valid:
                                score = evaluation.get('relevance_score', 0)
                                self.ui_callback('log', {'type': 'success', 'message': f"✅ MATCHED! Score: {score}"})
                                
                                # Use matched lists for display if specific filters were active
                                display_topics = matched_topics if user_topics_clean else found_topics
                                display_countries = matched_countries if user_countries_clean else found_countries_norm

                                final_results.append({
                                    'title': title,
                                    'url': url,
                                    'relevance_score': score,
                                    'summary': evaluation.get('summary', ''),
                                    'reason': evaluation.get('reason', ''),
                                    'topics': display_topics,      # <--- Now clean
                                    'countries': display_countries # <--- Now clean
                                })
                            else:
                                self.ui_callback('log', {'type': 'info', 'message': f"❌ {rejection_reason}"})

                        else:
                            if evaluation:
                                self.ui_callback('log', {'type': 'info', 'message': f"❌ Not relevant: {evaluation.get('reason', 'No reason')}"})

                    self.ui_callback('log', {'type': 'success', 'message': f'🎉 Done. {len(final_results)} results.'})
                    self.ui_callback('progress', {'stage': 'Research completed', 'fraction': 1.0})
                    return final_results

                except Exception as e:
                    traceback.print_exc()
                    self.ui_callback('log', {'type': 'error', 'message': str(e)})
                finally:
                    await browser.close()
# --- IMPROVED RESULTS DIALOG ---
class ResultsDialog(tk.Toplevel):
    """Dialog to view, select, and export research results (with topics & countries)."""
    def __init__(self, parent, results):
        super().__init__(parent)
        self.title(f"Research Results ({len(results)} found)")
        self.geometry("1400x700")
        self.configure(bg=COLORS['bg'])
        self.results = results
        self.transient(parent)
        self.grab_set()

        self.item_to_idx = {}

        # Main vertical paned window
        paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        paned.pack(fill='both', expand=True, padx=10, pady=10)

        # --- Top frame: result list + control buttons ---
        top_frame = tk.Frame(paned, bg=COLORS['bg'])
        paned.add(top_frame, weight=3)

        # Header with counter and action buttons
        header = tk.Frame(top_frame, bg=COLORS['bg'])
        header.pack(fill='x', pady=(0, 10))

        self.counter_var = tk.StringVar(value=f"0 selected / {len(results)} total")
        counter_label = tk.Label(header, textvariable=self.counter_var,
                                 bg=COLORS['bg'], fg=COLORS['accent'],
                                 font=('Segoe UI', 12, 'bold'))
        counter_label.pack(side='left')

        btn_frame = tk.Frame(header, bg=COLORS['bg'])
        btn_frame.pack(side='right')

        ModernButton(btn_frame, "Select All", self.select_all,
                     bg=COLORS['card']).pack(side='left', padx=2)
        ModernButton(btn_frame, "Clear Selection", self.clear_selection,
                     bg=COLORS['warning']).pack(side='left', padx=2)
        ModernButton(btn_frame, "Export Selected", self.export_results,
                     bg=COLORS['success']).pack(side='left', padx=2)

        # Frame for treeview + scrollbars
        tree_frame = tk.Frame(top_frame, bg=COLORS['bg'])
        tree_frame.pack(fill='both', expand=True)

        # Treeview for results
        cols = ("Score", "Topics", "Countries", "Title", "URL")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show='headings',
                                 selectmode='extended')
        self.tree.heading("Score", text="Relevance")
        self.tree.heading("Topics", text="Topics")
        self.tree.heading("Countries", text="Countries")
        self.tree.heading("Title", text="Title")
        self.tree.heading("URL", text="URL")
        self.tree.column("Score", width=70, anchor='center')
        self.tree.column("Topics", width=200)
        self.tree.column("Countries", width=150)
        self.tree.column("Title", width=400)
        self.tree.column("URL", width=300)

        # Vertical scrollbar
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')

        # Horizontal scrollbar
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=hsb.set)
        hsb.pack(side='bottom', fill='x')

        self.tree.pack(side='left', fill='both', expand=True)

        # Populate tree
        for idx, res in enumerate(results):
            score = f"{res.get('relevance_score', 0):.2f}"
            topics = ', '.join(res.get('topics', []))[:200]
            countries = ', '.join(res.get('countries', []))[:150]
            title = res.get('title', 'Untitled')[:100]
            url = res.get('url', '')[:80]
            item_id = self.tree.insert("", "end", values=(score, topics, countries, title, url))
            self.item_to_idx[item_id] = idx

        # Bind events
        self.tree.bind('<<TreeviewSelect>>', self.on_select)
        self.tree.bind('<Double-1>', self.open_selected_url)

        # --- Bottom frame: details panel ---
        bottom_frame = tk.Frame(paned, bg=COLORS['bg'])
        paned.add(bottom_frame, weight=1)

        tk.Label(bottom_frame, text="Details", bg=COLORS['bg'],
                 fg=COLORS['subtext'], font=FONTS['subheader']).pack(anchor='w')

        self.detail_text = scrolledtext.ScrolledText(
            bottom_frame, height=8,
            bg=COLORS['card'], fg='white',
            font=FONTS['normal'], wrap='word'
        )
        self.detail_text.pack(fill='both', expand=True, pady=(5, 0))

    def on_select(self, event=None):
        """Update counter and details when selection changes."""
        selected_items = self.tree.selection()
        count = len(selected_items)
        total = len(self.results)
        self.counter_var.set(f"{count} selected / {total} total")

        # Show details of the last selected item
        if selected_items:
            last_idx = self.item_to_idx[selected_items[-1]]
            res = self.results[last_idx]
            
            details = f"Title: {res.get('title')}\n"
            details += f"URL: {res.get('url')}\n"
            details += f"Score: {res.get('relevance_score')}\n"
            details += f"Topics: {', '.join(res.get('topics', []))}\n"
            details += f"Countries: {', '.join(res.get('countries', []))}\n"
            details += "-" * 40 + "\n"
            details += f"Summary:\n{res.get('summary', 'N/A')}\n\n"
            details += f"Reasoning:\n{res.get('reason', 'N/A')}"
            
            self.detail_text.delete('1.0', 'end')
            self.detail_text.insert('1.0', details)

    def select_all(self):
        """Select all items in the tree."""
        self.tree.selection_set(self.tree.get_children())
        self.on_select()

    def clear_selection(self):
        """Deselect all items."""
        self.tree.selection_remove(self.tree.selection())
        self.on_select()

    def open_selected_url(self, event=None):
        """Open the URL of the double-clicked item."""
        import webbrowser
        selected_items = self.tree.selection()
        if selected_items:
            idx = self.item_to_idx[selected_items[0]]
            url = self.results[idx].get('url')
            if url:
                webbrowser.open(url)

    def export_results(self):
        """Export selected items to JSON."""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select at least one result to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not file_path:
            return

        export_data = []
        for item in selected_items:
            idx = self.item_to_idx[item]
            export_data.append(self.results[idx])

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", f"Exported {len(export_data)} results.")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))


# --- MAIN APP (with countries manager and progress bar) ---
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


    def on_select(self, event=None):
        """Update counter when selection changes."""
        selected = len(self.tree.selection())
        total = len(self.results)
        self.counter_var.set(f"{selected} selected / {total} total")

    def select_all(self):
        """Select all rows in the tree."""
        self.tree.selection_set(self.tree.get_children())
        self.on_select()

    def clear_selection(self):
        """Clear all selections."""
        self.tree.selection_remove(self.tree.selection())
        self.on_select()

    def open_selected_url(self, event):
        """Open the URL of the selected row in a web browser."""
        item = self.tree.selection()
        if item:
            idx = self.item_to_idx[item[0]]
            url = self.results[idx].get('url', '')
            if url:
                import webbrowser
                webbrowser.open(url)

    def export_results(self):
        """Export selected results to a JSON file."""
        selected = self.tree.selection()
        if not selected:
            tk.messagebox.showwarning("No Selection", "Please select at least one result to export.")
            return
        file_path = tk.filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not file_path:
            return
        selected_results = [self.results[self.item_to_idx[item]] for item in selected]
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(selected_results, f, indent=2, ensure_ascii=False)
            tk.messagebox.showinfo("Export Successful", f"Exported {len(selected_results)} results.")
        except Exception as e:
            tk.messagebox.showerror("Export Error", str(e))
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
        style.configure("Horizontal.TProgressbar", background=COLORS['accent'], troughcolor=COLORS['card'])

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
        if self.topic_combo['values']:
            self.topic_combo.current(0)
        self.topic_combo.pack(side='left', fill='x', expand=True)
        ModernButton(topic_frame, "⚙", lambda: self.manage_list("topics", self.topic_combo), bg=COLORS['card'], width=3).pack(side='right', padx=(5,0))

        # URL
        self.create_label(parent, "Start URL")
        url_frame = tk.Frame(parent, bg=COLORS['bg'])
        url_frame.pack(fill='x', pady=(0, 15))
        self.url_var = tk.StringVar()
        self.url_combo = ttk.Combobox(url_frame, textvariable=self.url_var, font=FONTS['normal'])
        self.url_combo['values'] = data_manager.data['urls']
        if self.url_combo['values']:
            self.url_combo.current(0)
        self.url_combo.pack(side='left', fill='x', expand=True)
        ModernButton(url_frame, "⚙", lambda: self.manage_list("urls", self.url_combo), bg=COLORS['card'], width=3).pack(side='right', padx=(5,0))

        # Countries (new row, exactly like topics/urls)
        self.create_label(parent, "Countries")
        country_frame = tk.Frame(parent, bg=COLORS['bg'])
        country_frame.pack(fill='x', pady=(0, 15))
        self.country_var = tk.StringVar()
        self.country_combo = ttk.Combobox(country_frame, textvariable=self.country_var, font=FONTS['normal'])
        self.country_combo['values'] = data_manager.data['countries']
        if self.country_combo['values']:
            self.country_combo.current(0)
        self.country_combo.pack(side='left', fill='x', expand=True)
        ModernButton(country_frame, "⚙", lambda: self.manage_list("countries", self.country_combo), bg=COLORS['card'], width=3).pack(side='right', padx=(5,0))

        # Strict Country Filtering
        self.strict_var = tk.BooleanVar(value=False)
        strict_check = tk.Checkbutton(parent, text="Strict Country Filtering",
                                     variable=self.strict_var, bg=COLORS['bg'],
                                     fg='white', selectcolor=COLORS['bg'],
                                     font=FONTS['normal'])
        strict_check.pack(anchor='w', pady=(0, 15))

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
        # Main log area
        log_frame = tk.LabelFrame(parent, text="Log", bg=COLORS['bg'], fg=COLORS['text'], font=FONTS['subheader'], padx=10, pady=10)
        log_frame.pack(fill='both', expand=True)

        self.log_widget = scrolledtext.ScrolledText(log_frame, bg=COLORS['card'], fg='white', font=FONTS['code'], borderwidth=0)
        self.log_widget.pack(fill='both', expand=True)
        for tag, color in [('INFO', '#569cd6'), ('SUCCESS', COLORS['success']), ('ERROR', COLORS['error']), ('WARNING', COLORS['warning']), ('ACTION', COLORS['accent']), ('DECISION', '#c586c0')]:
            self.log_widget.tag_config(tag, foreground=color)

        # Progress bar and status at the bottom
        progress_frame = tk.Frame(parent, bg=COLORS['bg'], height=40)
        progress_frame.pack(fill='x', pady=(5, 0))

        self.status_label = tk.Label(progress_frame, text="Ready", bg=COLORS['bg'], fg=COLORS['subtext'], font=FONTS['normal'])
        self.status_label.pack(side='left', padx=5)

        self.progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate', style='Horizontal.TProgressbar')
        self.progress_bar.pack(side='right', fill='x', expand=True, padx=5)

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
            if new_list:
                combo_widget.current(0)
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
        if event_type == 'log':
            self.log(data['type'], data['message'])
        elif event_type == 'request_selection':
            dialog = LinkSelectorDialog(self, data)
            self.wait_window(dialog)
            if dialog.result:
                self.agent.answers.put(dialog.result)
            else:
                self.agent.answers.put("STOPPED")
        elif event_type == 'progress':
            # data contains 'stage' and 'fraction' (0-1)
            self.status_label.config(text=data['stage'])
            self.progress_bar['value'] = data['fraction'] * 100

    def start_research(self):
        topic = self.topic_var.get()
        url = self.url_var.get()
        if not topic:
            return messagebox.showerror("Error", "Enter a topic")

        self.agent_running = True
        self.start_btn.set_state('disabled')
        self.stop_btn.set_state('normal')
        self.log_widget.delete('1.0', 'end')
        self.progress_bar['value'] = 0
        self.status_label.config(text="Starting...")

        t = threading.Thread(target=self.run_agent_thread, args=(topic, url), daemon=True)
        t.start()

    def stop_research(self):
        if self.agent:
            self.agent.stop_flag = True
            try:
                self.agent.answers.put_nowait("STOPPED")
            except:
                pass
        self.log("WARNING", "Stopping...")
        self.start_btn.set_state('normal')
        self.stop_btn.set_state('disabled')
        self.status_label.config(text="Stopped")

    def run_agent_thread(self, topic, url):
        try:
            model = self.model_var.get() or "qwen2.5:3b"
            ignore = self.ignore_text.get('1.0', 'end').strip().split('\n')
            max_batches = int(self.batch_var.get() or "0")

            self.agent = ModernUIAgent(model, self.agent_callback, ignore)

            # Get selected country (if any) – for now just pass as list with one item if not empty
            country = self.country_var.get()
            countries = [c.strip() for c in country.split(',') if c.strip()] if country else []
            #countries = [country] if country else []

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.agent.research_ui(topic, url, countries, self.strict_var.get(), max_batches)
            )
            self.after(0, lambda: self.show_results_window(results or []))
        except ValueError:
            self.agent_callback('log', {'type': 'error', 'message': 'Invalid batch number. Enter 0 for unlimited or a positive integer.'})
        except Exception as e:
            self.agent_callback('log', {'type': 'error', 'message': str(e)})
        finally:
            self.after(0, lambda: self.start_btn.set_state('normal'))
            self.after(0, lambda: self.stop_btn.set_state('disabled'))
            if not self.agent.stop_flag:
                self.after(0, lambda: self.status_label.config(text="Ready"))

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