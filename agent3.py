"""
INTELLIGENT AI RESEARCH AGENT - BACKEND FIXED
"""
import asyncio
import json
import re
import html
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from playwright.async_api import async_playwright
import httpx
from datetime import datetime

# --- Data Models ---
class ResearchResult(BaseModel):
    url: str
    title: str
    relevance_score: float
    summary: str
    relevance_reason: str
    key_points: List[str]

# --- Link Evaluator (Fixed) ---
class LinkEvaluator:
    """Evaluates whether links are worth clicking based on titles and context"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.cache = {}

    async def _ask_ai(self, prompt: str) -> Optional[Dict]:
        """Send request to Ollama with robust JSON extraction"""
        enhanced_prompt = prompt + "\n\nIMPORTANT: Return ONLY the JSON object, nothing else."
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": enhanced_prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 1000, "num_ctx": 4096}
                    }
                )
                if response.status_code == 200:
                    text = response.json().get('response', '')
                    # Attempt to find JSON in the response
                    try:
                        match = re.search(r'\{.*\}', text, re.DOTALL)
                        if match: 
                            return json.loads(match.group())
                    except: 
                        pass
                    # Fallback: try raw parse
                    try:
                        return json.loads(text)
                    except:
                        pass
        except Exception as e:
            print(f"AI Connection Error: {e}")
        return None

    def _extract_text(self, html_content: str, max_length: int = 3000) -> str:
        """Clean HTML to text"""
        cleaned = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', cleaned)
        text = html.unescape(text)
        return ' '.join(text.split())[:max_length]

    async def evaluate_batch(self, topic: str, links: List[Dict]) -> List[Dict]:
        """Evaluate a batch of links at once"""
        # Format links for prompt
        links_text = "\n".join([f"ID {i}: {l['text']} ({l['url']})" for i, l in enumerate(links)])
        
        prompt = f"""You are a research filter. Identify links HIGHLY RELEVANT to: "{topic}"
        
        LINKS TO EVALUATE:
        {links_text}

        INSTRUCTIONS:
        1. Select only links that explicitly discuss "{topic}".
        2. Ignore general navigation, login, or unrelated news.
        
        RESPONSE FORMAT (JSON ONLY):
        {{
            "relevant_ids": [0, 2, 5],
            "reasoning": {{ "0": "Direct mention", "2": "Implied relevance" }}
        }}
        """
        
        response = await self._ask_ai(prompt)
        results = []
        rel_ids = response.get("relevant_ids", []) if response else []
        reasons = response.get("reasoning", {}) if response else {}

        for i, link in enumerate(links):
            is_rel = i in rel_ids
            results.append({
                "should_click": is_rel,
                "confidence": 0.9 if is_rel else 0.1,
                "reason": reasons.get(str(i), "Not selected by batch filter")
            })
        return results

    # --- FIX: This method is now correctly indented inside the class ---
    async def extract_complete_page_content(self, page, url: str) -> Tuple[str, str, str]:
        """Extract complete page content with scrolling"""
        try:
            await page.wait_for_load_state('domcontentloaded', timeout=15000)
            
            # Simple Scroll to trigger lazy loading
            await page.evaluate("""
                async () => {
                    for (let i = 0; i < 5; i++) {
                        window.scrollBy(0, 500);
                        await new Promise(r => setTimeout(r, 100));
                    }
                }
            """)
            
            content = await page.content()
            title = await page.title()
            text = self._extract_text(content, 4000)
            return title, text, content
        except Exception as e:
            print(f"Error reading page {url}: {e}")
            return "Error", "", ""

    async def evaluate_after_click(self, topic: str, url: str, title: str, page_text: str) -> Dict:
        """Evaluate page content after clicking"""
        if len(page_text) < 100: 
            return {"is_relevant": False, "reason": "Content too short"}
        
        prompt = f"""Analyze this page for research topic: "{topic}"
        Title: {title}
        Content (first 3000 chars): {page_text[:3000]}
        
        Return JSON ONLY: {{
            "is_relevant": true/false,
            "relevance_score": 0.0-1.0,
            "summary": "2 sentence summary",
            "reason": "Why it is relevant",
            "key_points": ["point 1", "point 2"]
        }}"""
        
        res = await self._ask_ai(prompt)
        
        # Default fallback if AI fails
        if not res: 
            return {
                "is_relevant": False, 
                "relevance_score": 0.0, 
                "summary": "AI Analysis Failed", 
                "reason": "Could not parse response",
                "key_points": []
            }
        return res

# --- Agent Wrapper for UI ---
class SmartResearchAgent:
    def __init__(self, model_name, ui_callback=None):
        self.model_name = model_name
        self.evaluator = LinkEvaluator(model_name)
        self.ui_callback = ui_callback
        self.answers = asyncio.Queue() # Using asyncio queue for async compatibility
        self.stop_flag = False

    async def _extract_links_smart(self, page, topic, start_url):
        # Using a reliable broad extraction for the UI
        links = await page.evaluate("""() => {
            const ls = [];
            document.querySelectorAll('a').forEach(a => {
                if (a.href.startsWith('http') && a.innerText.trim().length > 5) {
                    ls.push({text: a.innerText.trim(), url: a.href});
                }
            });
            return ls;
        }""")
        
        # Deduplicate
        seen = set()
        unique = []
        for l in links:
            if l['url'] not in seen:
                seen.add(l['url'])
                unique.append(l)
        return unique

    async def research_ui(self, topic, start_url, batch_size=10, max_batches=0):
        """Main entry point for UI-based research"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            final_results = []
            
            try:
                if self.ui_callback: self.ui_callback('log', {'type': 'action', 'message': f'🌐 Loading {start_url} ...'})
                await page.goto(start_url, wait_until='domcontentloaded')
                
                # 1. Extract
                links = await self._extract_links_smart(page, topic, start_url)
                if not links: return []
                
                # 2. User Selection
                if self.ui_callback: 
                    self.ui_callback('request_selection', links)
                    # Wait for UI response (handled via queue in UI thread)
                    # Note: In the UI code, we handle the queue crossing. 
                    # Here we assume the calling loop handles the pause/resume 
                    # OR we use a shared variable. 
                    # For simplicity with the provided UI pattern, we return 'WAITING' 
                    # but since we are inside a thread loop, we need the selection passed back.
                    pass 

                # *CRITICAL ADAPTATION*: The UI `agent_ui.py` expects this method 
                # to pause. Since `SmartResearchAgent` in the UI file was designed 
                # to read from `self.answers` (a synchronized queue), we do that here.
                
                # We need to access the standard threading queue from the UI instance
                # passed during initialization or attached later.
                # Assuming the UI attaches `self.answers` (threading.Queue) to this instance.
                import queue
                try:
                    selected = self.answers.get(timeout=300) # Wait 5 mins max
                except queue.Empty:
                    return []

                if not selected or selected == "STOPPED": return []

                # 3. Batch Processing
                total_links = len(selected)
                # Calculate total batches
                total_batches_available = (total_links + batch_size - 1) // batch_size
                
                # Apply Max Batches Limit
                batches_to_process = total_batches_available
                if max_batches > 0:
                    batches_to_process = min(total_batches_available, max_batches)
                    if self.ui_callback:
                        self.ui_callback('log', {'type': 'info', 'message': f'⚠️ Limit applied: Processing {batches_to_process} batches (of {total_batches_available} available)'})

                for i in range(batches_to_process):
                    if self.stop_flag: break
                    
                    start_idx = i * batch_size
                    chunk = selected[start_idx : start_idx + batch_size]
                    
                    if self.ui_callback:
                        self.ui_callback('log', {'type': 'info', 'message': f'📦 Batch {i+1}/{batches_to_process}: Analyzing {len(chunk)} links...'})
                    
                    # AI Filter
                    results = await self.evaluator.evaluate_batch(topic, chunk)
                    
                    # Visit Relevant
                    for link, res in zip(chunk, results):
                        if self.stop_flag: break
                        
                        if res['should_click']:
                            if self.ui_callback:
                                self.ui_callback('log', {'type': 'action', 'message': f"Visiting: {link['text'][:40]}..."})
                            
                            try:
                                new_page = await context.new_page()
                                await new_page.goto(link['url'], wait_until='domcontentloaded', timeout=20000)
                                
                                # Call the FIXED method
                                title, text, _ = await self.evaluator.extract_complete_page_content(new_page, link['url'])
                                
                                eval_res = await self.evaluator.evaluate_after_click(topic, link['url'], title, text)
                                
                                if eval_res['is_relevant']:
                                    result_obj = {
                                        'title': title,
                                        'url': link['url'],
                                        'relevance_score': eval_res.get('relevance_score', 0),
                                        'summary': eval_res.get('summary', ''),
                                        'reason': eval_res.get('reason', '')
                                    }
                                    final_results.append(result_obj)
                                    if self.ui_callback:
                                        self.ui_callback('log', {'type': 'success', 'message': f"✅ RELEVANT: {title[:30]}..."})
                                else:
                                    if self.ui_callback:
                                        self.ui_callback('log', {'type': 'info', 'message': f"❌ Not relevant: {eval_res.get('reason')}"})
                                
                                await new_page.close()
                            except Exception as e:
                                if self.ui_callback:
                                    self.ui_callback('log', {'type': 'error', 'message': f"Error visiting {link['url']}: {e}"})

                if self.ui_callback:
                    self.ui_callback('log', {'type': 'success', 'message': f"🎉 Done. Found {len(final_results)} results."})
                
                return final_results

            finally:
                await browser.close()