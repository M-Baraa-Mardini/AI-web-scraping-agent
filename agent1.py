"""
INTELLIGENT AI RESEARCH AGENT - HERMES-3 FIXED
Scans links, intelligently clicks only on relevant ones, and provides focused results
"""
import asyncio
import json
import re
import sys
import time
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
from playwright.async_api import async_playwright
import httpx
from datetime import datetime
import html


the_model = "qwen2.5:3b"
# --- Data Models ---
class ResearchResult(BaseModel):
    url: str
    title: str
    relevance_score: float
    summary: str
    relevance_reason: str
    key_points: List[str]

# --- Thinking Logger ---
class ThinkingLogger:
    """Logs the agent's thinking process"""
    
    def __init__(self):
        self.logs = []
        
    def log(self, message_type, message, details=""):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "type": message_type,
            "message": message,
            "details": details
        }
        
        self.logs.append(entry)
        
        # Format for console output with colors
        colors = {
            "success": "\033[92m",  # Green
            "error": "\033[91m",    # Red
            "warning": "\033[93m",  # Yellow
            "info": "\033[94m",     # Blue
            "thinking": "\033[95m", # Purple
            "action": "\033[96m",   # Cyan
            "decision": "\033[93m", # Yellow
        }
        
        color = colors.get(message_type, "\033[97m")
        reset = "\033[0m"
        
        print(f"{color}[{timestamp}] {message_type.upper()}: {message}{reset}")
        if details:
            print(f"   Details: {details}")
        
        self.save_to_file()
    
    def save_to_file(self, filename="thinking_log.txt"):
        """Save thinking log to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in self.logs:
                f.write(f"[{entry['timestamp']}] {entry['type'].upper()}\n")
                f.write(f"  Message: {entry['message']}\n")
                if entry['details']:
                    f.write(f"  Details: {entry['details']}\n")
                f.write("-" * 60 + "\n")
    
    def get_recent_logs(self, count=10):
        """Get recent log entries"""
        return self.logs[-count:]

# Initialize thinking logger
logger = ThinkingLogger()

# --- Ollama Connection ---
async def check_ollama_connection() -> Tuple[bool, str]:
    """Check if Ollama is running and find a model"""
    logger.log("info", "Checking Ollama connection...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get('name', 'unknown') for m in models]
                
                logger.log("success", f"✅ Ollama is running! Available models: {model_names}")
                
                # Try to find Hermes-3 first
                found_model = None
                
                # Check for Hermes models
                hermes_models = [name for name in model_names if any(
                    alias in name.lower() for alias in the_model
                )]
                
                if hermes_models:
                    found_model = hermes_models[0]
                    if len(hermes_models) > 1:
                        logger.log("info", f"Multiple Hermes models found, using: {found_model}")
                else:
                    # Try llama3.1 as fallback
                    llama_models = [name for name in model_names if "llama" in name.lower()]
                    if llama_models:
                        found_model = llama_models[0]
                    elif model_names:
                        found_model = model_names[0]
                
                if found_model:
                    logger.log("success", f"✅ Using model: {found_model}")
                    return True, found_model
                else:
                    logger.log("error", "❌ No models found!")
                    return False, ""
            else:
                logger.log("error", f"Ollama returned status code: {response.status_code}")
                return False, ""
    except Exception as e:
        logger.log("error", f"Cannot connect to Ollama: {e}")
        print("\n⚠️ Please make sure Ollama is running:")
        print("   1. Open a new PowerShell window")
        print("   2. Run: ollama serve")
        print("   3. In another window, run: ollama pull hermes-3:latest")
        return False, ""

# --- Cookie Handler ---
async def handle_popups(page):
    """Quick popup handler"""
    try:
        # Quick consent buttons
        consent_buttons = ['button:has-text("Accept")', 'button:has-text("Agree")', 
                          'button:has-text("OK")', 'button[aria-label*="accept"]']
        for selector in consent_buttons:
            try:
                if await page.locator(selector).count() > 0:
                    await page.click(selector)
                    logger.log("action", "Clicked consent button")
                    await asyncio.sleep(0.5)
                    break
            except:
                continue
    except:
        pass

# --- Intelligent Link Evaluator ---
class LinkEvaluator:
    """Evaluates whether links are worth clicking based on titles and context"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.cache = {}  # Cache decisions to avoid re-evaluating
        
    async def evaluate_link_from_title(self, topic: str, link_text: str, url: str, countries: List[str] = None) -> Dict:
        """Evaluate if a link is worth clicking based on title AND countries"""
        cache_key = f"{topic}_{link_text}_{url}_{'_'.join(countries) if countries else ''}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        logger.log("decision", f"Evaluating link: {link_text[:50]}...")
        
        # Build country filter string
        country_filter = ""
        if countries:
            country_filter = f" AND must be specifically about {', '.join(countries)}"
        
        # Enhanced prompt with specific relevance guidance
        prompt = f"""You are an AI research assistant. Evaluate if this link is relevant to the research topic: "{topic}"{country_filter}

        LINK TEXT: {link_text}
        URL: {url}
        TOPIC: "{topic}"
        {"COUNTRIES: " + ", ".join(countries) if countries else "NO COUNTRY FILTER"}

        Instructions:
        1. The link is relevant if it discusses, mentions, or relates to "{topic}" in any meaningful way.
        2. Be inclusive: if the link text contains any of these related terms, consider it relevant:
        - For AI: GPT, machine learning, deep learning, neural networks, LLM, language model, AI, artificial intelligence, chatbot, etc.
        3. {"The link MUST be specifically about " + " AND ".join(countries) + " (not just mentions them in passing)" if countries else ""}
        4. Provide a confidence score (0.0-1.0) based on how likely it is that the link contains useful information.

        Response format (JSON only):
        {{
            "should_click": true/false,
            "confidence": 0.0-1.0,
            "reason": "Brief explanation",
            "needs_inspection": true/false (if title is ambiguous)
        }}"""
        
        decision = await self._ask_ai(prompt)
        if not decision:
            # Default decision for ambiguous cases
            decision = {
                "should_click": True,
                "confidence": 0.3,
                "reason": "Title ambiguous, needs inspection",
                "needs_inspection": True
            }
        
        self.cache[cache_key] = decision
        return decision
        
        
        async def extract_complete_page_content(self, page, url: str, html_content: str = "") -> Tuple[str, str, str]:
            """Extract complete page content with scrolling and waiting"""
            logger.log("action", f"📖 Reading complete content from: {url[:50]}...")
            
            try:
                # Wait for page to fully load
                await page.wait_for_load_state('networkidle', timeout=30000)
                await asyncio.sleep(1)
                
                # Scroll down to trigger lazy loading
                await page.evaluate("""
                    async () => {
                        const scrollStep = 500;
                        const scrollDelay = 300;
                        const maxScrolls = 20;
                        
                        for (let i = 0; i < maxScrolls; i++) {
                            window.scrollBy(0, scrollStep);
                            await new Promise(resolve => setTimeout(resolve, scrollDelay));
                            
                            // Check if we're at the bottom
                            if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 100) {
                                break;
                            }
                        }
                        
                        // Scroll back to top
                        window.scrollTo(0, 0);
                    }
                """)
                
                await asyncio.sleep(1)
                
                # Extract title
                title = await page.title()
                if len(title) > 100:
                    title = title[:97] + "..."
                
                # Extract main content using multiple strategies
                content = await page.evaluate("""
                    () => {
                        // Strategy 1: Try to find main content areas
                        const mainSelectors = [
                            'article', 'main', '.article-content', '.post-content', 
                            '.entry-content', '.content-area', '.story-content',
                            '#content', '#main-content', '.main-content',
                            '.post-body', '.article-body', '.story-body'
                        ];
                        
                        let mainContent = "";
                        
                        for (const selector of mainSelectors) {
                            const element = document.querySelector(selector);
                            if (element && element.textContent.trim().length > 500) {
                                mainContent = element.textContent;
                                break;
                            }
                        }
                        
                        // Strategy 2: If no main content found, find the element with most text
                        if (!mainContent) {
                            const allElements = document.querySelectorAll('body > *:not(script):not(style):not(nav):not(header):not(footer)');
                            let maxTextLength = 0;
                            let maxElement = null;
                            
                            for (const el of allElements) {
                                const text = el.textContent.trim();
                                if (text.length > maxTextLength) {
                                    maxTextLength = text.length;
                                    maxElement = el;
                                }
                            }
                            
                            if (maxElement) {
                                mainContent = maxElement.textContent;
                            }
                        }
                        
                        // Strategy 3: Fallback to body text (cleaned)
                        if (!mainContent) {
                            // Remove navigation, sidebars, etc.
                            const bodyClone = document.body.cloneNode(true);
                            const removals = bodyClone.querySelectorAll('nav, header, footer, aside, .sidebar, .navigation, .menu, .ad, .banner');
                            removals.forEach(el => el.remove());
                            mainContent = bodyClone.textContent;
                        }
                        
                        // Clean the content
                        return {
                            title: document.title,
                            content: mainContent.trim().replace(/\\s+/g, ' '),
                            fullHtml: document.documentElement.outerHTML
                        };
                    }
                """)
                
                # Get clean text
                clean_text = self._extract_text(content['content'], 3000)  # Increased to 3000 chars
                
                logger.log("info", f"📖 Extracted {len(clean_text)} characters from page")
                return title, clean_text, content['fullHtml']
                
            except Exception as e:
                logger.log("error", f"❌ Error reading page content: {str(e)[:80]}")
                # Fallback to simple extraction
                try:
                    html_content = await page.content()
                    title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE)
                    title = "Unknown"
                    if title_match:
                        title = html.unescape(title_match.group(1).strip())[:100]
                    
                    text = self._extract_text(html_content, 2000)
                    return title, text, html_content
                except:
                    return "Unknown", "Could not extract content", ""
   
    async def evaluate_after_click(self, topic: str, url: str, title: str, page_text: str, html_content: str = "") -> Dict:
        """Evaluate page content after clicking with complete content"""
        logger.log("decision", f"📊 Evaluating page: {title[:50]}...")
        
        # Enhanced content analysis
        word_count = len(page_text.split())
        logger.log("info", f"  Analyzing {word_count} words of content...")
        
        # Check if content seems substantial
        if word_count < 100:
            return {
                "is_relevant": False,
                "relevance_score": 0.0,
                "reason": f"Page too short ({word_count} words), not enough content to evaluate",
                "key_points": [],
                "summary": "Insufficient content"
            }
        
        # Prepare content for AI (limit to reasonable size)
        content_for_ai = page_text[:4000]
        
        if len(page_text) > 4000:
            logger.log("info", f"  Truncated from {len(page_text)} to 4000 characters for AI analysis")
        
        prompt = f"""Evaluate if this page is relevant to research topic: "{topic}"

        PAGE TITLE: {title}
        URL: {url}
        TOPIC: {topic}

        PAGE CONTENT (first {len(content_for_ai)} characters):
        {content_for_ai}

        INSTRUCTIONS (CRITICAL):
        1. Read the provided content carefully
        2. Determine if this page is relevant to "{topic}"
        3. Provide a relevance score (0-1)
        4. Explain WHY it's relevant or not with specific examples
        5. If relevant, extract 2-3 key points
        6. Provide a 2-3 line summary

        RETURN FORMAT (STRICT JSON ONLY - NO OTHER TEXT):
        {{
            "is_relevant": true/false,
            "relevance_score": 0.0-1.0,
            "reason": "Explanation here",
            "key_points": ["point1", "point2"],
            "summary": "Summary here"
        }}

        IMPORTANT: Return ONLY the JSON object above, nothing else."""
        
        result = await self._ask_ai(prompt)
        
        # If AI returns None or invalid JSON, use fallback
        if not result:
            logger.log("warning", "⚠️ AI evaluation failed, using fallback evaluation")
            result = await self._fallback_evaluation(topic, title, content_for_ai)
        
        # Ensure result has all required fields
        required_fields = ["is_relevant", "relevance_score", "reason", "key_points", "summary"]
        for field in required_fields:
            if field not in result:
                if field == "is_relevant":
                    result[field] = result.get("relevance_score", 0) > 0.4
                elif field == "relevance_score":
                    result[field] = 0.0
                elif field == "key_points":
                    result[field] = []
                else:
                    result[field] = "Not available"
        
        return result

    async def _ask_ai(self, prompt: str) -> Optional[Dict]:
        """Ask AI with better JSON parsing and error handling"""
        # Add strict instruction to only return JSON
        enhanced_prompt = prompt + "\n\nIMPORTANT: Return ONLY the JSON object, nothing else. No explanations, no additional text."
        
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    response = await client.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": enhanced_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "num_predict": 600,  # Reduced for faster response
                                "num_ctx": 2048
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = result.get('response', '').strip()
                        
                        # Log the raw response for debugging (truncated)
                        if attempt > 0:  # Only log on retry
                            logger.log("debug", f"AI raw response (attempt {attempt+1}): {text[:200]}...")
                        
                        # Multiple strategies to extract JSON
                        json_data = None
                        
                        # Strategy 1: Try to parse the entire response as JSON
                        try:
                            json_data = json.loads(text)
                            logger.log("debug", "✓ Parsed entire response as JSON")
                            return json_data
                        except json.JSONDecodeError:
                            pass
                        
                        # Strategy 2: Extract JSON using regex (more robust)
                        try:
                            # Look for JSON object with nested structures
                            json_pattern = r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
                            matches = re.findall(json_pattern, text, re.DOTALL)
                            
                            if matches:
                                # Try each match until one works
                                for match in matches:
                                    try:
                                        json_data = json.loads(match)
                                        logger.log("debug", f"✓ Extracted JSON from response (length: {len(match)})")
                                        return json_data
                                    except json.JSONDecodeError:
                                        continue
                        except Exception as e:
                            logger.log("debug", f"Regex extraction failed: {e}")
                        
                        # Strategy 3: Try to fix common JSON issues
                        try:
                            # Remove markdown code blocks
                            clean_text = re.sub(r'```json\s*|\s*```', '', text)
                            clean_text = re.sub(r'```\s*|\s*```', '', clean_text)
                            
                            # Remove non-ASCII characters
                            clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
                            
                            # Fix common JSON formatting issues
                            clean_text = re.sub(r',\s*}', '}', clean_text)  # Remove trailing commas
                            clean_text = re.sub(r',\s*]', ']', clean_text)  # Remove trailing commas in arrays
                            clean_text = re.sub(r"(\w+)\s*:", r'"\1":', clean_text)  # Quote keys
                            
                            json_data = json.loads(clean_text)
                            logger.log("debug", "✓ Fixed and parsed JSON")
                            return json_data
                        except json.JSONDecodeError:
                            pass
                        
                        # Strategy 4: Extract just the JSON part manually
                        try:
                            # Find first { and last }
                            start = text.find('{')
                            end = text.rfind('}')
                            
                            if start != -1 and end != -1 and end > start:
                                json_str = text[start:end+1]
                                json_data = json.loads(json_str)
                                logger.log("debug", f"✓ Manually extracted JSON (chars {start}-{end})")
                                return json_data
                        except json.JSONDecodeError:
                            pass
                        
                        logger.log("warning", f"⚠️ Could not parse AI response after {attempt+1} attempts")
                        
                    else:
                        logger.log("warning", f"Ollama returned status code: {response.status_code}")
                        
            except httpx.TimeoutException:
                logger.log("warning", f"AI timeout (attempt {attempt+1})")
                await asyncio.sleep(2)
            except Exception as e:
                logger.log("warning", f"AI error (attempt {attempt+1}): {str(e)[:50]}")
                await asyncio.sleep(1)
        
        logger.log("error", "❌ Failed to get valid JSON from AI after 3 attempts")
        return None

    def _extract_text(self, html_content: str, max_length: int = 1000) -> str:
        """Extract clean text from HTML"""
        # Remove scripts, styles
        cleaned = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Get text content
        text = re.sub(r'<[^>]+>', ' ', cleaned)
        text = html.unescape(text)
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text[:max_length]

    async def _fallback_evaluation(self, topic: str, title: str, content: str) -> Dict:
        """Fallback evaluation when JSON parsing fails"""
        # Simple keyword-based relevance check
        topic_words = set(topic.lower().split())
        content_lower = content.lower()
        title_lower = title.lower()
        
        # Count topic word matches
        word_matches = sum(1 for word in topic_words if word in content_lower or word in title_lower)
        
        # Calculate simple relevance score
        relevance_score = min(word_matches / max(len(topic_words), 1) * 0.7, 0.9)
        
        # If title contains topic, boost score
        if any(word in title_lower for word in topic_words):
            relevance_score = max(relevance_score, 0.6)
        
        # Extract first few lines as summary
        lines = content.split('. ')
        summary = '. '.join(lines[:3]) + '.' if len(lines) > 3 else content[:150] + '...'
        
        return {
            "is_relevant": relevance_score > 0.4,
            "relevance_score": relevance_score,
            "reason": f"Fallback evaluation: Found {word_matches} topic word matches",
            "key_points": [f"Contains keywords: {', '.join(topic_words)}"],
            "summary": summary[:200]
        }

    async def _debug_json_parsing(self, text: str):
        """Debug method to log JSON parsing issues"""
        logger.log("debug", "=" * 60)
        logger.log("debug", "JSON PARSING DEBUG")
        logger.log("debug", "=" * 60)
        logger.log("debug", f"Raw text length: {len(text)}")
        logger.log("debug", f"First 500 chars: {text[:500]}")
        logger.log("debug", f"Last 500 chars: {text[-500:]}")
        
        # Look for JSON patterns
        brace_count = text.count('{')
        bracket_count = text.count('[')
        logger.log("debug", f"Braces {{: {brace_count}, Brackets [: {bracket_count}")
        
        # Find positions of braces
        brace_positions = [i for i, char in enumerate(text) if char == '{']
        if brace_positions:
            logger.log("debug", f"First brace at position: {brace_positions[0]}")
            if len(brace_positions) > 1:
                logger.log("debug", f"Second brace at position: {brace_positions[1]}")
        
        logger.log("debug", "=" * 60)

# --- Smart Research Agent ---
class SmartResearchAgent:
    """Intelligent agent that selectively clicks links based on relevance"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.evaluator = LinkEvaluator(model_name)
        self.results = []
        self.visited_urls = set()
        self.stats = {
            "links_scanned": 0,
            "links_clicked": 0,
            "pages_analyzed": 0,
            "relevant_found": 0
        }
    
    async def _analyze_page_structure(self, page, topic: str) -> Dict:
        """Analyze page structure - SIMPLIFIED VERSION"""
        logger.log("thinking", "🔍 Analyzing page structure...")
        
        try:
            # Quick analysis without complex selectors
            structure = await page.evaluate("""
                () => {
                    // Get basic page info
                    const info = {
                        title: document.title || "",
                        url: window.location.href,
                        isHackerNews: window.location.hostname.includes('news.ycombinator.com'),
                        hasTable: document.querySelector('table') !== null,
                        hasLists: document.querySelector('ul, ol') !== null,
                        totalLinks: document.querySelectorAll('a[href]').length,
                        visibleLinks: 0
                    };
                    
                    // Count visible links
                    const allLinks = document.querySelectorAll('a[href]');
                    for (const link of allLinks) {
                        const style = window.getComputedStyle(link);
                        if (style.display !== 'none' && style.visibility !== 'hidden' && 
                            link.textContent.trim().length > 3) {
                            info.visibleLinks++;
                        }
                    }
                    
                    return info;
                }
            """)
            
            # For Hacker News, use specific selectors
            if structure.get('isHackerNews', False):
                return {
                    "is_promising_page": True,
                    "target_selectors": [".athing .titleline > a", "tr.athing"],
                    "avoid_selectors": ["#hnmain > tbody > tr:first-child", "#hnmain > tbody > tr:last-child"],
                    "confidence": 0.9,
                    "reasoning": "Hacker News page detected - targeting story links"
                }
            
            # For pages with tables (like data listings)
            if structure.get('hasTable', False) and structure.get('visibleLinks', 0) > 50:
                return {
                    "is_promising_page": True,
                    "target_selectors": ["table a", "tr a", "td a"],
                    "avoid_selectors": ["nav", ".sidebar", "header", "footer"],
                    "confidence": 0.7,
                    "reasoning": "Table-based layout with many links - targeting table cells"
                }
            
            # For pages with lists
            if structure.get('hasLists', False) and structure.get('visibleLinks', 0) > 30:
                return {
                    "is_promising_page": True,
                    "target_selectors": ["li a", "ul a", "ol a", ".list-item a"],
                    "avoid_selectors": ["nav", ".sidebar", "header", "footer", ".pagination"],
                    "confidence": 0.7,
                    "reasoning": "List-based layout - targeting list items"
                }
            
            # Default: look for main content areas
            return {
                "is_promising_page": True,
                "target_selectors": ["main", "article", ".content", "#content", ".main", ".post", ".item"],
                "avoid_selectors": ["nav", ".sidebar", "aside", "header", "footer", ".ad", ".banner"],
                "confidence": 0.5,
                "reasoning": "Default targeting main content areas"
            }
            
        except Exception as e:
            logger.log("error", f"Structure analysis error: {e}")
            # Fallback
            return {
                "is_promising_page": True,
                "target_selectors": ["a[href]"],  # All links as fallback
                "avoid_selectors": [],
                "confidence": 0.3,
                "reasoning": "Fallback to all links"
            }
            
    async def _get_page_preview(self, page) -> str:
        """Get a quick preview of the page for debugging"""
        try:
            preview = await page.evaluate("""
                () => {
                    return {
                        title: document.title,
                        url: window.location.href,
                        mainHeading: document.querySelector('h1')?.textContent?.trim() || 'No H1',
                        firstParagraph: document.querySelector('p')?.textContent?.trim()?.slice(0, 100) || 'No paragraph',
                        linkCount: document.querySelectorAll('a[href]').length
                    };
                }
            """)
            return f"Title: {preview['title']}\nURL: {preview['url']}\nH1: {preview['mainHeading']}\nFirst 100 chars: {preview['firstParagraph']}\nTotal links: {preview['linkCount']}"
        except:
            return "Could not get page preview"
   
    async def research(self, topic: str, start_url: str) -> List[ResearchResult]:
        """Main research function with simple, reliable link extraction"""
        logger.log("info", f"🚀 Starting focused research on: {topic}")
        logger.log("info", f"📍 Starting from: {start_url}")
        logger.log("info", f"🤖 Using model: {self.model_name}")
        
        if not start_url.startswith('http'):
            start_url = 'https://' + start_url
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                headless=False,
                args=['--disable-blink-features=AutomationControlled']
            )
            
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            try:
                # Step 1: Visit start page
                logger.log("action", "🌐 Loading start page...")
                page = await context.new_page()
                await page.goto(start_url, wait_until='domcontentloaded', timeout=30000)
                await asyncio.sleep(2)
                await handle_popups(page)
                
                preview = await self._get_page_preview(page)
                logger.log("info", f"📄 Page preview:\n{preview}")
                await self._debug_page_content(page)
                # Step 2: SIMPLE link extraction (reliable method)
                logger.log("action", "🔍 Scanning for article links...")
                all_links = await self._extract_links_smart(page, topic, start_url)                
                self.stats["links_scanned"] = len(all_links)
                
                if not all_links:
                    logger.log("error", "❌ No links found on start page")
                    return []
                
                logger.log("success", f"📊 Found {len(all_links)} article links")
                
                # Step 3: Show found links
                print(f"\n{'='*60}")
                print(f"🔍 Found {len(all_links)} article links")
                print("\nTop 15 links found:")
                for i, link in enumerate(all_links[:15], 1):
                    print(f"  {i}. {link['text'][:60]}...")
                
                # Step 4: Ask how many to evaluate
                while True:
                    try:
                        eval_input = input(f"\nHow many links should I evaluate? (1-{min(50, len(all_links))}, or 'all'): ").strip()
                        
                        if eval_input.lower() == 'all':
                            links_to_evaluate = all_links[:50]  # Cap at 50
                            print(f"⚡ Will evaluate {len(links_to_evaluate)} links")
                            break
                        else:
                            eval_count = int(eval_input)
                            if 1 <= eval_count <= min(50, len(all_links)):
                                links_to_evaluate = all_links[:eval_count]
                                break
                            else:
                                print(f"❌ Please enter a number between 1 and {min(50, len(all_links))}")
                    except ValueError:
                        print("❌ Please enter a valid number or 'all'")
                
                # Step 5: SIMPLE sequential evaluation (more reliable)
                logger.log("thinking", f"🤔 Evaluating {len(links_to_evaluate)} links...")
                promising_links = []
                
                for i, link in enumerate(links_to_evaluate, 1):
                    logger.log("info", f"  Evaluating {i}/{len(links_to_evaluate)}: {link['text'][:40]}...")
                    
                    if link['url'] in self.visited_urls:
                        continue
                    
                    # Quick heuristic: skip obvious non-articles
                    text_lower = link['text'].lower()
                    if (len(text_lower) < 10 or 
                        text_lower in ['more', 'next', 'previous', 'home', 'about', 'contact'] or
                        any(x in text_lower for x in ['login', 'sign up', 'register', 'password'])):
                        logger.log("decision", "    ✗ Skipping (not an article)")
                        continue
                    
                    # Get AI decision
                    decision = await self.evaluator.evaluate_link_from_title(topic, link['text'], link['url'])
                    
                    if decision.get('should_click', False):
                        confidence = decision.get('confidence', 0)
                        if confidence > 0.2:  # Minimum confidence threshold
                            promising_links.append({
                                'url': link['url'],
                                'text': link['text'],
                                'confidence': confidence,
                                'reason': decision.get('reason', '')
                            })
                            logger.log("decision", f"    ✓ Will inspect (confidence: {confidence:.2f})")
                        else:
                            logger.log("decision", f"    ✗ Skipping (low confidence: {confidence:.2f})")
                    else:
                        logger.log("decision", f"    ✗ Skipping: {decision.get('reason', 'Not relevant')}")
                
                if not promising_links:
                    logger.log("warning", "⚠️ No promising links found!")
                    return []
                
                # Step 6: Sort by confidence and ask how many to visit
                promising_links.sort(key=lambda x: x['confidence'], reverse=True)
                logger.log("success", f"🎯 Found {len(promising_links)} promising links!")
                
                print(f"\n{'='*60}")
                print(f"✅ Found {len(promising_links)} promising links (confidence > 0.3)")
                print("\nTop 10 promising links:")
                for i, link in enumerate(promising_links[:10], 1):
                    print(f"  {i}. [{link['confidence']:.2f}] {link['text'][:60]}...")
                    print(f"     Reason: {link['reason'][:60]}...")
                
                while True:
                    try:
                        visit_input = input(f"\nHow many promising links should I visit? (1-{len(promising_links)}): ").strip()
                        links_to_visit_count = int(visit_input)
                        if 1 <= links_to_visit_count <= len(promising_links):
                            links_to_visit = promising_links[:links_to_visit_count]
                            break
                        else:
                            print(f"❌ Please enter a number between 1 and {len(promising_links)}")
                    except ValueError:
                        print("❌ Please enter a valid number")
                
                logger.log("info", f"📋 Will visit top {len(links_to_visit)} most promising links")
                
                # Step 7: Visit and analyze selected links
                failed_pages = []
                successful_pages = []
                
                for i, link_info in enumerate(links_to_visit, 1):
                    url = link_info['url']
                    logger.log("action", f"🔗 [{i}/{len(links_to_visit)}] Inspecting: {url[:70]}...")
                    
                    try:
                        # Open link in new tab
                        new_page = await context.new_page()
                        await new_page.goto(url, wait_until='domcontentloaded', timeout=20000)
                        await asyncio.sleep(2)
                        await handle_popups(new_page)
                        
                        # Extract complete page content
                        title, page_text, html_content = await self.evaluator.extract_complete_page_content(new_page, url)
                        
                        logger.log("info", f"  Page title: {title}")
                        logger.log("info", f"  Content length: {len(page_text)} characters")
                        
                        # Skip if page looks like search/results page
                        title_lower = title.lower()
                        if any(x in title_lower for x in ['search', 'results', 'query', 'find', 'looking for']):
                            logger.log("warning", f"⚠️ Skipping search results page: {title}")
                            await new_page.close()
                            continue
                        
                        # Skip if content is too short (likely not an article)
                        if len(page_text) < 300:
                            logger.log("warning", f"⚠️ Page too short ({len(page_text)} chars), likely not an article")
                            await new_page.close()
                            continue
                        
                        # Evaluate page content
                        evaluation = await self.evaluator.evaluate_after_click(topic, url, title, page_text, html_content)
                        
                        self.stats["links_clicked"] += 1
                        self.stats["pages_analyzed"] += 1
                        
                        if evaluation.get('is_relevant', False) and evaluation.get('relevance_score', 0) > 0.4:
                            self.stats["relevant_found"] += 1
                            
                            # Create result
                            result = ResearchResult(
                                url=url,
                                title=title,
                                relevance_score=evaluation.get('relevance_score', 0.5),
                                summary=evaluation.get('summary', 'No summary available'),
                                relevance_reason=evaluation.get('reason', ''),
                                key_points=evaluation.get('key_points', [])
                            )
                            
                            self.results.append(result)
                            successful_pages.append(url)
                            
                            logger.log("success", f"✅ RELEVANT! Score: {result.relevance_score:.2f}")
                            logger.log("info", f"   Summary: {result.summary[:80]}...")
                            
                            # Save intermediate results
                            await self._save_results(topic)
                        else:
                            score = evaluation.get('relevance_score', 0)
                            reason = evaluation.get('reason', 'Unknown')
                            
                            # Check if this might be a JSON parsing issue
                            if score == 0.0 and "AI evaluation failed" in reason:
                                failed_pages.append({
                                    'url': url,
                                    'title': title,
                                    'reason': 'JSON parsing issue'
                                })
                                logger.log("warning", f"⚠️ Potential parsing issue: {reason[:60]}...")
                            else:
                                logger.log("warning", f"⚠️ Not relevant (score: {score:.2f}): {reason[:60]}...")
                        
                        self.visited_urls.add(url)
                        await new_page.close()
                        
                        # Small delay between visits
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.log("error", f"❌ Error inspecting {url[:50]}...: {str(e)[:80]}")
                        continue
                
                # Retry failed pages (if any)
                if failed_pages and len(successful_pages) < max(3, len(links_to_visit) // 2):
                    logger.log("info", f"🔄 Retrying {len(failed_pages)} pages with JSON parsing issues...")
                    
                    for i, failed_page in enumerate(failed_pages, 1):
                        logger.log("action", f"🔄 Retry {i}/{len(failed_pages)}: {failed_page['url'][:70]}...")
                        
                        try:
                            # Re-open the page
                            retry_page = await context.new_page()
                            await retry_page.goto(failed_page['url'], wait_until='domcontentloaded', timeout=15000)
                            await asyncio.sleep(1)
                            
                            # Extract content again (simpler this time)
                            html = await retry_page.content()
                            title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE)
                            title = failed_page['title']
                            if title_match and title == "Unknown":
                                title = html.unescape(title_match.group(1).strip())[:100]
                            
                            text = self.evaluator._extract_text(html, 2000)
                            
                            # Use simplified evaluation
                            evaluation = await self.evaluator.evaluate_after_click(topic, failed_page['url'], title, text, html)
                            
                            if evaluation.get('is_relevant', False) and evaluation.get('relevance_score', 0) > 0.4:
                                result = ResearchResult(
                                    url=failed_page['url'],
                                    title=title,
                                    relevance_score=evaluation.get('relevance_score', 0.5),
                                    summary=evaluation.get('summary', 'Evaluation succeeded on retry'),
                                    relevance_reason="Retry successful: " + evaluation.get('reason', ''),
                                    key_points=evaluation.get('key_points', [])
                                )
                                
                                self.results.append(result)
                                logger.log("success", f"✅ RETRY SUCCESS! Score: {result.relevance_score:.2f}")
                                
                                # Save updated results
                                await self._save_results(topic)
                            
                            await retry_page.close()
                            await asyncio.sleep(0.5)
                            
                        except Exception as e:
                            logger.log("error", f"❌ Retry failed: {str(e)[:80]}")                
                            continue
                
                await page.close()
                
            finally:
                await browser.close()
        
        # Final summary and save
        await self._save_results(topic)
        logger.log("success", "🎉 Research complete!")
        logger.log("info", f"📊 Statistics:")
        logger.log("info", f"   Links scanned: {self.stats['links_scanned']}")
        logger.log("info", f"   Links evaluated: {len(links_to_evaluate)}")
        logger.log("info", f"   Links clicked: {self.stats['links_clicked']}")
        logger.log("info", f"   Pages analyzed: {self.stats['pages_analyzed']}")
        logger.log("info", f"   Relevant found: {self.stats['relevant_found']}")
        
        return self.results

    async def _extract_links_smart(self, page, topic: str, start_url: str) -> List[Dict]:
        """Smart link extraction - FIXED VERSION"""
        logger.log("action", "🤔 Analyzing page to find relevant link areas...")
        
        # First, analyze the page structure
        structure_analysis = await self._analyze_page_structure(page, topic)
        
        if not structure_analysis.get('is_promising_page', True):
            logger.log("warning", "⚠️ Page doesn't seem promising for our topic")
            return []
        
        target_selectors = structure_analysis.get('target_selectors', ["a[href]"])
        avoid_selectors = structure_analysis.get('avoid_selectors', [])
        
        logger.log("info", f"🎯 Will target: {target_selectors}")
        logger.log("info", f"🚫 Will avoid: {avoid_selectors}")
        
        try:
            # Build selector strings
            target_selector_str = ", ".join([f"{sel}:not(nav):not(header):not(footer):not(.sidebar)" 
                                           for sel in target_selectors])
            avoid_selector_str = ", ".join(avoid_selectors)
            
            # SINGLE ARGUMENT version to fix "4 were given" error
            links = await page.evaluate("""
                ([targetSelectors, avoidSelectors]) => {
                    const links = [];
                    const targetSelectorStr = targetSelectors;
                    const avoidSelectorStr = avoidSelectors;
                    
                    // Get all matching links
                    const allLinks = document.querySelectorAll(targetSelectorStr);
                    
                    for (const link of allLinks) {
                        const text = link.textContent.trim();
                        const url = link.href;
                        
                        // Basic filtering
                        if (!text || !url || text.length < 5) {
                            continue;
                        }
                        
                        // Must be HTTP link
                        if (!url.startsWith('http')) {
                            continue;
                        }
                        
                        // Check if link is inside avoided container
                        let isInAvoided = false;
                        let parent = link.parentElement;
                        while (parent && parent !== document.body) {
                            if (avoidSelectorStr && parent.matches && parent.matches(avoidSelectorStr)) {
                                isInAvoided = true;
                                break;
                            }
                            parent = parent.parentElement;
                        }
                        
                        if (isInAvoided) {
                            continue;
                        }
                        
                        // Skip navigation/UI links
                        const lowerText = text.toLowerCase();
                        const skipWords = [
                            'login', 'signin', 'register', 'signup', 'account',
                            'profile', 'settings', 'home', 'about', 'contact',
                            'privacy', 'terms', 'cookie', 'policy',
                            'next', 'previous', 'back', 'menu',
                            'share', 'like', 'follow', 'subscribe'
                        ];
                        
                        if (skipWords.some(word => lowerText.includes(word))) {
                            continue;
                        }
                        
                        // Skip very short or single-word links
                        if (text.split(' ').length < 2 && text.length < 10) {
                            continue;
                        }
                        
                        // Add the link
                        links.push({
                            text: text,
                            url: url
                        });
                    }
                    
                    return links;
                }
            """, [target_selector_str, avoid_selector_str])  # Pass as single array
            
            # Remove duplicates by URL
            seen = set()
            unique_links = []
            for link in links:
                # Clean URL
                clean_url = link['url'].split('#')[0].split('?')[0].rstrip('/')
                if clean_url not in seen and clean_url.startswith('http'):
                    seen.add(clean_url)
                    unique_links.append({
                        'text': link['text'][:150],
                        'url': clean_url
                    })
            
            logger.log("success", f"📰 Smart extraction found {len(unique_links)} links")
            
            # If we didn't find enough links, try broader approach
            if len(unique_links) < 10:
                logger.log("warning", "⚠️ Smart extraction found few links, trying broader search...")
                fallback_links = await self._extract_links_fallback(page, start_url)
                # Combine and deduplicate
                for link in fallback_links:
                    clean_url = link['url'].split('#')[0].split('?')[0].rstrip('/')
                    if clean_url not in seen and clean_url.startswith('http'):
                        seen.add(clean_url)
                        unique_links.append(link)
                logger.log("success", f"📰 After fallback: {len(unique_links)} total links")
            
            return unique_links
            
        except Exception as e:
            logger.log("error", f"Smart extraction error: {str(e)[:100]}")
            # Fallback to simple extraction
            return await self._extract_links_fallback(page, start_url)    

    async def _extract_links_fallback(self, page, start_url: str) -> List[Dict]:
        """Fallback method - MORE INCLUSIVE VERSION"""
        try:
            # Different strategies for different sites
            if 'news.ycombinator.com' in start_url:
                # Hacker News - get ALL story links including jobs
                links = await page.evaluate("""
                    () => {
                        const links = [];
                        // Get main story links
                        const storyLinks = document.querySelectorAll('.titleline > a, .athing .title > a');
                        for (const link of storyLinks) {
                            const text = link.textContent.trim();
                            const url = link.href;
                            if (text && url && url.startsWith('http') && text.length > 5) {
                                links.push({text: text, url: url});
                            }
                        }
                        return links;
                    }
                """)
            else:
                # Generic extraction - MORE BROAD
                links = await page.evaluate("""
                    () => {
                        const links = [];
                        const allLinks = document.querySelectorAll('a[href]');
                        
                        for (const link of allLinks) {
                            const text = link.textContent.trim();
                            const url = link.href;
                            
                            // Basic filtering
                            if (!text || !url || text.length < 5) {
                                continue;
                            }
                            
                            // Must be HTTP link
                            if (!url.startsWith('http')) {
                                continue;
                            }
                            
                            // Skip obvious navigation
                            const lowerText = text.toLowerCase();
                            const skipExact = ['login', 'signin', 'register', 'signup', 'more', 'next', 'previous'];
                            if (skipExact.includes(lowerText)) {
                                continue;
                            }
                            
                            // Skip very common UI patterns
                            if (lowerText.includes('privacy policy') || 
                                lowerText.includes('terms of service') ||
                                lowerText.includes('cookie policy')) {
                                continue;
                            }
                            
                            // Get link visibility
                            const style = window.getComputedStyle(link);
                            if (style.display === 'none' || style.visibility === 'hidden') {
                                continue;
                            }
                            
                            // Check if link is likely in main content (not nav/sidebar)
                            let parent = link.parentElement;
                            let depth = 0;
                            let inNav = false;
                            while (parent && parent !== document.body && depth < 5) {
                                const tag = parent.tagName.toLowerCase();
                                const cls = parent.className.toLowerCase();
                                const id = parent.id.toLowerCase();
                                
                                if (tag === 'nav' || cls.includes('nav') || id.includes('nav') ||
                                    cls.includes('sidebar') || id.includes('sidebar') ||
                                    cls.includes('menu') || id.includes('menu')) {
                                    inNav = true;
                                    break;
                                }
                                parent = parent.parentElement;
                                depth++;
                            }
                            
                            if (!inNav) {
                                links.push({text: text, url: url});
                            }
                        }
                        return links;
                    }
                """)
            
            # Remove duplicates
            seen = set()
            unique_links = []
            for link in links:
                clean_url = link['url'].split('#')[0].split('?')[0].rstrip('/')
                if clean_url not in seen and clean_url.startswith('http'):
                    seen.add(clean_url)
                    unique_links.append({
                        'text': link['text'][:200],  # Allow longer text
                        'url': clean_url
                    })
            
            logger.log("success", f"📰 Fallback extracted {len(unique_links)} links")
            return unique_links
            
        except Exception as e:
            logger.log("error", f"Fallback extraction error: {str(e)[:100]}")
            return []  
    
    async def _save_results(self, topic: str):
        """Save results to file"""
        if not self.results:
            return
        
        # Sort by relevance score
        sorted_results = sorted(self.results, key=lambda x: x.relevance_score, reverse=True)
        
        # Save JSON
        try:
            with open("research_results.json", "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in sorted_results], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.log("error", f"Failed to save JSON: {e}")
        
        # Save clean summary
        try:
            with open("research_summary.txt", "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write(f"RESEARCH REPORT: {topic}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model used: {self.model_name}\n")
                f.write(f"Relevant pages found: {len(sorted_results)}\n")
                f.write("=" * 70 + "\n\n")
                
                for i, result in enumerate(sorted_results, 1):
                    f.write(f"{i}. [{result.relevance_score:.2f}] {result.title}\n")
                    f.write(f"   URL: {result.url}\n")
                    f.write(f"   Why relevant: {result.relevance_reason}\n")
                    f.write(f"   Summary: {result.summary}\n")
                    
                    if result.key_points:
                        f.write(f"   Key points:\n")
                        for point in result.key_points:
                            f.write(f"     • {point}\n")
                    
                    f.write("\n" + "-" * 70 + "\n\n")
            
            logger.log("success", f"💾 Saved {len(sorted_results)} results to research_summary.txt")
            
        except Exception as e:
            logger.log("error", f"Failed to save summary: {e}")

    async def _debug_page_content(self, page):
        """Debug: See what's actually on the page"""
        try:
            debug_info = await page.evaluate("""
                () => {
                    const info = {
                        title: document.title,
                        url: window.location.href,
                        allLinks: document.querySelectorAll('a[href]').length,
                        visibleLinks: 0,
                        tables: document.querySelectorAll('table').length,
                        lists: document.querySelectorAll('ul, ol').length,
                        mainElements: [],
                        hackerNewsCheck: document.querySelector('.athing') !== null
                    };
                    
                    // Count visible links
                    const links = document.querySelectorAll('a[href]');
                    for (const link of links) {
                        const style = window.getComputedStyle(link);
                        if (style.display !== 'none' && style.visibility !== 'hidden') {
                            info.visibleLinks++;
                            if (link.textContent.trim().length > 20) {
                                info.mainElements.push({
                                    text: link.textContent.trim().substring(0, 50),
                                    href: link.href.substring(0, 50)
                                });
                            }
                        }
                    }
                    
                    return info;
                }
            """)
            
            logger.log("debug", f"📊 Page debug - Title: {debug_info['title']}")
            logger.log("debug", f"📊 Total links: {debug_info['allLinks']}, Visible: {debug_info['visibleLinks']}")
            logger.log("debug", f"📊 Tables: {debug_info['tables']}, Lists: {debug_info['lists']}")
            logger.log("debug", f"📊 Hacker News detected: {debug_info['hackerNewsCheck']}")
            
            # Show some sample links
            if debug_info['mainElements']:
                logger.log("debug", "📊 Sample links found:")
                for i, elem in enumerate(debug_info['mainElements'][:5]):
                    logger.log("debug", f"  {i+1}. {elem['text']} -> {elem['href']}")
                    
        except Exception as e:
            logger.log("error", f"Debug error: {e}")

    async def generate_link_summaries(self, links: List[Dict], topic: str) -> List[Dict]:
        """Generate summaries for a list of links without clicking"""
        logger.log("info", f"📝 Generating summaries for {len(links)} links...")
        
        summaries = []
        
        for i, link in enumerate(links, 1):
            logger.log("info", f"  Summarizing link {i}/{len(links)}: {link['text'][:50]}...")
            
            # Create a prompt for summary generation
            prompt = f"""Generate a summary for this link in relation to research topic: "{topic}"

            LINK TEXT: {link['text']}
            URL: {link['url']}
            TOPIC: {topic}

            Instructions:
            1. Based on the link text and URL, infer what the content might be about
            2. Explain how it might relate to "{topic}"
            3. Predict what kind of information it might contain
            4. Rate the potential relevance (0-1)

            Response format (JSON only):
            {{
                "link_text": "{link['text'][:100]}",
                "url": "{link['url']}",
                "predicted_content": "What you think this link contains",
                "potential_relevance": 0.0-1.0,
                "relevance_explanation": "Why it might be relevant to {topic}",
                "recommendation": "Should this link be clicked? (yes/no/maybe)"
            }}"""
            
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.3,
                                "num_predict": 400
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = result.get('response', '')
                        
                        # Extract JSON
                        try:
                            json_match = re.search(r'\{.*\}', text, re.DOTALL)
                            if json_match:
                                summary = json.loads(json_match.group())
                                summaries.append(summary)
                                logger.log("success", f"    ✓ Generated summary (relevance: {summary.get('potential_relevance', 0):.2f})")
                            else:
                                logger.log("warning", "    Could not parse summary response")
                        except:
                            logger.log("warning", "    Failed to parse summary as JSON")
            except Exception as e:
                logger.log("error", f"    Error generating summary: {str(e)[:50]}")
        
        return summaries

# --- Main Menu ---
async def main():
    """Main application"""
    print("=" * 70)
    print("🤖 INTELLIGENT RESEARCH AGENT - SELECTIVE CLICKING")
    print("=" * 70)
    print("Strategy: Scans links, evaluates titles, clicks only on relevant ones")
    print("=" * 70)
    
    # Check Ollama and find model
    success, model_name = await check_ollama_connection()
    
    if not success:
        print("\n❌ Cannot connect to Ollama or find suitable model")
        print("Please make sure Ollama is running with a model loaded")
        return
    
    while True:
        print("\n" + "-" * 70)
        print("MAIN MENU:")
        print("1. Start Intelligent Research")
        print("2. Generate Link Summaries (without clicking)")
        print("3. View Previous Results")
        print("4. View Thinking Log")
        print("5. Exit")
        print("-" * 70)
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == "1":
            print("\n" + "=" * 60)
            print("🎯 INTELLIGENT RESEARCH SETUP")
            print("=" * 60)
            
            topic = input("\nResearch topic (be specific): ").strip()
            if not topic:
                print("❌ Topic is required")
                continue
            
            start_url = input("\nStarting URL (press Enter for Hacker News): ").strip()
            if not start_url:
                start_url = "https://news.ycombinator.com"
            
            print(f"\n{'='*60}")
            print(f"Starting research for: {topic}")
            print(f"Starting from: {start_url}")
            print(f"Using model: {model_name}")
            print(f"{'='*60}")
            print("\nThe agent will now:")
            print("1. Scan the page for ALL links")
            print("2. Show you how many links found")
            print("3. Ask how many to evaluate")
            print("4. Show promising links")
            print("5. Ask how many to visit")
            print("6. Analyze visited pages")
            print(f"{'='*60}")
            
            input("\nPress Enter to start (this will open a browser)...")
            
            # Run research WITHOUT max_pages parameter
            agent = SmartResearchAgent(model_name=model_name)
            await agent.research(topic, start_url)  # No max_pages parameter
            
            if agent.results:
                print(f"\n✅ Found {len(agent.results)} relevant pages!")
                print("\n📋 TOP RESULTS:")
                for i, result in enumerate(agent.results[:5], 1):
                    print(f"\n{i}. [{result.relevance_score:.2f}] {result.title}")
                    print(f"   URL: {result.url[:60]}...")
                    print(f"   Summary: {result.summary[:80]}...")
            else:
                print("\n❌ No relevant pages found.")
            
            input("\nPress Enter to continue...")
        
        elif choice == "2":
            print("\n" + "=" * 60)
            print("📝 LINK SUMMARIES GENERATOR")
            print("=" * 60)
            
            topic = input("\nResearch topic: ").strip()
            if not topic:
                print("❌ Topic is required")
                continue
            
            start_url = input("\nURL to scan for links: ").strip()
            if not start_url:
                start_url = "https://news.ycombinator.com"
            
            print(f"\n{'='*60}")
            print(f"Generating summaries for links about: {topic}")
            print(f"Scanning page: {start_url}")
            print(f"{'='*60}")
            
            input("\nPress Enter to start...")
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False)
                context = await browser.new_context()
                page = await context.new_page()
                
                try:
                    await page.goto(start_url, wait_until='domcontentloaded', timeout=20000)
                    await asyncio.sleep(2)
                    
                    # Extract links
                    agent = SmartResearchAgent(model_name=model_name)
                    all_links = await agent._extract_links_smart(page, topic, start_url)
                    
                    if not all_links:
                        print("❌ No links found")
                        continue
                    
                    print(f"\n✅ Found {len(all_links)} links")
                    
                    # Ask how many to summarize
                    while True:
                        try:
                            count_input = input(f"How many links to summarize? (1-{min(20, len(all_links))}): ").strip()
                            count = int(count_input)
                            if 1 <= count <= min(20, len(all_links)):
                                links_to_summarize = all_links[:count]
                                break
                            else:
                                print(f"❌ Please enter 1-{min(20, len(all_links))}")
                        except ValueError:
                            print("❌ Please enter a valid number")
                    
                    # Generate summaries
                    summaries = await agent.generate_link_summaries(links_to_summarize, topic)
                    
                    # Display results
                    print(f"\n{'='*60}")
                    print(f"📋 SUMMARIES FOR {len(summaries)} LINKS")
                    print(f"{'='*60}")
                    
                    for i, summary in enumerate(summaries, 1):
                        print(f"\n{i}. {summary.get('link_text', 'Unknown')}")
                        print(f"   URL: {summary.get('url', 'Unknown')}")
                        print(f"   Predicted: {summary.get('predicted_content', 'Unknown')[:100]}...")
                        print(f"   Potential Relevance: {summary.get('potential_relevance', 0):.2f}")
                        print(f"   Recommendation: {summary.get('recommendation', 'Unknown')}")
                    
                    # Save to file
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"link_summaries_{timestamp}.json"
                        with open(filename, "w", encoding="utf-8") as f:
                            json.dump(summaries, f, indent=2, ensure_ascii=False)
                        print(f"\n💾 Saved to {filename}")
                    except Exception as e:
                        print(f"❌ Failed to save: {e}")
                    
                finally:
                    await browser.close()
            
            input("\nPress Enter to continue...")
        
        elif choice == "3":
            print("\n" + "=" * 60)
            print("📝 THINKING LOG")
            print("=" * 60)
            
            logs = logger.get_recent_logs(30)
            if logs:
                for log in logs:
                    print(f"[{log['timestamp']}] {log['type'].upper()}: {log['message']}")
                    if log['details']:
                        print(f"    Details: {log['details']}")
            else:
                print("No logs yet.")
            
            input("\nPress Enter to continue...")
        
        elif choice == "4":
            print("\n👋 Goodbye!")
            logger.save_to_file()
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    # Banner
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                🧠 INTELLIGENT RESEARCH AGENT                 ║
    ║                Smart Link Evaluation System                  ║
    ║                                                              ║
    ║  How it works:                                               ║
    ║  1. Scans page for all links                                 ║
    ║  2. Uses AI to evaluate link titles                          ║
    ║  3. Clicks ONLY on promising links                           ║
    ║  4. Analyzes page content                                    ║
    ║  5. Returns focused, relevant results                        ║
    ║                                                              ║
    ║  Output: Score + Why it's relevant + 4-line summary          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    print(banner)
    
    asyncio.run(main())