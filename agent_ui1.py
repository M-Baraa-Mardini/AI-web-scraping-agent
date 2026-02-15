"""
AI Research Agent - Windows Desktop Application
COMPLETE UI CONTROL VERSION - No command line prompts
"""
import sys
import os

# Check for tkinter
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
except ImportError as e:
    print("ERROR: tkinter is not available.")
    print("Please install tkinter for your system:")
    print("  Windows: Comes with Python installation")
    print("  Ubuntu/Debian: sudo apt-get install python3-tk")
    input("Press Enter to exit...")
    sys.exit(1)
import logger
import threading
import queue
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import traceback
import re

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from playwright.async_api import async_playwright  
# Try to import the agent
try:
    from agent import SmartResearchAgent, check_ollama_connection
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agent module: {e}")
    AGENT_AVAILABLE = False

class ModernUIAgent(SmartResearchAgent):
    """Extended agent that communicates with UI instead of command line"""
    
    def __init__(self, model_name: str, ui_callback=None, ignore_terms=None):
        super().__init__(model_name)
        self.ui_callback = ui_callback  # Function to call for UI updates
        self.user_choices = {}  # Store user choices from UI
        self.pending_questions = queue.Queue()
        self.answers = queue.Queue()
        self.stop_flag = False  # Add stop flag
        self.ignore_terms = ignore_terms or []  # Add ignore terms
    
    
    def stop(self):
        """Stop the agent"""
        self.stop_flag = True
    
# Update the ModernUIAgent class methods:

    async def research(self, topic: str, start_url: str, countries: List[str] = None) -> List[Any]:
        """Override parent research method to handle countries and UI communication"""
        return await self._research_with_ui(topic, start_url, countries)
    
    async def ask_user_choice(self, question: str, options: List[str], default=None) -> str:
        """Ask user for a choice via UI - FIXED VERSION"""
        if self.ui_callback:
            # Send question to UI
            self.ui_callback('user_choice', {
                'question': question,
                'options': options,
                'default': str(default) if default else None
            })
            
            # Wait for response from UI - use asyncio queue properly
            try:
                # Use asyncio.wait_for to handle timeouts
                answer = await asyncio.wait_for(self._get_answer_from_queue(), timeout=300)
                return answer
            except asyncio.TimeoutError:
                return str(default) if default else (options[0] if options else "")
        return str(default) if default else (options[0] if options else "")

    async def _get_answer_from_queue(self):
        """Get answer from queue, converting to async"""
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, self.answers.get)
        return answer
    
    async def show_progress(self, stage: str, current: int, total: int, message: str):
        """Show progress in UI"""
        if self.ui_callback:
            self.ui_callback('progress', {
                'stage': stage,
                'current': current,
                'total': total,
                'message': message
            })
        
    async def _extract_links_smart(self, page, topic: str, start_url: str) -> List[Dict]:
        """Smart link extraction - FIXED VERSION (ORIGINAL)"""
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
            
            # Apply ignore terms from UI if provided
            if hasattr(self, 'ignore_terms') and self.ignore_terms:
                filtered_links = []
                for link in unique_links:
                    text_lower = link['text'].lower()
                    url_lower = link['url'].lower()
                    
                    # Check if link contains any ignore term
                    should_skip = False
                    for term in self.ignore_terms:
                        if term in text_lower or term in url_lower:
                            should_skip = True
                            if self.ui_callback:
                                self.ui_callback('log', {
                                    'type': 'decision',
                                    'message': f'✗ Skipping (contains ignore term: {term})'
                                })
                            break
                    
                    if not should_skip:
                        filtered_links.append(link)
                
                unique_links = filtered_links
                logger.log("info", f"📰 After ignore filter: {len(unique_links)} links")
            
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
            fallback_links = await self._extract_links_fallback(page, start_url)
            
            # Apply ignore terms to fallback links too
            if hasattr(self, 'ignore_terms') and self.ignore_terms:
                filtered_links = []
                for link in fallback_links:
                    text_lower = link['text'].lower()
                    url_lower = link['url'].lower()
                    
                    should_skip = False
                    for term in self.ignore_terms:
                        if term in text_lower or term in url_lower:
                            should_skip = True
                            break
                    
                    if not should_skip:
                        filtered_links.append(link)
                
                fallback_links = filtered_links
            
            return fallback_links

    async def _research_with_ui(self, topic: str, start_url: str, countries: List[str] = None) -> List[Any]:
        """Research method that communicates with UI - FIXED ASYNC VERSION"""
        # Initialize variables
        promising_links = []
        results = []
        
        # Parse countries if provided
        if countries and isinstance(countries, str):
            countries = [c.strip() for c in countries.split(',') if c.strip()]
        
        # Log start with countries
        if self.ui_callback:
            country_text = f" in {', '.join(countries)}" if countries else ""
            self.ui_callback('log', {
                'type': 'info',
                'message': f'🚀 Starting focused research on: {topic}{country_text}',
                'details': f'Starting from: {start_url}'
            })
        
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
            
            page = None
            
            try:
                # Step 1: Visit start page
                if self.ui_callback:
                    self.ui_callback('log', {
                        'type': 'action',
                        'message': '🌐 Loading start page...'
                    })
                
                page = await context.new_page()
                
                # Use new navigation method with cookie handling
                success = await self._navigate_and_handle_cookies(page, start_url)
                if not success:
                    if self.ui_callback:
                        self.ui_callback('log', {
                            'type': 'error',
                            'message': f'❌ Failed to load page: {start_url}'
                        })
                    return []
                
                # Step 2: Extract links
                if self.ui_callback:
                    self.ui_callback('log', {
                        'type': 'action',
                        'message': '🔍 Scanning for links...'
                    })
                
                # Extract links
                all_links = await self._extract_links_smart(page, topic, start_url)
                self.stats["links_scanned"] = len(all_links)
                
                # Store links for UI access
                self.all_links = all_links

                if not all_links:
                    if self.ui_callback:
                        self.ui_callback('log', {
                            'type': 'error',
                            'message': '❌ No links found on start page'
                        })
                    return []
                
                # Show links in log window
                if self.ui_callback:
                    self.ui_callback('log', {
                        'type': 'success',
                        'message': f'📊 Found {len(all_links)} total links'
                    })
                    
                    # Send links to UI for display - This will trigger the links dialog
                    self.ui_callback('links_found', {
                        'links': all_links,
                        'count': 10  # Default
                    })
                
                # Wait for user input from the links dialog
                try:
                    # Get answer from queue using async method
                    choice = await self._get_answer_from_queue()
                    eval_count = int(choice)
                except (ValueError, asyncio.TimeoutError):
                    eval_count = min(10, len(all_links))
                
                # Ensure bounds
                eval_count = max(1, eval_count)
                eval_count = min(eval_count, len(all_links))
                
                links_to_evaluate = all_links[:eval_count]
            
                if self.ui_callback:
                    self.ui_callback('log', {
                        'type': 'info',
                        'message': f'🤔 Will evaluate {len(links_to_evaluate)} links...'
                    })
                
                # Step 3: Evaluate links
                for i, link in enumerate(links_to_evaluate):
                    if self.stop_flag:
                        break
                    
                    # Update progress
                    if self.ui_callback:
                        self.ui_callback('progress', {
                            'stage': "Evaluating links",
                            'current': i + 1,
                            'total': len(links_to_evaluate),
                            'message': f"Evaluating: {link['text'][:40]}..."
                        })
                    
                    # Send thinking log to UI
                    if self.ui_callback:
                        self.ui_callback('log', {
                            'type': 'thinking',
                            'message': f'🤔 Evaluating: {link["text"][:50]}...',
                            'details': link['url']
                        })
                    
                    if link['url'] in self.visited_urls:
                        continue
                    
                    # Quick heuristic filter
                    text_lower = link['text'].lower()
                    if (len(text_lower) < 10 or 
                        any(x in text_lower for x in ['login', 'sign up', 'register', 'password'])):
                        if self.ui_callback:
                            self.ui_callback('log', {
                                'type': 'decision',
                                'message': f'✗ Skipping (looks like login/UI element)'
                            })
                        continue
                    
                    # Get AI decision with country filter
                    # Import the evaluator method
                    decision = await self.evaluator.evaluate_link_from_title(topic, link['text'], link['url'], countries)
                    
                    if decision.get('should_click', False):
                        confidence = decision.get('confidence', 0)
                        if confidence > 0.3:
                            promising_links.append({
                                'url': link['url'],
                                'text': link['text'],
                                'confidence': confidence,
                                'reason': decision.get('reason', '')
                            })
                            
                            if self.ui_callback:
                                self.ui_callback('log', {
                                    'type': 'decision',
                                    'message': f'✓ Promising (confidence: {confidence:.2f})'
                                })
                        else:
                            if self.ui_callback:
                                self.ui_callback('log', {
                                    'type': 'decision',
                                    'message': f'✗ Low confidence: {confidence:.2f}'
                                })
                    else:
                        if self.ui_callback:
                            self.ui_callback('log', {
                                'type': 'decision',
                                'message': f'✗ Not relevant: {decision.get("reason", "")[:50]}...'
                            })
                
                if not promising_links:
                    if self.ui_callback:
                        self.ui_callback('log', {
                            'type': 'warning',
                            'message': '⚠️ No promising links found!'
                        })
                    return []
                
                # Sort by confidence
                promising_links.sort(key=lambda x: x['confidence'], reverse=True)
                
                if self.ui_callback:
                    self.ui_callback('log', {
                        'type': 'success',
                        'message': f'🎯 Found {len(promising_links)} promising links!'
                    })
                    
                    # Show top promising links
                    self.ui_callback('log', {
                        'type': 'info',
                        'message': '🏆 Top promising links:'
                    })
                    
                    for i, link in enumerate(promising_links[:5], 1):
                        self.ui_callback('log', {
                            'type': 'link',
                            'message': f'  {i}. [{link["confidence"]:.2f}] {link["text"][:50]}...',
                            'details': link['reason'][:100] if link['reason'] else ''
                        })
                
                # Step 4: Ask user how many to visit
                if self.stop_flag:
                    return []
                
                if len(promising_links) > 1:
                    # Create options for visit count
                    max_options = min(10, len(promising_links))
                    options = [str(i) for i in range(1, max_options + 1)]
                    
                    visit_choice = await self.ask_user_choice(
                        question=f"Found {len(promising_links)} promising links. How many should I visit? (1-{len(promising_links)})",
                        options=options,
                        default="3"
                    )
                    
                    try:
                        visit_count = int(visit_choice)
                    except ValueError:
                        visit_count = min(3, len(promising_links))
                    
                    # Ensure bounds
                    visit_count = max(1, visit_count)
                    visit_count = min(visit_count, len(promising_links))
                else:
                    visit_count = 1
                
                links_to_visit = promising_links[:visit_count]
                
                if self.ui_callback:
                    self.ui_callback('log', {
                        'type': 'info',
                        'message': f'📋 Will visit top {len(links_to_visit)} most promising links'
                    })
                
                # Step 5: Visit and analyze links
                for i, link_info in enumerate(links_to_visit):
                    if self.stop_flag:
                        break
                    
                    url = link_info['url']
                    
                    if self.ui_callback:
                        self.ui_callback('progress', {
                            'stage': "Analyzing pages",
                            'current': i + 1,
                            'total': len(links_to_visit),
                            'message': f"Analyzing: {url[:50]}..."
                        })
                    
                    if self.ui_callback:
                        self.ui_callback('log', {
                            'type': 'action',
                            'message': f'🔗 [{i+1}/{len(links_to_visit)}] Visiting: {link_info["text"][:50]}...'
                        })
                    
                    try:
                        new_page = await context.new_page()
                        
                        # Use new navigation method with cookie handling
                        link_success = await self._navigate_and_handle_cookies(new_page, url)
                        if not link_success:
                            if self.ui_callback:
                                self.ui_callback('log', {
                                    'type': 'warning',
                                    'message': f'⚠️ Failed to load link: {url[:50]}...'
                                })
                            await new_page.close()
                            continue
                        
                        # Extract content
                        content = await new_page.content()
                        
                        # Extract title
                        title = "Unknown"
                        try:
                            title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE)
                            if title_match:
                                import html
                                title = html.unescape(title_match.group(1).strip())[:100]
                        except:
                            title = link_info['text'][:80] + "..."
                        
                        if self.ui_callback:
                            self.ui_callback('log', {
                                'type': 'thinking',
                                'message': f'📖 Reading: {title[:60]}...',
                                'details': f'URL: {url}'
                            })
                        
                        # Evaluate page
                        evaluation = await self.evaluator.evaluate_after_click(
                            topic, url, title, content , content
                        )
                        
                        self.stats["links_clicked"] += 1
                        self.stats["pages_analyzed"] += 1
                        
                        if evaluation.get('is_relevant', False) and evaluation.get('relevance_score', 0) > 0.4:
                            self.stats["relevant_found"] += 1
                            
                            # Create result
                            from agent import ResearchResult
                            result = ResearchResult(
                                url=url,
                                title=title,
                                relevance_score=evaluation.get('relevance_score', 0.5),
                                summary=evaluation.get('summary', 'No summary available'),
                                relevance_reason=evaluation.get('reason', ''),
                                key_points=evaluation.get('key_points', [])
                            )
                            
                            results.append(result)
                            
                            if self.ui_callback:
                                self.ui_callback('log', {
                                    'type': 'success',
                                    'message': f'✅ RELEVANT! Score: {result.relevance_score:.2f}'
                                })
                                self.ui_callback('result', result.model_dump())
                        
                        self.visited_urls.add(url)
                        await new_page.close()
                        
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        if self.ui_callback:
                            self.ui_callback('log', {
                                'type': 'error',
                                'message': f'❌ Error inspecting {url[:50]}...',
                                'details': str(e)[:100]
                            })
                        continue
                
                if page:
                    await page.close()
                
            except Exception as e:
                if self.ui_callback:
                    self.ui_callback('log', {
                        'type': 'error',
                        'message': f'❌ Research error: {str(e)}'
                    })
                import traceback
                traceback.print_exc()
                return []
            
            finally:
                await browser.close()
        
        # Return results
        return results

    async def _navigate_with_retry(self, page, url: str, max_retries: int = 3) -> bool:
        """Navigate to URL with retry logic for network errors"""
        for attempt in range(max_retries):
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                return True
            except Exception as e:
                error_msg = str(e)
                if self.ui_callback:
                    self.ui_callback('log', {
                        'type': 'warning',
                        'message': f'⚠️ Navigation attempt {attempt + 1} failed: {error_msg[:50]}...'
                    })
                
                if attempt == max_retries - 1:
                    if self.ui_callback:
                        self.ui_callback('log', {
                            'type': 'error',
                            'message': f'❌ Failed to load {url} after {max_retries} attempts'
                        })
                    return False
                
                # Wait longer between retries
                await asyncio.sleep(3 * (attempt + 1))
        
        return False

    async def _setup_page_network_handlers(self, page):
        """Setup network event handlers to handle network issues"""
        
        async def handle_request(request):
            # You can log or modify requests here
            pass
        
        async def handle_response(response):
            if response.status >= 400:
                print(f"Response error: {response.status} for {response.url}")
        
        async def handle_request_failed(request):
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'warning',
                    'message': f'Request failed: {request.url[:50]}...'
                })
        
        page.on("request", handle_request)
        page.on("response", handle_response)
        page.on("requestfailed", handle_request_failed)

    async def _robust_navigate(self, page, url: str, max_retries: int = 3) -> bool:
        """Robust navigation with multiple fallback strategies"""
        strategies = [
            {'wait_until': 'domcontentloaded', 'timeout': 60000},
            {'wait_until': 'load', 'timeout': 60000},
            {'wait_until': 'networkidle', 'timeout': 60000},
            {'wait_until': 'commit', 'timeout': 60000}
        ]
        
        for attempt in range(max_retries):
            for strategy_idx, strategy in enumerate(strategies):
                try:
                    if self.ui_callback:
                        self.ui_callback('log', {
                            'type': 'info',
                            'message': f'🌐 Attempt {attempt + 1}, strategy {strategy_idx + 1}: Loading {url[:50]}...'
                        })
                    
                    await page.goto(
                        url, 
                        wait_until=strategy['wait_until'], 
                        timeout=strategy['timeout']
                    )
                    
                    # Check if page loaded successfully
                    try:
                        await page.wait_for_load_state('load', timeout=5000)
                    except:
                        pass
                    
                    if self.ui_callback:
                        self.ui_callback('log', {
                            'type': 'success',
                            'message': f'✅ Page loaded successfully'
                        })
                    
                    return True
                    
                except Exception as e:
                    error_msg = str(e)
                    if self.ui_callback:
                        self.ui_callback('log', {
                            'type': 'warning',
                            'message': f'⚠️ Navigation failed: {error_msg[:50]}...'
                        })
                    
                    # Try reload if it's a network error
                    if 'net::' in error_msg or 'ERR_' in error_msg:
                        try:
                            await page.reload(wait_until='domcontentloaded', timeout=30000)
                            return True
                        except:
                            pass
                    
                    # Wait before retrying
                    await asyncio.sleep(2)
        
        return False

    async def _extract_links_simple(self, page) -> List[Dict]:
        """Simple link extraction that matches the parent class logic"""
        try:
            # Use the SAME logic as the parent class
            if 'news.ycombinator.com' in str(page.url):
                # Hacker News specific extraction
                links = await page.evaluate("""
                    () => {
                        const links = [];
                        // Get main story links (Hacker News specific)
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
                # Generic extraction (same as parent's fallback)
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
                        'text': link['text'][:200],  # Limit text length
                        'url': clean_url
                    })
            
            return unique_links
            
        except Exception as e:
            return []

    async def _extract_links_fallback(self, page, start_url: str) -> List[Dict]:
        """Fallback method - INCLUSIVE VERSION"""
        try:
            # Get links using the simple method
            links = await self._extract_links_simple(page)
            
            # Apply ignore terms filtering
            if hasattr(self, 'ignore_terms') and self.ignore_terms:
                filtered_links = []
                for link in links:
                    text_lower = link['text'].lower()
                    url_lower = link['url'].lower()
                    
                    should_skip = False
                    for term in self.ignore_terms:
                        if term in text_lower or term in url_lower:
                            should_skip = True
                            break
                    
                    if not should_skip:
                        filtered_links.append(link)
                
                links = filtered_links
            
            return links
            
        except Exception as e:
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'error',
                    'message': f'Fallback extraction error: {str(e)[:100]}'
                })
            return [] 
    
    async def _extract_links_smart0(self, page, topic: str, start_url: str) -> List[Dict]:
        """Smart link extraction - matches the parent class logic"""
        import re
        
        # Log action
        if self.ui_callback:
            self.ui_callback('log', {
                'type': 'action',
                'message': '🤔 Analyzing page structure...'
            })
        
        try:
            # First, analyze the page structure (simplified version)
            structure_info = await page.evaluate("""
                () => {
                    return {
                        title: document.title || "",
                        url: window.location.href,
                        isHackerNews: window.location.hostname.includes('news.ycombinator.com'),
                        hasTable: document.querySelector('table') !== null,
                        hasLists: document.querySelector('ul, ol') !== null,
                        totalLinks: document.querySelectorAll('a[href]').length,
                        visibleLinks: 0
                    };
                }
            """)
            
            # Different strategies for different sites
            if structure_info.get('isHackerNews', False):
                # Hacker News specific extraction
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
                # Generic smart extraction
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
                        'text': link['text'][:150],
                        'url': clean_url
                    })
            
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'success',
                    'message': f'📊 Found {len(unique_links)} links'
                })
            
            return unique_links
            
        except Exception as e:
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'error',
                    'message': f'❌ Error extracting links: {str(e)[:100]}'
                })
            return []

        """Comprehensive link extraction to get ALL visible links"""
        try:
            # Get ALL visible links from the page
            links = await page.evaluate("""
                () => {
                    const links = [];
                    const allAnchors = document.querySelectorAll('a[href]');
                    
                    for (const anchor of allAnchors) {
                        const text = anchor.textContent.trim();
                        const url = anchor.href;
                        
                        // Basic filtering - very inclusive
                        if (!text || !url || text.length < 3) {
                            continue;
                        }
                        
                        // Must be HTTP/HTTPS link
                        if (!url.startsWith('http')) {
                            continue;
                        }
                        
                        // Skip anchor links
                        if (url.includes('#') && !url.includes('#!')) {
                            continue;
                        }
                        
                        // Get link visibility
                        const style = window.getComputedStyle(anchor);
                        if (style.display === 'none' || style.visibility === 'hidden') {
                            continue;
                        }
                        
                        // Skip extremely common non-content links
                        const lowerText = text.toLowerCase();
                        const skipExactMatches = [
                            'login', 'sign in', 'register', 'sign up', 'log in',
                            'more', 'next', 'previous', 'back', 'home',
                            'about', 'contact', 'privacy', 'terms', 'cookies',
                            'policy', 'advertise', 'advertisement', 'ads',
                            'subscribe', 'follow', 'share', 'twitter', 'facebook',
                            'linkedin', 'instagram', 'youtube', 'rss', 'feed'
                        ];
                        
                        // Skip if exact match
                        if (skipExactMatches.includes(lowerText)) {
                            continue;
                        }
                        
                        // Skip if contains certain patterns (but be less restrictive)
                        const skipPatterns = [
                            'privacy policy', 'terms of service', 'cookie policy',
                            'all rights reserved', '©', 'copyright'
                        ];
                        
                        let skip = false;
                        for (const pattern of skipPatterns) {
                            if (lowerText.includes(pattern)) {
                                skip = true;
                                break;
                            }
                        }
                        
                        if (skip) {
                            continue;
                        }
                        
                        // Check if link is interactive (not just decorative)
                        const rect = anchor.getBoundingClientRect();
                        if (rect.width < 10 || rect.height < 10) {
                            continue;  // Too small, likely decorative
                        }
                        
                        links.push({
                            text: text.substring(0, 200),
                            url: url,
                            area: rect.width * rect.height  // For sorting by size
                        });
                    }
                    return links;
                }
            """)
            
            # Remove duplicates by URL
            seen = set()
            unique_links = []
            
            for link in links:
                # Clean URL
                clean_url = link['url'].split('#')[0].split('?')[0].rstrip('/')
                if clean_url not in seen and clean_url.startswith('http'):
                    seen.add(clean_url)
                    unique_links.append({
                        'text': link['text'],
                        'url': clean_url,
                        'area': link['area']
                    })
            
            # Sort by area (largest first) - larger elements are more likely to be content
            unique_links.sort(key=lambda x: x['area'], reverse=True)
            
            # Remove area from final result
            final_links = []
            for link in unique_links:
                final_links.append({
                    'text': link['text'],
                    'url': link['url']
                })
            
            return final_links
            
        except Exception as e:
            return []

    async def _extract_links_smart(self, page, topic: str, start_url: str) -> List[Dict]:
        """Use _extract_links_smart0 but add ignore terms filtering"""
        if self.ui_callback:
            self.ui_callback('log', {
                'type': 'action',
                'message': '🤔 Analyzing page to find relevant links...'
            })
        
        # Use the good extraction method
        links = await self._extract_links_smart0(page, topic, start_url)
        
        if not links:
            return []
        
        if self.ui_callback:
            self.ui_callback('log', {
                'type': 'success',
                'message': f'📊 Found {len(links)} links'
            })
        
        # Apply ignore terms filtering
        if hasattr(self, 'ignore_terms') and self.ignore_terms:
            filtered_links = []
            for link in links:
                text_lower = link['text'].lower()
                url_lower = link['url'].lower()
                
                # Check if link contains any ignore term
                should_skip = False
                for term in self.ignore_terms:
                    if term in text_lower or term in url_lower:
                        should_skip = True
                        if self.ui_callback:
                            self.ui_callback('log', {
                                'type': 'decision',
                                'message': f'✗ Skipping (contains ignore term: {term})'
                            })
                        break
                
                if not should_skip:
                    filtered_links.append(link)
            
            links = filtered_links
            
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'info',
                    'message': f'📰 After ignore filter: {len(links)} links'
                })
        
        return links 
    
    async def evaluate_with_and_logic(self, topic: str, link_text: str, url: str, countries: List[str] = None) -> Dict:
        """Evaluate link with strict AND logic (must be about topic AND countries)"""
        if not countries:
            # No country filter, use regular evaluation
            return await self.evaluate_link_from_title(topic, link_text, url)
        
        # First check if link is about any of the countries
        country_check_prompt = f"""Check if this link is about any of these countries: {', '.join(countries)}

        LINK TEXT: {link_text}
        URL: {url}
        COUNTRIES: {', '.join(countries)}

        Instructions:
        1. Analyze the link text and URL
        2. Check if it mentions or is clearly about any of the specified countries
        3. If it's not about any of these countries, return false

        Response format (JSON only):
        {{
            "is_about_countries": true/false,
            "country_found": "name of country if found, else empty",
            "confidence": 0.0-1.0,
            "reason": "Brief explanation"
        }}"""
        
        country_result = await self._ask_ai(country_check_prompt)
        
        # If not about countries, return false immediately
        if not country_result or not country_result.get('is_about_countries', False):
            return {
                "should_click": False,
                "confidence": 0.0,
                "reason": f"Not about specified countries: {', '.join(countries)}",
                "needs_inspection": False
            }
        
        # If it is about countries, then check topic relevance
        return await self.evaluate_link_from_title(topic, link_text, url)
    
    async def evaluate_link_with_countries(self, topic: str, link_text: str, url: str, countries: List[str]) -> Dict:
        """Evaluate link with strict AND logic for topic AND countries"""
        
        prompt = f"""Evaluate if this link meets ALL these criteria:
        1. Is about the topic: "{topic}"
        2. Is specifically about {', '.join(countries)} (not just mentions them in passing)

        LINK TEXT: {link_text}
        URL: {url}

        STRICT RULES:
        - The link must be PRIMARILY about BOTH the topic AND at least one of the specified countries
        - If it mentions the topic but not in relation to the countries, it's NOT relevant
        - If it mentions the countries but not about the topic, it's NOT relevant
        - Both conditions must be met

        Examples:
        - "AI development in USA" - RELEVANT (topic + country)
        - "AI development globally" - NOT RELEVANT (no country)
        - "Economy of UK" when topic is "AI" - NOT RELEVANT (wrong topic)

        Response format (JSON only):
        {{
            "should_click": true/false,
            "confidence": 0.0-1.0,
            "reason": "Brief explanation of why it meets/fails BOTH criteria",
            "topic_match": true/false,
            "country_match": true/false,
            "needs_inspection": true/false
        }}"""
        
        decision = await self._ask_ai(prompt)
        
        if not decision:
            # Conservative default
            return {
                "should_click": False,
                "confidence": 0.0,
                "reason": "AI evaluation failed - being conservative",
                "topic_match": False,
                "country_match": False,
                "needs_inspection": True
            }
        
        return decision

    async def _handle_popups_ui(self, page):
        """Handle cookies and popups"""
        try:
            # More comprehensive list of consent selectors
            consent_selectors = [
                'button:has-text("Accept")',
                'button:has-text("Agree")', 
                'button:has-text("OK")',
                'button:has-text("I agree")',
                'button:has-text("Accept all")',
                'button:has-text("Accept cookies")',
                'button:has-text("Allow all")',
                'button[aria-label*="accept"]',
                'button[aria-label*="agree"]',
                'button[data-test*="accept"]',
                'button[class*="accept"]',
                'button[class*="agree"]',
                '.accept-cookies',
                '.agree-button',
                '.consent-accept',
                '#accept',
                '#agree',
                '#accept-cookies'
            ]
            
            for selector in consent_selectors:
                try:
                    if await page.locator(selector).count() > 0:
                        await page.click(selector)
                        if self.ui_callback:
                            self.ui_callback('log', {
                                'type': 'action',
                                'message': '✅ Clicked consent button'
                            })
                        await asyncio.sleep(0.5)
                        break
                except:
                    continue
            
            # Also try to click any button that contains "cookie" and "accept/agree"
            try:
                all_buttons = await page.locator('button').all()
                for button in all_buttons:
                    try:
                        text = await button.text_content()
                        if text and ('cookie' in text.lower() or 'consent' in text.lower()) and \
                        ('accept' in text.lower() or 'agree' in text.lower() or 'ok' in text.lower()):
                            await button.click()
                            await asyncio.sleep(0.5)
                            break
                    except:
                        continue
            except:
                pass
                
        except Exception as e:
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'warning',
                    'message': f'⚠️ Popup handling issue: {str(e)[:50]}...'
                })


    async def _navigate_and_handle_cookies(self, page, url: str) -> bool:
        """Navigate to URL, wait for full load, and handle cookies"""
        try:
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'action',
                    'message': f'🌐 Navigating to: {url[:50]}...'
                })
            
            # Navigate with multiple wait states
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            
            # Wait for page to be fully interactive
            await asyncio.sleep(2)
            
            # Try to wait for network to be idle
            try:
                await page.wait_for_load_state('networkidle', timeout=10000)
            except:
                # If networkidle times out, at least wait for load
                await page.wait_for_load_state('load', timeout=5000)
            
            # Handle cookies and popups
            await self._handle_cookies_robust(page)
            
            # Additional wait for dynamic content
            await asyncio.sleep(1)
            
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'success',
                    'message': f'✅ Page loaded successfully'
                })
            
            return True
            
        except Exception as e:
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'error',
                    'message': f'❌ Navigation failed: {str(e)[:50]}...'
                })
            return False

    async def _handle_cookies_robust(self, page):
        """Robust cookie and popup handler with click attempts"""
        try:
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'action',
                    'message': '🍪 Checking for cookie consent...'
                })
            
            # Comprehensive list of consent selectors
            consent_selectors = [
                # Text-based selectors
                'button:has-text("Accept")',
                'button:has-text("Agree")',
                'button:has-text("I agree")',
                'button:has-text("Accept all")',
                'button:has-text("Accept all cookies")',
                'button:has-text("Allow all")',
                'button:has-text("Allow cookies")',
                'button:has-text("OK")',
                'button:has-text("Got it")',
                'button:has-text("Continue")',
                'button:has-text("Close")',
                'button:has-text("Dismiss")',
                
                # ARIA labels
                'button[aria-label*="accept"]',
                'button[aria-label*="agree"]',
                'button[aria-label*="dismiss"]',
                'button[aria-label*="close"]',
                
                # Data attributes
                'button[data-test*="accept"]',
                'button[data-test*="agree"]',
                'button[data-testid*="accept"]',
                
                # Class-based selectors
                'button[class*="accept"]',
                'button[class*="agree"]',
                'button[class*="consent"]',
                'button[class*="cookie"]',
                'button[class*="close"]',
                'button[class*="dismiss"]',
                
                # CSS classes
                '.accept-cookies',
                '.agree-button',
                '.consent-accept',
                '.cookie-accept',
                '.btn-accept',
                '.btn-agree',
                '.close-button',
                '.dismiss-button',
                
                # IDs
                '#accept',
                '#agree',
                '#accept-cookies',
                '#close-cookies'
            ]
            
            clicked = False
            
            # Try each selector
            for selector in consent_selectors:
                try:
                    # Check if element exists and is visible
                    elements = await page.locator(selector).count()
                    if elements > 0:
                        element = page.locator(selector).first
                        is_visible = await element.is_visible()
                        
                        if is_visible:
                            await element.click()
                            if self.ui_callback:
                                self.ui_callback('log', {
                                    'type': 'action',
                                    'message': f'✅ Clicked consent button: {selector[:30]}...'
                                })
                            clicked = True
                            await asyncio.sleep(0.5)
                            break
                except Exception as e:
                    continue
            
            # Also try to find any overlay/dialog close buttons
            if not clicked:
                try:
                    # Look for common overlay patterns
                    overlay_selectors = [
                        '[role="dialog"] button',
                        '.modal button',
                        '.popup button',
                        '.overlay button',
                        '.cookie-banner button',
                        '.consent-banner button'
                    ]
                    
                    for selector in overlay_selectors:
                        try:
                            elements = await page.locator(selector).count()
                            if elements > 0:
                                # Try to find a close/accept button in the overlay
                                for i in range(min(elements, 5)):  # Check first 5 elements
                                    element = page.locator(selector).nth(i)
                                    text = await element.text_content()
                                    if text and any(word in text.lower() for word in ['accept', 'agree', 'ok', 'close', 'got it']):
                                        await element.click()
                                        if self.ui_callback:
                                            self.ui_callback('log', {
                                                'type': 'action',
                                                'message': f'✅ Clicked overlay button: {text[:20]}...'
                                            })
                                        clicked = True
                                        await asyncio.sleep(0.5)
                                        break
                                if clicked:
                                    break
                        except:
                            continue
                except:
                    pass
            
            if not clicked and self.ui_callback:
                self.ui_callback('log', {
                    'type': 'info',
                    'message': 'ℹ️ No cookie consent found (or already accepted)'
                })
                    
        except Exception as e:
            if self.ui_callback:
                self.ui_callback('log', {
                    'type': 'warning',
                    'message': f'⚠️ Cookie handling issue: {str(e)[:50]}...'
                })


class ResearchAgentApp:
    """Main Windows Application with Complete UI Control"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI Research Agent v3.0 - Complete UI Control")
        self.root.geometry("1400x800")
        
        # Variables
        self.session_active = False
        self.current_agent = None
        self.user_choice_queue = queue.Queue()
        self.results = []
        
        # Modern colors
        self.colors = {
            'bg': '#1a1a1a',
            'card': '#2d2d30',
            'text': '#ffffff',
            'accent': '#007acc',
            'success': '#107c10',
            'warning': '#ffb900',
            'error': '#dc3545',
            'border': '#3e3e42'
        }
        
        # Configure root
        self.root.configure(bg=self.colors['bg'])
        #self.root.option_add('*Font', 'TkDefaultFont')        
        # Build UI
        self.build_ui()
        
        # Check Ollama
        self.check_ollama()
    
    def show_links_dialog(self, links: List[Dict], default_count: int = 10) -> int:
        """Show links in a scrollable dialog and get evaluation count"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Links Found - Select Evaluation Count")
        dialog.geometry("900x600")
        dialog.configure(bg=self.colors['card'])
        dialog.transient(self.root)
        dialog.grab_set()  # Make it modal
        
        # Title
        title_label = tk.Label(dialog, 
                            text=f"Found {len(links)} links. How many should I evaluate?",
                            font=('Arial', 12, 'bold'),
                            bg=self.colors['card'], fg=self.colors['text'])
        title_label.pack(pady=10, padx=20)
        
        # Create a frame for the list with scrollbars
        list_frame = tk.Frame(dialog, bg=self.colors['card'])
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create text widget with scrollbars
        text_frame = tk.Frame(list_frame, bg=self.colors['card'])
        text_frame.pack(fill='both', expand=True)
        
        # Text widget for displaying links
        text_widget = tk.Text(text_frame, 
                            bg='#3c3c3c', fg='white',
                            font=('Consolas', 9),
                            wrap='none')  # No wrap for better readability
        text_widget.pack(side='left', fill='both', expand=True)
        
        # Vertical scrollbar
        v_scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
        v_scrollbar.pack(side='right', fill='y')
        text_widget.config(yscrollcommand=v_scrollbar.set)
        
        # Horizontal scrollbar
        h_scrollbar = tk.Scrollbar(list_frame, orient='horizontal', command=text_widget.xview)
        h_scrollbar.pack(fill='x')
        text_widget.config(xscrollcommand=h_scrollbar.set)
        
        # Insert links into text widget
        for i, link in enumerate(links, 1):
            # Create formatted line
            line = f"{i:3}. {link['text'][:80]}"
            # Pad with spaces
            line = line.ljust(100)
            line += f" | {link['url'][:80]}\n"
            
            text_widget.insert('end', line)
        
        # Make text widget read-only
        text_widget.config(state='disabled')
        
        # Control frame for count input
        control_frame = tk.Frame(dialog, bg=self.colors['card'])
        control_frame.pack(pady=10, padx=20)
        
        tk.Label(control_frame, text="Evaluate first:", 
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0, 10))
        
        # Create variable for count
        count_var = tk.StringVar(value=str(default_count))
        
        # Entry for count
        count_entry = tk.Entry(control_frame, 
                            textvariable=count_var,
                            font=('Arial', 10),
                            bg='#3c3c3c', fg='white',
                            width=10)
        count_entry.pack(side='left', padx=(0, 10))
        count_entry.select_range(0, tk.END)
        count_entry.focus_set()
        
        tk.Label(control_frame, text=f"links (1-{len(links)})", 
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left')
        
        # Result variable
        result = {"count": default_count}
        
        def submit():
            try:
                count = int(count_var.get())
                # Ensure bounds
                if count < 1:
                    count = 1
                elif count > len(links):
                    count = len(links)
                result["count"] = count
            except ValueError:
                result["count"] = default_count
            dialog.destroy()
        
        # Submit on Enter
        count_entry.bind('<Return>', lambda e: submit())
        
        # Button frame
        button_frame = tk.Frame(dialog, bg=self.colors['card'])
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="OK", 
                bg=self.colors['accent'], fg='white',
                command=submit, padx=20).pack()
        
        # Center dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Wait for dialog
        dialog.wait_window(dialog)
        
        return result["count"]
    
    def build_ui(self):
        """Build the complete UI"""
        # Configure grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Header
        header = tk.Frame(self.root, bg=self.colors['bg'], height=80)
        header.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
        header.grid_propagate(False)
        
        tk.Label(header, text="🤖 AI RESEARCH AGENT", 
                font=('Arial', 24, 'bold'),  # Changed from Arial to Arial
                bg=self.colors['bg'], fg=self.colors['text']).pack(side='left', padx=20)

        self.status_label = tk.Label(header, text="🔴 Offline", 
                                    font=('Arial', 10),  # Changed from Arial
                                    bg=self.colors['bg'], fg=self.colors['warning'])
        self.status_label.pack(side='right', padx=20)
        
        # Left panel - Controls
        left_panel = tk.Frame(self.root, bg=self.colors['card'], width=400)
        left_panel.grid(row=1, column=0, sticky='nsew', padx=(10, 5), pady=5)
        left_panel.grid_propagate(False)
        
        # Right panel - Logs & Results
        right_panel = tk.Frame(self.root, bg=self.colors['bg'])
        right_panel.grid(row=1, column=1, sticky='nsew', padx=(5, 10), pady=5)
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Build panels
        self.build_control_panel(left_panel)
        self.build_log_panel(right_panel)
        self.build_results_panel(right_panel)
        
        # User choice dialog (hidden initially)
        self.build_choice_dialog()
    
    def build_control_panel(self, parent):
        """Build control panel"""
        # Control frame
        control_frame = tk.LabelFrame(parent, text="🔧 Research Controls", 
                             font=('Arial', 12, 'bold'),  # Changed
                             bg=self.colors['card'], fg=self.colors['text'],
                             relief='flat', borderwidth=2)
        control_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Topic
        tk.Label(control_frame, text="Research Topic:", 
            bg=self.colors['card'], fg=self.colors['text'],
            font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5), padx=10)  # Changed
        
        self.topic_entry = tk.Entry(control_frame, font=('Arial', 10),
                                   bg='#3c3c3c', fg='white', insertbackground='white')
        self.topic_entry.pack(fill='x', padx=10, pady=(0, 15))
        self.topic_entry.insert(0, "Artificial Intelligence")
        
        # Start URL
        tk.Label(control_frame, text="Start URL:", 
            bg=self.colors['card'], fg=self.colors['text'],
            font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5), padx=10)  # Changed
        
        self.url_entry = tk.Entry(control_frame, font=('Arial', 10),
                                 bg='#3c3c3c', fg='white', insertbackground='white')
        self.url_entry.pack(fill='x', padx=10, pady=(0, 15))
        self.url_entry.insert(0, "https://news.ycombinator.com")
        

        tk.Label(control_frame, text="Countries (comma-separated, optional):", 
            bg=self.colors['card'], fg=self.colors['text'],
            font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5), padx=10)
    
        self.countries_entry = tk.Entry(control_frame, font=('Arial', 10),
                                    bg='#3c3c3c', fg='white', insertbackground='white')
        self.countries_entry.pack(fill='x', padx=10, pady=(0, 15))
        self.countries_entry.insert(0, "USA,UK,Canada")  # Example default
        # Instructions notebook
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill='both', expand=True, padx=10, pady=(0, 15))
        
        # What to look for
        look_frame = tk.Frame(notebook, bg=self.colors['card'])
        self.look_text = scrolledtext.ScrolledText(look_frame, height=6,
                                                  bg='#3c3c3c', fg='white',
                                                  insertbackground='white',
                                                  font=('Arial', 9))
        self.look_text.pack(fill='both', expand=True, padx=2, pady=2)
        self.look_text.insert('1.0', "Technical articles, research papers, tutorials, case studies")
        notebook.add(look_frame, text="What to Look For")
        
        # What to ignore
        ignore_frame = tk.Frame(notebook, bg=self.colors['card'])
        self.ignore_text = scrolledtext.ScrolledText(ignore_frame, height=6,
                                                    bg='#3c3c3c', fg='white',
                                                    insertbackground='white',
                                                    font=('Arial', 9))
        self.ignore_text.pack(fill='both', expand=True, padx=2, pady=2)
        self.ignore_text.insert('1.0', "Ads, sidebars, login pages, unrelated topics, job postings")
        notebook.add(ignore_frame, text="What to Ignore")
        
        # Action buttons
        button_frame = tk.Frame(control_frame, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.start_btn = tk.Button(button_frame, text="▶ START RESEARCH", 
                                  font=('Arial', 11, 'bold'),
                                  bg=self.colors['success'], fg='white',
                                  command=self.start_research,
                                  padx=20, pady=10)
        self.start_btn.pack(side='left', padx=(0, 10))
        
        self.stop_btn = tk.Button(button_frame, text="⏹ STOP", 
                                 font=('Arial', 11, 'bold'),
                                 bg=self.colors['error'], fg='white',
                                 command=self.stop_research,
                                 state='disabled',
                                 padx=20, pady=10)
        self.stop_btn.pack(side='left')
    
    def build_log_panel(self, parent):
        """Build log panel"""
        log_frame = tk.LabelFrame(parent, text="🧠 Thinking Log", 
                         font=('Arial', 12, 'bold'),  # Changed
                         bg=self.colors['card'], fg=self.colors['text'],
                         relief='flat', borderwidth=2)
        log_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=(0, 5))
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        
        # Log controls
        control_frame = tk.Frame(log_frame, bg=self.colors['card'])
        control_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)
        
        tk.Button(control_frame, text="Clear Logs", 
                 bg=self.colors['border'], fg='white',
                 command=self.clear_logs).pack(side='left')
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Auto-scroll", 
                      variable=self.auto_scroll_var,
                      bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card']).pack(side='left', padx=(20, 0))
        self.detailed_logging_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="Detailed Logging", 
                      variable=self.detailed_logging_var,
                      bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card']).pack(side='left', padx=(20, 0))
                
        
        
        # Log display
        text_frame = tk.Frame(log_frame, bg=self.colors['card'])
        text_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=(0, 10))
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        self.log_text = tk.Text(text_frame, bg='#1e1e1e', fg='white',
                               insertbackground='white', wrap='word',
                               font=('Consolas', 9), state='disabled')
        
        scrollbar = tk.Scrollbar(text_frame)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        
        self.log_text.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Configure tags
        for tag, color in [
            ('info', '#569cd6'),
            ('success', '#4ec9b0'),
            ('warning', '#ffd700'),
            ('error', '#f48771'),
            ('action', '#9cdcfe'),
            ('decision', '#c586c0')
        ]:
            self.log_text.tag_config(tag, foreground=color)
    
    def build_results_panel(self, parent):
        """Build results panel"""
        # In build_results_panel():
        results_frame = tk.LabelFrame(parent, text="📊 Results", 
                             font=('Arial', 12, 'bold'),  # Changed
                             bg=self.colors['card'], fg=self.colors['text'],
                             relief='flat', borderwidth=2)
        results_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=(0, 5))
        results_frame.grid_rowconfigure(1, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Results controls
        control_frame = tk.Frame(results_frame, bg=self.colors['card'])
        control_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)
        
        self.results_count = tk.Label(control_frame, text="0 results", 
                                     bg=self.colors['card'], fg=self.colors['text'])
        self.results_count.pack(side='left')
        
        tk.Button(control_frame, text="Export", 
                 bg=self.colors['border'], fg='white',
                 command=self.export_results).pack(side='right', padx=(0, 5))
        
        tk.Button(control_frame, text="Clear", 
                 bg=self.colors['border'], fg='white',
                 command=self.clear_results).pack(side='right')
        
        # Results list
        list_frame = tk.Frame(results_frame, bg=self.colors['card'])
        list_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=(0, 10))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        self.results_listbox = tk.Listbox(list_frame, bg='#3c3c3c', fg='white',
                                         selectbackground=self.colors['accent'],
                                         font=('Arial', 9))
        
        scrollbar = tk.Scrollbar(list_frame)
        self.results_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_listbox.yview)
        
        self.results_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Bind double-click to show details
        self.results_listbox.bind('<Double-Button-1>', self.show_result_details)
   
    def build_choice_dialog(self):
        """Build hidden choice dialog"""
        self.choice_window = tk.Toplevel(self.root)
        self.choice_window.title("User Choice Required")
        self.choice_window.geometry("500x300")
        self.choice_window.configure(bg=self.colors['card'])
        self.choice_window.withdraw()  # Hide initially
        
        # Question label
        self.choice_question = tk.Label(self.choice_window, 
                                       font=('Arial', 11, 'bold'),
                                       bg=self.colors['card'], fg=self.colors['text'],
                                       wraplength=450)
        self.choice_question.pack(pady=20, padx=20, anchor='w')
        
        # Options frame - store as attribute
        self.options_frame = tk.Frame(self.choice_window, bg=self.colors['card'])
        self.options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.choice_var = tk.StringVar()
        self.choice_buttons = []
        
        # Custom entry for numbers
        self.custom_frame = tk.Frame(self.choice_window, bg=self.colors['card'])
        self.custom_entry = tk.Entry(self.custom_frame, font=('Arial', 10),
                                    bg='#3c3c3c', fg='white', width=10)
        self.custom_entry.pack(side='left', padx=(0, 10))
        tk.Button(self.custom_frame, text="Submit", 
                 bg=self.colors['accent'], fg='white',
                 command=self.submit_custom_choice).pack(side='left')
        self.custom_entry.bind('<Return>', lambda e: self.submit_custom_choice())
        # OK button
        button_frame = tk.Frame(self.choice_window, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Button(button_frame, text="OK", 
                 bg=self.colors['accent'], fg='white',
                 command=self.submit_choice,
                 padx=20).pack()    

    def ui_callback(self, event_type: str, data: Any):
        """Callback for agent to communicate with UI"""
        if event_type == 'log':
            # Check if this is a "decision" type log about ignoring
            if data.get('type') == 'decision' and 'Skipping (contains ignore term:' in data.get('message', ''):
                self.add_log('info', data['message'], data.get('details', ''))  # Show as info instead
            else:
                self.add_log(data['type'], data['message'], data.get('details', ''))
        
        elif event_type == 'progress':
            self.update_progress(data['stage'], data['current'], data['total'], data['message'])
            
        elif event_type == 'user_choice':
            # Show a simple number input dialog
            question = data['question']
            default = data.get('default', '10')
            max_val = 50
            
            # Extract max value from question
            import re
            match = re.search(r'1-(\d+)', question)
            if match:
                max_val = int(match.group(1))
            
            # Show dialog and get result
            result = self.ask_number_dialog(
                "User Choice Required",
                question,
                int(default) if default.isdigit() else 10,
                min_val=1,
                max_val=max_val
            )
            
            # Send response back to agent
            if hasattr(self, 'current_agent') and self.current_agent:
                self.current_agent.answers.put(str(result))
                
        elif event_type == 'links_found':
            # New event type: agent found links
            links = data['links']
            count = data.get('count', 10)
            
            # Show links dialog synchronously
            eval_count = self.show_links_dialog(links, count)
            
            # Send result back
            if hasattr(self, 'current_agent') and self.current_agent:
                self.current_agent.answers.put(str(eval_count))
        
        elif event_type == 'result':
            self.add_result(data)

    def add_log(self, log_type: str, message: str, details: str = ""):
        """Add log entry"""
        if not self.detailed_logging_var.get():
            # Skip these types when not in detailed mode
            skip_types = ['thinking', 'progress']
            if log_type in skip_types and 'Evaluating' not in message and 'Analyzing' not in message:
                return

        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.log_text.config(state='normal')
        
        # Insert timestamp
        self.log_text.insert('end', f"[{timestamp}] ", 'info')
        
        # Insert type and message
        self.log_text.insert('end', f"{log_type.upper()}: {message}\n", log_type)
        
        # Insert details if present
        if details:
            self.log_text.insert('end', f"    {details}\n", 'info')
        
        # Auto-scroll
        if self.auto_scroll_var.get():
            self.log_text.see('end')
        
        self.log_text.config(state='disabled')
    
    def clear_logs(self):
        """Clear all logs"""
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.config(state='disabled')
    
    def add_result(self, result: dict):
        """Add result to list"""
        self.results.append(result)
        
        # Update listbox
        title = result.get('title', 'Unknown')[:50]
        score = result.get('relevance_score', 0)
        text = f"[{score:.2f}] {title}"
        self.results_listbox.insert('end', text)
        
        # Update count
        self.results_count.config(text=f"{len(self.results)} results")
    
    def clear_results(self):
        """Clear all results"""
        self.results.clear()
        self.results_listbox.delete(0, 'end')
        self.results_count.config(text="0 results")
    
    def export_results(self):
        """Export results to file"""
        if not self.results:
            messagebox.showinfo("No Results", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
                self.add_log('success', f'Results exported to {filename}')
            except Exception as e:
                messagebox.showerror("Export Error", str(e))
    
    def show_result_details(self, event):
        """Show result details on double-click"""
        selection = self.results_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index < len(self.results):
            result = self.results[index]
            
            # Create details window
            details = tk.Toplevel(self.root)
            details.title("Result Details")
            details.geometry("600x500")
            details.configure(bg=self.colors['card'])
            
            # Title
            tk.Label(details, text=result.get('title', 'Unknown'), 
                    font=('Arial', 14, 'bold'),
                    bg=self.colors['card'], fg=self.colors['text'],
                    wraplength=550).pack(pady=10, padx=20, anchor='w')
            
            # Score
            score = result.get('relevance_score', 0)
            color = '#4ec9b0' if score > 0.7 else '#ffd700' if score > 0.4 else '#f48771'
            tk.Label(details, text=f"Relevance Score: {score:.2f}", 
                    font=('Arial', 12),
                    bg=self.colors['card'], fg=color).pack(pady=5, padx=20, anchor='w')
            
            # URL
            url_frame = tk.Frame(details, bg=self.colors['card'])
            url_frame.pack(fill='x', padx=20, pady=5)
            tk.Label(url_frame, text="URL:", 
                    font=('Arial', 10, 'bold'),
                    bg=self.colors['card'], fg=self.colors['text']).pack(side='left')
            url_text = tk.Text(url_frame, height=2, font=('Arial', 9),
                              bg='#3c3c3c', fg='white', wrap='word')
            url_text.insert('1.0', result.get('url', ''))
            url_text.config(state='disabled')
            url_text.pack(side='left', fill='x', expand=True, padx=(10, 0))
            
            # Summary
            tk.Label(details, text="Summary:", 
                    font=('Arial', 10, 'bold'),
                    bg=self.colors['card'], fg=self.colors['text']).pack(pady=(10, 5), padx=20, anchor='w')
            
            summary_text = scrolledtext.ScrolledText(details, height=8,
                                                    bg='#3c3c3c', fg='white',
                                                    font=('Arial', 9),
                                                    wrap='word')
            summary_text.pack(fill='both', expand=True, padx=20, pady=(0, 10))
            summary_text.insert('1.0', result.get('summary', 'No summary available'))
            summary_text.config(state='disabled')
    
    def show_user_choice_dialog(self, question: str, options: List[str], default=None):
        """Show a simple user choice dialog - returns choice or None if cancelled"""
        # Create a simple dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Choose Option")
        dialog.geometry("500x300")
        dialog.configure(bg=self.colors['card'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {"choice": default or ""}
        
        # Question label
        tk.Label(dialog, text=question,
                font=('Arial', 11, 'bold'),
                bg=self.colors['card'], fg=self.colors['text'],
                wraplength=450, justify='left').pack(pady=20, padx=20)
        
        # Variable for selection
        choice_var = tk.StringVar(value=default or "")
        
        # If custom number input needed
        if "Enter number" in question or not options:
            input_frame = tk.Frame(dialog, bg=self.colors['card'])
            input_frame.pack(pady=10)
            
            tk.Label(input_frame, text="Number:", 
                    bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0, 10))
            
            entry = tk.Entry(input_frame, font=('Arial', 10), width=10,
                            bg='#3c3c3c', fg='white')
            entry.pack(side='left')
            if default:
                entry.insert(0, default)
            entry.select_range(0, tk.END)
            entry.focus_set()
            
            def submit():
                result["choice"] = entry.get()
                dialog.destroy()
            
            entry.bind('<Return>', lambda e: submit())
        else:
            # Radio buttons for options
            for option in options:
                rb = tk.Radiobutton(dialog, text=option,
                                variable=choice_var,
                                value=option,
                                bg=self.colors['card'], fg=self.colors['text'],
                                selectcolor=self.colors['card'],
                                font=('Arial', 10))
                rb.pack(anchor='w', padx=40, pady=2)
            
            def submit():
                result["choice"] = choice_var.get()
                dialog.destroy()
        
        # OK button
        button_frame = tk.Frame(dialog, bg=self.colors['card'])
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="OK", 
                bg=self.colors['accent'], fg='white',
                command=submit, padx=20).pack()
        
        # Center dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Wait for dialog
        dialog.wait_window(dialog)
        
        return result["choice"]
    def get_link_count_from_question(self, question: str) -> int:
        """Extract link count from question text"""
        import re
        match = re.search(r'Found (\d+)', question)
        if match:
            return int(match.group(1))
        return 50  # Default

    def show_radio_choice_dialog(self, question: str, options: List[str], default=None) -> str:
        """Show radio button choice dialog"""
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Choose Option")
        dialog.geometry("500x300")
        dialog.configure(bg=self.colors['card'])
        
        # Question label
        tk.Label(dialog, text=question, 
                font=('Arial', 11, 'bold'),
                bg=self.colors['card'], fg=self.colors['text'],
                wraplength=450).pack(pady=20, padx=20, anchor='w')
        
        # Variable for selection
        choice_var = tk.StringVar(value=default)
        
        # Options frame
        options_frame = tk.Frame(dialog, bg=self.colors['card'])
        options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create radio buttons
        for option in options:
            tk.Radiobutton(options_frame, text=option,
                          variable=choice_var,
                          value=option,
                          bg=self.colors['card'], fg=self.colors['text'],
                          selectcolor=self.colors['card'],
                          font=('Arial', 10)).pack(anchor='w', pady=2)
        
        # OK button
        button_frame = tk.Frame(dialog, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=20, pady=20)
        
        result = {"choice": None}
        
        def on_ok():
            result["choice"] = choice_var.get()
            dialog.destroy()
        
        tk.Button(button_frame, text="OK", 
                 bg=self.colors['accent'], fg='white',
                 command=on_ok,
                 padx=20).pack()
        
        # Wait for dialog
        self.root.wait_window(dialog)
        
        return result["choice"] or default or options[0]
    
    def submit_choice(self):
        """Submit choice from dialog"""
        choice = self.choice_var.get()
        
        # If no radio button selected but we have custom entry with value
        if not choice and self.custom_entry.get():
            choice = self.custom_entry.get()
        
        if choice:
            # Send choice back to agent
            if hasattr(self, 'current_agent') and self.current_agent:
                self.current_agent.answers.put(choice)
            
            # Hide window
            self.choice_window.grab_release()
            self.choice_window.withdraw()
            
            # Clear custom entry for next use
            self.custom_entry.delete(0, tk.END)
    
    def submit_custom_choice(self):
        """Submit custom choice from entry"""
        choice = self.custom_entry.get()
        if choice:
            # Set the choice variable
            self.choice_var.set(choice)
            # Submit the choice
            self.submit_choice()
    
    def update_progress(self, stage: str, current: int, total: int, message: str):
        """Update progress in status"""
        progress = f"{current}/{total}" if total > 0 else str(current)
        self.status_label.config(
            text=f"🟡 {stage}: {progress} - {message[:30]}...",
            fg=self.colors['warning']
        )
        self.root.update()
    
    def check_ollama(self):
        """Check Ollama connection"""
        def check():
            try:
                success, model = asyncio.run(check_ollama_connection())
                if success:
                    self.status_label.config(text=f"🟢 Connected to Ollama ({model})", 
                                           fg=self.colors['success'])
                    self.add_log('success', f'Connected to Ollama. Using model: {model}')
                else:
                    self.status_label.config(text="🔴 Ollama not running", 
                                           fg=self.colors['error'])
                    self.add_log('error', 'Cannot connect to Ollama')
            except Exception as e:
                self.status_label.config(text="🔴 Connection error", 
                                       fg=self.colors['error'])
                self.add_log('error', f'Ollama check failed: {str(e)}')
        
        threading.Thread(target=check, daemon=True).start()
    
    def start_research(self):
        """Start research session"""
        topic = self.topic_entry.get().strip()
        url = self.url_entry.get().strip()
        
        if not topic:
            messagebox.showerror("Error", "Please enter a research topic.")
            return
        
        # Disable start button, enable stop button
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # Clear previous results and logs
        self.clear_results()
        self.clear_logs()
        
        # Start research in background thread
        thread = threading.Thread(target=self.run_research, 
                                 args=(topic, url), 
                                 daemon=True)
        thread.start()
    
    def stop_research(self):
        """Stop research session"""
        self.session_active = False
        # Stop the agent if it exists
        if hasattr(self, 'current_agent') and self.current_agent:
                self.current_agent.stop_flag = True
                # Put a dummy answer to unblock any waiting ask_user_choice
                try:
                    self.current_agent.answers.put_nowait("STOPPED")
                except:
                    pass

        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="🟡 Research stopped by user", 
                               fg=self.colors['warning'])
        self.add_log('warning', 'Research stopped by user')

    def run_research(self, topic: str, url: str):
        """Run research in background thread"""
        try:
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Check Ollama
            success, model = loop.run_until_complete(check_ollama_connection())
            if not success:
                self.add_log('error', 'Cannot connect to Ollama. Please start Ollama first.')
                self.research_completed(False, "Ollama not running")
                return
            
            # Get ignore terms from the "What to ignore" text box FIRST
            ignore_text = self.ignore_text.get("1.0", tk.END).strip()
            ignore_terms = []
            if ignore_text:
                # Split by commas or newlines
                import re
                ignore_terms = re.split(r'[,;\n]', ignore_text)
                ignore_terms = [term.strip().lower() for term in ignore_terms if term.strip()]
                self.add_log('info', f'Will ignore terms containing: {", ".join(ignore_terms[:5])}')
            
            # Create agent with UI callback AND ignore_terms
            self.current_agent = ModernUIAgent(model, self.ui_callback, ignore_terms)  # Pass ignore_terms here
            
            # Get countries from entry field
            countries_str = self.countries_entry.get().strip()
            countries = []
            if countries_str:
                countries = [c.strip() for c in countries_str.split(',') if c.strip()]
            
            # Run research WITH COUNTRIES
            self.add_log('info', f'Starting research on: {topic}')
            self.add_log('info', f'Starting from: {url}')
            if countries:
                self.add_log('info', f'Country filter: {", ".join(countries)}')
            
            # Add a small delay to ensure UI is ready
            import time
            time.sleep(1)
            
            # Pass countries parameter
            results = loop.run_until_complete(self.current_agent.research(topic, url, countries))
            
            # Handle results
            if results:
                self.research_completed(True, f"Found {len(results)} relevant pages")
            else:
                self.research_completed(True, "No relevant pages found")
        
        except Exception as e:
            # Log the full traceback for debugging
            import traceback
            error_details = traceback.format_exc()
            print(f"DEBUG: Full traceback:\n{error_details}")
            
            self.add_log('error', f'Research error: {str(e)}')
            self.research_completed(False, f"Error: {str(e)}")    

    def research_completed(self, success: bool, message: str):
        """Handle research completion"""
        # Update UI in main thread
        self.root.after(0, lambda: self._update_after_research(success, message))
    
    def _update_after_research(self, success: bool, message: str):
        """Update UI after research completes"""
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        if success:
            self.status_label.config(text=f"🟢 Research complete: {message}", 
                                fg=self.colors['success'])
            self.add_log('success', f'Research completed: {message}')
            
            # Show final summary
            if self.results:
                self.add_log('success', f'🏁 Final Results Summary:')
                self.add_log('info', f'   Total links scanned: {self.current_agent.stats["links_scanned"] if self.current_agent else "N/A"}')
                self.add_log('info', f'   Links evaluated: {len(self.current_agent.links_to_evaluate) if hasattr(self.current_agent, "links_to_evaluate") else "N/A"}')
                self.add_log('info', f'   Pages analyzed: {self.current_agent.stats["pages_analyzed"] if self.current_agent else "N/A"}')
                self.add_log('info', f'   Relevant pages found: {len(self.results)}')
                
                # Show top 3 results
                self.add_log('success', '🏆 Top Results:')
                for i, result in enumerate(self.results[:3], 1):
                    self.add_log('link', f'  {i}. [{result.get("relevance_score", 0):.2f}] {result.get("title", "Unknown")[:50]}...')
        else:
            self.status_label.config(text=f"🔴 Research failed: {message}", 
                                fg=self.colors['error'])
            self.add_log('error', f'Research failed: {message}')

    def ask_number_dialog(self, title: str, prompt: str, default: int = 10, min_val: int = 1, max_val: int = 100) -> int:
        """Show a simple number input dialog (synchronous version)"""
        import threading
        from queue import Queue
        
        result_queue = Queue()
        
        def show_dialog():
            dialog = tk.Toplevel(self.root)
            dialog.title(title)
            dialog.geometry("400x200")
            dialog.configure(bg=self.colors['card'])
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Prompt label
            tk.Label(dialog, text=prompt,
                    font=('Arial', 11),
                    bg=self.colors['card'], fg=self.colors['text'],
                    wraplength=350, justify='left').pack(pady=20, padx=20)
            
            # Number input
            input_frame = tk.Frame(dialog, bg=self.colors['card'])
            input_frame.pack(pady=10)
            
            tk.Label(input_frame, text="Number:", 
                    bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0, 10))
            
            entry_var = tk.StringVar(value=str(default))
            entry = tk.Entry(input_frame, textvariable=entry_var,
                            font=('Arial', 12), width=10,
                            bg='#3c3c3c', fg='white', justify='center')
            entry.pack(side='left')
            entry.select_range(0, tk.END)
            entry.focus_set()
            
            def submit():
                try:
                    num = int(entry_var.get())
                    if num < min_val:
                        num = min_val
                    elif num > max_val:
                        num = max_val
                    result_queue.put(num)
                except ValueError:
                    result_queue.put(default)
                dialog.destroy()
            
            # Submit on Enter
            entry.bind('<Return>', lambda e: submit())
            
            # Buttons
            button_frame = tk.Frame(dialog, bg=self.colors['card'])
            button_frame.pack(pady=20)
            
            tk.Button(button_frame, text="OK", 
                    bg=self.colors['accent'], fg='white',
                    command=submit, padx=20).pack()
            
            # Center dialog
            dialog.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
            dialog.geometry(f"+{x}+{y}")
            
            # Wait for dialog
            dialog.wait_window(dialog)
        
        # Run dialog in main thread
        self.root.after(0, show_dialog)
        
        # Wait for result
        return result_queue.get()
    
    def parse_countries(self, countries_str: str) -> List[str]:
        """Parse comma-separated countries string"""
        if not countries_str:
            return []
        
        countries = []
        for country in countries_str.split(','):
            country = country.strip()
            if country:
                countries.append(country)
        
        return countries
def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set window icon
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    # Center window
    root.update_idletasks()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    x = (width - 1400) // 2
    y = (height - 800) // 2
    root.geometry(f"1400x800+{x}+{y}")
    
    # Create app
    app = ResearchAgentApp(root)
    
    # Handle window close
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    # Simple requirements check
    requirements = ['pydantic', 'playwright', 'httpx']
    missing = []
    
    for req in requirements:
        try:
            __import__(req)
        except ImportError:
            missing.append(req)
    
    if missing:
        print("Missing requirements. Please install:")
        for req in missing:
            print(f"  pip install {req}")
        
        if 'playwright' in missing:
            print("\nAfter installing playwright, run:")
            print("  playwright install chromium")
        
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Run app
    main()