"""
Helper script to update HTML files with API configuration for deployment.

Usage:
    python configure_frontend.py --api-url https://your-ec2-instance.com:8000
    
This will update all HTML files in the static/ directory to use the specified
API URL instead of relative paths.
"""

import argparse
import os
import re
from pathlib import Path

def inject_api_config(html_content, api_url):
    """Inject API configuration script into HTML."""
    
    config_script = f'''<script>
    // Auto-generated API Configuration
    (function() {{
        const API_BASE_URL = '{api_url}';
        const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');
        
        window.getApiUrl = function(path) {{
            if (path.startsWith('/')) {{
                return API_BASE_URL + path;
            }}
            return API_BASE_URL + '/' + path;
        }};
        
        window.getWsUrl = function(path) {{
            if (path.startsWith('/')) {{
                return WS_BASE_URL + path;
            }}
            return WS_BASE_URL + '/' + path;
        }};
        
        // Override fetch for API calls
        const originalFetch = window.fetch;
        window.fetch = function(url, options) {{
            if (typeof url === 'string' && url.startsWith('/api/')) {{
                url = getApiUrl(url);
            }}
            return originalFetch(url, options);
        }};
        
        console.log('API Config:', {{ base: API_BASE_URL, ws: WS_BASE_URL }});
    }})();
    </script>'''
    
    # Check if config already exists
    if 'API_BASE_URL' in html_content:
        # Replace existing config
        pattern = r'<script>.*?API_BASE_URL.*?</script>'
        html_content = re.sub(pattern, config_script, html_content, flags=re.DOTALL)
    else:
        # Inject after <body> tag
        html_content = html_content.replace('<body>', f'<body>\n{config_script}\n', 1)
    
    return html_content

def update_websocket_calls(html_content):
    """Update WebSocket connections to use getWsUrl helper."""
    
    # Pattern: new WebSocket(`${proto}//${location.host}/ws`)
    # Replace with: new WebSocket(window.getWsUrl('/ws'))
    
    patterns = [
        (r'new WebSocket\(`\$\{proto\}//\$\{location\.host\}(/[^`]*)`\)',
         r"new WebSocket(window.getWsUrl('\1'))"),
        (r'new WebSocket\(["\']ws[s]?://[^"\']+(/[^"\']*)["\']',
         r"new WebSocket(window.getWsUrl('\1'))"),
    ]
    
    for pattern, replacement in patterns:
        html_content = re.sub(pattern, replacement, html_content)
    
    return html_content

def process_html_file(file_path, api_url, dry_run=False):
    """Process a single HTML file."""
    
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply transformations
    original_content = content
    content = inject_api_config(content, api_url)
    content = update_websocket_calls(content)
    
    if content == original_content:
        print(f"  ℹ️  No changes needed")
        return False
    
    if dry_run:
        print(f"  ✓ Would update {file_path}")
        return True
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Updated {file_path}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Configure frontend HTML files for production deployment"
    )
    parser.add_argument(
        '--api-url',
        required=True,
        help='Backend API URL (e.g., https://api.example.com or http://ec2-xx-xx-xx-xx.compute.amazonaws.com:8000)'
    )
    parser.add_argument(
        '--static-dir',
        default='static',
        help='Directory containing HTML files (default: static)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing files'
    )
    
    args = parser.parse_args()
    
    # Validate API URL
    if not args.api_url.startswith(('http://', 'https://')):
        print("❌ API URL must start with http:// or https://")
        return 1
    
    # Find HTML files
    static_dir = Path(args.static_dir)
    if not static_dir.exists():
        print(f"❌ Directory not found: {static_dir}")
        return 1
    
    html_files = list(static_dir.glob('*.html'))
    if not html_files:
        print(f"❌ No HTML files found in {static_dir}")
        return 1
    
    print(f"\nConfiguring frontend for API: {args.api_url}")
    print(f"Processing {len(html_files)} HTML file(s)...\n")
    
    updated_count = 0
    for html_file in html_files:
        if process_html_file(html_file, args.api_url, args.dry_run):
            updated_count += 1
    
    print(f"\n{'Would update' if args.dry_run else 'Updated'} {updated_count}/{len(html_files)} file(s)")
    
    if args.dry_run:
        print("\n💡 Run without --dry-run to apply changes")
    
    return 0

if __name__ == '__main__':
    exit(main())
