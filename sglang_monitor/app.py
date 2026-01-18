import time
import requests
import threading
import os
import subprocess
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Configuration
REPO_OWNER = "sgl-project"
REPO_NAME = "sglang"
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
LABEL = "good first issue"

# Global state
current_issues = []
known_issue_ids = set()
last_updated = None

def send_mac_notification(title, message, link):
    """Sends a native macOS desktop notification."""
    try:
        # Escape quotes for AppleScript
        safe_title = title.replace('"', '\\"')
        safe_message = message.replace('"', '\\"')

        script = f'display notification "{safe_message}" with title "{safe_title}" sound name "Glass"'
        subprocess.run(["osascript", "-e", script])
    except Exception as e:
        print(f"Failed to send notification: {e}")

def fetch_issues():
    """Fetches issues from GitHub API."""
    global current_issues, known_issue_ids, last_updated

    params = {
        "state": "open",
        "labels": LABEL,
        "sort": "created",
        "direction": "desc"
    }

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(API_URL, params=params, headers=headers)
        if response.status_code == 200:
            issues = response.json()

            # Process issues
            new_issue_found = False
            processed_issues = []

            for issue in issues:
                # Skip pull requests (PRs also appear in issues endpoint)
                if "pull_request" in issue:
                    continue

                processed_issues.append({
                    "id": issue["id"],
                    "number": issue["number"],
                    "title": issue["title"],
                    "html_url": issue["html_url"],
                    "created_at": issue["created_at"],
                    "user": issue["user"]["login"],
                    "comments": issue["comments"]
                })

                # Check for new issues (only if we have fetched before)
                if known_issue_ids and issue["id"] not in known_issue_ids:
                    print(f"New issue found: {issue['title']}")
                    send_mac_notification(
                        "New Good First Issue!",
                        f"{issue['title']} (#{issue['number']})",
                        issue["html_url"]
                    )
                    new_issue_found = True

            # Update global state
            current_issues = processed_issues
            known_issue_ids = {i["id"] for i in processed_issues}
            last_updated = time.strftime("%H:%M:%S")

            return True
        elif response.status_code == 403:
            print(f"Error fetching issues: Rate limit exceeded. Please set GITHUB_TOKEN environment variable or wait.")
            return False
        else:
            print(f"Error fetching issues: {response.status_code}")
            return False
    except Exception as e:
        print(f"Exception fetching issues: {e}")
        return False

def background_monitor():
    """Background thread to poll GitHub every 5 minutes."""
    while True:
        fetch_issues()
        time.sleep(300)

# Start background thread
monitor_thread = threading.Thread(target=background_monitor, daemon=True)
monitor_thread.start()

@app.route("/")
def index():
    return render_template("index.html", issues=current_issues, last_updated=last_updated)

@app.route("/api/issues")
def api_issues():
    return jsonify({
        "issues": current_issues,
        "last_updated": last_updated
    })

if __name__ == "__main__":
    # Initial fetch
    fetch_issues()
    app.run(debug=True, port=5001, use_reloader=False)
    # use_reloader=False is important when using background threads to avoid duplicates
