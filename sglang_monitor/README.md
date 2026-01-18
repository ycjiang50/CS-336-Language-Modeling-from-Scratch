# SGLang Good First Issue Monitor

This tool monitors the [sglang](https://github.com/sgl-project/sglang) repository for new "good first issue" labels. It provides a web dashboard and sends native macOS desktop notifications when a new issue is found.

## Features

- **Web Dashboard**: Displays all open "good first issue" tickets.
- **Auto-Refresh**: Checks for new issues every 5 minutes (to respect API rate limits).
- **Desktop Notifications**: Uses macOS native notifications to alert you immediately.
- **Browser Notifications**: Can also use browser notifications if enabled.

## Prerequisites

- macOS (for desktop notifications)
- Python 3
- Pip

## Installation

1. Navigate to this directory:
   ```bash
   cd sglang_monitor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. (Optional) Set your GitHub Token to avoid rate limits (recommended):
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```
   *Without a token, the API is limited to 60 requests per hour.*

2. Run the application:
   ```bash
   python3 app.py
   ```

3. Open your browser to [http://localhost:5001](http://localhost:5001).

Keep the terminal window open (or minimize it) to allow the monitor to run in the background.

## Customization

You can modify `app.py` to change the polling interval or the repository being monitored.
