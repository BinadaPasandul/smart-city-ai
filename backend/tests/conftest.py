"""Keep the normal pytest suite isolated from developer live-provider config."""

import os


# Settings may load backend/.env when app.main is imported during collection.
# Live provider checks belong to scripts/verify_system_acceptance.py instead.
os.environ["WEB_SEARCH_ENABLED"] = "false"
os.environ["TAVILY_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["AUTH_ENABLED"] = "false"
