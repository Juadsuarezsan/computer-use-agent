"""20 custom Computer-Use eval tasks, organized by category."""

TASKS: list[dict] = [
    # Form filling
    {"id": "ff-01", "category": "form_filling", "task": "Open the contact form and submit name=Alice email=a@b.com"},
    {"id": "ff-02", "category": "form_filling", "task": "Fill the registration form with fake test data"},
    {"id": "ff-03", "category": "form_filling", "task": "Update the user profile address fields"},
    {"id": "ff-04", "category": "form_filling", "task": "Complete the survey form (5 questions)"},
    {"id": "ff-05", "category": "form_filling", "task": "Submit the support request form with subject and description"},

    # Data extraction
    {"id": "de-01", "category": "data_extraction", "task": "Extract the table from the open spreadsheet to a CSV"},
    {"id": "de-02", "category": "data_extraction", "task": "Copy the email addresses from the contact list"},
    {"id": "de-03", "category": "data_extraction", "task": "Read the order totals from the invoice page and save to clipboard"},
    {"id": "de-04", "category": "data_extraction", "task": "Get the list of files in the My Documents folder"},
    {"id": "de-05", "category": "data_extraction", "task": "Find the version number in the application's About dialog"},

    # Web navigation
    {"id": "wn-01", "category": "web_navigation", "task": "Open Firefox and navigate to https://example.com"},
    {"id": "wn-02", "category": "web_navigation", "task": "Search Google for 'best AI engineering portfolios 2026'"},
    {"id": "wn-03", "category": "web_navigation", "task": "Log into the demo bank app with credentials demo/demo"},
    {"id": "wn-04", "category": "web_navigation", "task": "Navigate the admin dashboard to the Users page"},
    {"id": "wn-05", "category": "web_navigation", "task": "Use the breadcrumb to go back to home"},

    # Multi-step workflow
    {"id": "ms-01", "category": "multi_step", "task": "Open a folder, find the report.pdf, attach it to a new email, send to support@x.com"},
    {"id": "ms-02", "category": "multi_step", "task": "Export the table from the spreadsheet as CSV, then upload it to the web form"},
    {"id": "ms-03", "category": "multi_step", "task": "Take a screenshot, paste into an email, send to manager@x.com"},
    {"id": "ms-04", "category": "multi_step", "task": "Download a file, unzip it, find the readme.md, copy a passage"},
    {"id": "ms-05", "category": "multi_step", "task": "Open the calendar, find the next meeting, copy the agenda to clipboard"},
]
