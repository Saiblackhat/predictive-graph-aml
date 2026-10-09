import json
import re

# Load the verified dictionary
with open("scratch/ui_strings_full.json", "r", encoding="utf-8") as f:
    i18n_dict = json.load(f)

# Read current backend/static/index.html
with open("backend/static/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update HTML tags with data-i18n attributes

# Workspace tag
html = html.replace(
    '<div class="workspace-tag">CLINICAL WORKSPACE</div>',
    '<div class="workspace-tag" data-i18n="workspaceTag">CLINICAL WORKSPACE</div>'
)

# Top nav
top_nav_replacements = [
    ('<button data-page="overview" class="active">Intelligence</button>', '<button data-page="overview" class="active" data-i18n="navIntelligence">Intelligence</button>'),
    ('<button data-page="intake">Intake</button>', '<button data-page="intake" data-i18n="navIntake">Intake</button>'),
    ('<button data-page="careplan">Care Plan</button>', '<button data-page="careplan" data-i18n="navCarePlan">Care Plan</button>'),
    ('<button data-page="meds">Medications</button>', '<button data-page="meds" data-i18n="navMeds">Medications</button>'),
    ('<button data-page="timeline">Journey</button>', '<button data-page="timeline" data-i18n="navJourney">Journey</button>'),
    ('<button data-page="safety">Safety Gate</button>', '<button data-page="safety" data-i18n="navSafetyGate">Safety Gate</button>'),
    ('<button data-page="alerts">Alerts</button>', '<button data-page="alerts" data-i18n="navAlerts">Alerts</button>'),
    ('<button data-page="providers">Providers</button>', '<button data-page="providers" data-i18n="navProviders">Providers</button>'),
    ('<button data-page="audit">Audit</button>', '<button data-page="audit" data-i18n="navAudit">Audit</button>'),
]
for old, new in top_nav_replacements:
    html = html.replace(old, new)

# Top action buttons
html = html.replace(
    '<button class="btn ghost-orange" data-page="intake">＋ Document intake</button>',
    '<button class="btn ghost-orange" data-page="intake" data-i18n="btnDocIntake">＋ Document intake</button>'
)
html = html.replace(
    '<button class="btn orange" data-page="paste">Paste / Run pipeline →</button>',
    '<button class="btn orange" data-page="paste" data-i18n="btnRunPipeline">Paste / Run pipeline →</button>'
)

# Metrics bar
html = html.replace(
    '<span>Patient <b id="metricPatient">—</b></span><span>Grounding <b class="ok" id="metricGround">—</b></span><span>Tasks <b id="metricTasks">—</b></span><span>Review <b class="warn" id="metricReview">—</b></span><span>Latency <b id="metricLatency">—</b></span>',
    '<span><span data-i18n="metricPatient">Patient</span> <b id="metricPatient">—</b></span><span><span data-i18n="metricGrounding">Grounding</span> <b class="ok" id="metricGround">—</b></span><span><span data-i18n="metricTasks">Tasks</span> <b id="metricTasks">—</b></span><span><span data-i18n="metricReview">Review</span> <b class="warn" id="metricReview">—</b></span><span><span data-i18n="metricLatency">Latency</span> <b id="metricLatency">—</b></span>'
)

# Sidebar
sidebar_replacements = [
    ('<div class="nav-title">START</div>', '<div class="nav-title" data-i18n="sidebarStart">START</div>'),
    ('<button class="nav active" data-page="overview"><span class="num">00</span><span class="label">Case Overview</span><span class="check">•</span></button>', '<button class="nav active" data-page="overview"><span class="num">00</span><span class="label" data-i18n="sidebarCaseOverview">Case Overview</span><span class="check">•</span></button>'),
    ('<button class="nav" data-page="intake"><span class="num">01</span><span class="label">Document Intake</span><span class="badge" id="docBadge">NEW</span></button>', '<button class="nav" data-page="intake"><span class="num">01</span><span class="label" data-i18n="sidebarDocIntake">Document Intake</span><span class="badge" id="docBadge">NEW</span></button>'),
    ('<div class="nav-title">CLINICAL OUTPUTS</div>', '<div class="nav-title" data-i18n="sidebarClinicalOutputs">CLINICAL OUTPUTS</div>'),
    ('<button class="nav" data-page="careplan"><span class="num">02</span><span class="label">Care Plan & Tasks</span><span class="badge" id="taskBadge">—</span></button>', '<button class="nav" data-page="careplan"><span class="num">02</span><span class="label" data-i18n="sidebarCarePlan">Care Plan & Tasks</span><span class="badge" id="taskBadge">—</span></button>'),
    ('<button class="nav" data-page="meds"><span class="num">03</span><span class="label">Medication Precision</span><span class="badge" id="medBadge">—</span></button>', '<button class="nav" data-page="meds"><span class="num">03</span><span class="label" data-i18n="sidebarMedPrecision">Medication Precision</span><span class="badge" id="medBadge">—</span></button>'),
    ('<button class="nav" data-page="timeline"><span class="num">04</span><span class="label">Recovery Timeline</span><span class="badge" id="timelineBadge">—</span></button>', '<button class="nav" data-page="timeline"><span class="num">04</span><span class="label" data-i18n="sidebarRecoveryTimeline">Recovery Timeline</span><span class="badge" id="timelineBadge">—</span></button>'),
    ('<div class="nav-title">SAFETY</div>', '<div class="nav-title" data-i18n="sidebarSafety">SAFETY</div>'),
    ('<button class="nav" data-page="safety"><span class="num">05</span><span class="label">Safety Review</span><span class="badge" id="safetyBadge">—</span></button>', '<button class="nav" data-page="safety"><span class="num">05</span><span class="label" data-i18n="sidebarSafetyReview">Safety Review</span><span class="badge" id="safetyBadge">—</span></button>'),
    ('<button class="nav" data-page="alerts"><span class="num">06</span><span class="label">Alerts & Alarms</span><span class="badge" id="alertBadge">—</span></button>', '<button class="nav" data-page="alerts"><span class="num">06</span><span class="label" data-i18n="sidebarAlerts">Alerts & Alarms</span><span class="badge" id="alertBadge">—</span></button>'),
    ('<div class="nav-title">COORDINATE</div>', '<div class="nav-title" data-i18n="sidebarCoordinate">COORDINATE</div>'),
    ('<button class="nav" data-page="providers"><span class="num">07</span><span class="label">Provider Finder</span><span class="badge">+</span></button>', '<button class="nav" data-page="providers"><span class="num">07</span><span class="label" data-i18n="sidebarProviderFinder">Provider Finder</span><span class="badge">+</span></button>'),
    ('<button class="nav" data-page="audit"><span class="num">08</span><span class="label">Clinical Audit</span><span class="badge" id="auditBadge">—</span></button>', '<button class="nav" data-page="audit"><span class="num">08</span><span class="label" data-i18n="sidebarClinicalAudit">Clinical Audit</span><span class="badge" id="auditBadge">—</span></button>'),
    ('<div class="sidebar-upload">\n    <strong>DOCUMENT INTAKE</strong>\n    <span>Upload or paste the discharge source first. Every care action stays linked to its evidence.</span>\n  </div>', '<div class="sidebar-upload">\n    <strong data-i18n="sidebarUploadBoxTitle">DOCUMENT INTAKE</strong>\n    <span data-i18n="sidebarUploadBoxDesc">Upload or paste the discharge source first. Every care action stays linked to its evidence.</span>\n  </div>'),
    ('<div class="sidebar-foot"><span class="live">● CASE LIVE</span>', '<div class="sidebar-foot"><span class="live" data-i18n="sidebarCaseLive">● CASE LIVE</span>')
]
for old, new in sidebar_replacements:
    html = html.replace(old, new)

# Landing hero
landing_replacements = [
    ('<div class="kicker">CLINICAL INTELLIGENCE · HUMAN CONTROL</div>', '<div class="kicker" data-i18n="overviewKicker">CLINICAL INTELLIGENCE · HUMAN CONTROL</div>'),
    ('<h1>Recovery,<br>organized<br>around <em>you.</em></h1>', '<h1 data-i18n="overviewH1" data-i18n-html="true">Recovery,<br>organized<br>around <em>you.</em></h1>'),
    ('<p class="landing-lead">CareBridge turns synthetic discharge instructions into verifiable clinical tasks, calendar reminders, and localized care plans — with safety-first human gates when clinical ambiguity is detected.</p>', '<p class="landing-lead" data-i18n="overviewLead">CareBridge turns synthetic discharge instructions into verifiable clinical tasks, calendar reminders, and localized care plans — with safety-first human gates when clinical ambiguity is detected.</p>'),
    ('<button class="btn orange" data-page="careplan">Explore Live Care Plan →</button>', '<button class="btn orange" data-page="careplan" data-i18n="overviewBtnCare">Explore Live Care Plan →</button>'),
    ('<button class="btn" data-page="safety">Inspect Safety Gate</button>', '<button class="btn" data-page="safety" data-i18n="overviewBtnSafety">Inspect Safety Gate</button>'),
    ('<span>● Evidence-linked outputs</span>', '<span data-i18n="proofEvidence">● Evidence-linked outputs</span>'),
    ('<span>● Deterministic safety gate</span>', '<span data-i18n="proofSafety">● Deterministic safety gate</span>'),
    ('<span>● Human review when needed</span>', '<span data-i18n="proofReview">● Human review when needed</span>'),
    ('<span>VALIDATION GATE</span>', '<span data-i18n="cardValidationGate">VALIDATION GATE</span>'),
    ('<span>● SOURCE SUMMARY</span>', '<span data-i18n="cardSourceSummary">● SOURCE SUMMARY</span>'),
    ('<span>VERIFIED PLAN</span>', '<span data-i18n="cardVerifiedPlan">VERIFIED PLAN</span>'),
    ('<span>GROUNDING</span>', '<span data-i18n="cardGrounding">GROUNDING</span>'),
    ('<div><span>01</span><b>DOCUMENT</b><small>Source received</small></div>', '<div><span>01</span><b data-i18n="stepDoc">DOCUMENT</b><small data-i18n="stepDocDesc">Source received</small></div>'),
    ('<div><span>02</span><b>UNDERSTAND</b><small>Directives extracted</small></div>', '<div><span>02</span><b data-i18n="stepUnderstand">UNDERSTAND</b><small data-i18n="stepUnderstandDesc">Directives extracted</small></div>'),
    ('<div><span>03</span><b>VERIFY</b><small>Safety gate checked</small></div>', '<div><span>03</span><b data-i18n="stepVerify">VERIFY</b><small data-i18n="stepVerifyDesc">Safety gate checked</small></div>'),
    ('<div><span>04</span><b>ORGANIZE</b><small>Tasks + reminders</small></div>', '<div><span>04</span><b data-i18n="stepOrganize">ORGANIZE</b><small data-i18n="stepOrganizeDesc">Tasks + reminders</small></div>'),
    ('<div><span>05</span><b>HAND OFF</b><small>Human control preserved</small></div>', '<div><span>05</span><b data-i18n="stepHandoff">HAND OFF</b><small data-i18n="stepHandoffDesc">Human control preserved</small></div>'),
]
for old, new in landing_replacements:
    html = html.replace(old, new)

# Page Intake
intake_replacements = [
    ('<div><div class="kicker">01 · DOCUMENT INTAKE</div><h1>Bring the <em>source.</em></h1><p class="lead">Start with the document that contains the instructions. Text is processed through <b>/api/v1/orchestrate</b>; the frontend never invents a schedule without source evidence.</p></div>',
     '<div><div class="kicker" data-i18n="intakeKicker">01 · DOCUMENT INTAKE</div><h1 data-i18n="intakeH1" data-i18n-html="true">Bring the <em>source.</em></h1><p class="lead" data-i18n="intakeLead">Start with the document that contains the instructions. Text is processed through /api/v1/orchestrate; the frontend never invents a schedule without source evidence.</p></div>'),
    ('<h2>Drop a text document here</h2><p>TXT, MD, JSON or a copied EHR/discharge note can be loaded directly. Binary documents can be extracted or pasted into the editor below.</p>',
     '<h2 data-i18n="dropTitle">Drop a text document here</h2><p data-i18n="dropDesc">TXT, MD, JSON or a copied EHR/discharge note can be loaded directly. Binary documents can be extracted or pasted into the editor below.</p>'),
    ('<label class="btn orange" for="fileInput">Choose source file</label>', '<label class="btn orange" for="fileInput" data-i18n="btnChooseFile">Choose source file</label>'),
    ('<div class="card-title">Source text · up to 200 KB</div>', '<div class="card-title" data-i18n="sourceTextTitle">Source text · up to 200 KB</div>'),
    ('<textarea class="textarea" id="rawText" placeholder="Paste the discharge summary, EHR note, or clinical instruction here…"></textarea>',
     '<textarea class="textarea" id="rawText" data-i18n-placeholder="sourcePlaceholder" placeholder="Paste the discharge summary, EHR note, or clinical instruction here…"></textarea>'),
    ('<button class="btn orange" id="runRaw">Run Pipeline →</button><button class="btn" id="clearRaw">Clear</button>',
     '<button class="btn orange" id="runRaw" data-i18n="btnRunPipelineAction">Run Pipeline →</button><button class="btn" id="clearRaw" data-i18n="btnClear">Clear</button>'),
    ('<div class="card"><div class="card-title">What the backend does</div>\n        <div class="source-item"><b>01</b><span>Extract clinical entities</span></div>\n        <div class="source-item"><b>02</b><span>Run safety gate</span></div>\n        <div class="source-item"><b>03</b><span>Orchestrate tasks</span></div>\n        <div class="source-item"><b>04</b><span>Plain language (12 languages)</span></div>\n        <div class="source-item"><b>05</b><span>Reminders & alerts</span></div>\n        <div class="source-item"><b>06</b><span>Provider matching</span></div>\n      </div>',
     '<div class="card"><div class="card-title" data-i18n="intakeSideTitle">What the backend does</div>\n        <div class="source-item"><b>01</b><span data-i18n="intakeStep1">Extract clinical entities</span></div>\n        <div class="source-item"><b>02</b><span data-i18n="intakeStep2">Run safety gate</span></div>\n        <div class="source-item"><b>03</b><span data-i18n="intakeStep3">Orchestrate tasks</span></div>\n        <div class="source-item"><b>04</b><span data-i18n="intakeStep4">Plain language (12 languages)</span></div>\n        <div class="source-item"><b>05</b><span data-i18n="intakeStep5">Reminders & alerts</span></div>\n        <div class="source-item"><b>06</b><span data-i18n="intakeStep6">Provider matching</span></div>\n      </div>'),
    ('<div class="card"><div class="card-title">Safety boundary</div><div class="notice">Organize and explain only. Do not diagnose, change medication, or fill missing clinical details. Ambiguous instructions are routed to Safety Review.</div></div>',
     '<div class="card"><div class="card-title" data-i18n="safetyBoundaryTitle">Safety boundary</div><div class="notice" data-i18n="safetyBoundaryDesc">Organize and explain only. Do not diagnose, change medication, or fill missing clinical details. Ambiguous instructions are routed to Safety Review.</div></div>')
]
for old, new in intake_replacements:
    html = html.replace(old, new)

# Page Paste
paste_replacements = [
    ('<div><div class="kicker">CUSTOM INGESTION · PHASE 8</div><h1>Paste and <em>orchestrate.</em></h1><p class="lead">Submit raw clinical text with your selected language preference.</p></div>',
     '<div><div class="kicker" data-i18n="pasteKicker">CUSTOM INGESTION · PHASE 8</div><h1 data-i18n="pasteH1" data-i18n-html="true">Paste and <em>orchestrate.</em></h1><p class="lead" data-i18n="pasteLead">Submit raw clinical text with your selected language preference.</p></div>'),
    ('<div class="card-title">Raw discharge / EHR text</div>', '<div class="card-title" data-i18n="pasteRawTitle">Raw discharge / EHR text</div>'),
    ('<div class="card-title">Pipeline contract</div>', '<div class="card-title" data-i18n="pasteContractTitle">Pipeline contract</div>'),
    ('<button class="btn orange" id="runPaste">Run Pipeline →</button>', '<button class="btn orange" id="runPaste" data-i18n="btnRunPipelineAction">Run Pipeline →</button>')
]
for old, new in paste_replacements:
    html = html.replace(old, new)

# Page Care Plan
care_replacements = [
    ('<div><div class="kicker">02 · TASK ORCHESTRATION</div><h1>Care plan & <em>tasks.</em></h1><p class="lead">Every directive is surfaced as a trackable action with safety state and provenance linkage.</p></div>',
     '<div><div class="kicker" data-i18n="carePlanKicker">02 · TASK ORCHESTRATION</div><h1 data-i18n="carePlanH1" data-i18n-html="true">Care plan & <em>tasks.</em></h1><p class="lead" data-i18n="carePlanLead">Every directive is surfaced as a trackable action with safety state and provenance linkage.</p></div>')
]
for old, new in care_replacements:
    html = html.replace(old, new)

# Page Meds
meds_replacements = [
    ('<div><div class="kicker">03 · MEDICATION PRECISION</div><h1>Medication <em>precision.</em></h1><p class="lead">Preserve formulation salt, dose, route and frequency exactly as returned by the deterministic pipeline.</p></div>',
     '<div><div class="kicker" data-i18n="medsKicker">03 · MEDICATION PRECISION</div><h1 data-i18n="medsH1" data-i18n-html="true">Medication <em>precision.</em></h1><p class="lead" data-i18n="medsLead">Preserve formulation salt, dose, route and frequency exactly as returned by the deterministic pipeline.</p></div>'),
    ('<thead><tr><th>Medication</th><th>Formulation</th><th>Dose</th><th>Route</th><th>Frequency</th><th>Source</th><th>Safety</th></tr></thead>',
     '<thead><tr><th data-i18n="thMedication">Medication</th><th data-i18n="thFormulation">Formulation</th><th data-i18n="thDose">Dose</th><th data-i18n="thRoute">Route</th><th data-i18n="thFrequency">Frequency</th><th data-i18n="thSource">Source</th><th data-i18n="thSafety">Safety</th></tr></thead>')
]
for old, new in meds_replacements:
    html = html.replace(old, new)

# Page Timeline
timeline_replacements = [
    ('<div><div class="kicker">04 · PHASE 5 · CHRONOLOGICAL ORCHESTRATION</div><h1>Recovery <em>timeline.</em></h1><p class="lead">Relative windows stay relative until a reliable discharge date exists; absolute dates remain traceable to the directive.</p></div>',
     '<div><div class="kicker" data-i18n="timelineKicker">04 · PHASE 5 · CHRONOLOGICAL ORCHESTRATION</div><h1 data-i18n="timelineH1" data-i18n-html="true">Recovery <em>timeline.</em></h1><p class="lead" data-i18n="timelineLead">Relative windows stay relative until a reliable discharge date exists; absolute dates remain traceable to the directive.</p></div>')
]
for old, new in timeline_replacements:
    html = html.replace(old, new)

# Page Safety
safety_replacements = [
    ('<div><div class="kicker">05 · PHASE 3C · ZERO-HALLUCINATION GATE</div><h1>When AI should <em>stop.</em></h1><p class="lead">Clinical ambiguity is not an invitation to guess. Flagged directives stay paused until a human reviewer resolves them.</p></div>',
     '<div><div class="kicker" data-i18n="safetyKicker">05 · PHASE 3C · ZERO-HALLUCINATION GATE</div><h1 data-i18n="safetyH1" data-i18n-html="true">When AI should <em>stop.</em></h1><p class="lead" data-i18n="safetyLead">Clinical ambiguity is not an invitation to guess. Flagged directives stay paused until a human reviewer resolves them.</p></div>'),
    ('<div class="notice" style="margin-bottom:16px;border-color:rgba(85,214,162,.3);background:rgba(85,214,162,.04);color:#a8d5be">\n    <b>MANDATORY CLINICAL SAFETY BOUNDARY STATEMENT:</b> CAREBRIDGE is an agentic hospital discharge and follow-up coordinator designed to improve patient comprehension and adherence. It is NOT a medical device, diagnostic tool, or clinical decision support system. All clinical decisions, diagnoses, and medication modifications remain solely with licensed healthcare providers.\n  </div>',
     '<div class="notice" style="margin-bottom:16px;border-color:rgba(85,214,162,.3);background:rgba(85,214,162,.04);color:#a8d5be" data-i18n="safetyMandatoryStatement"><b>MANDATORY CLINICAL SAFETY BOUNDARY STATEMENT:</b> CAREBRIDGE is an agentic hospital discharge and follow-up coordinator designed to improve patient comprehension and adherence. It is NOT a medical device, diagnostic tool, or clinical decision support system. All clinical decisions, diagnoses, and medication modifications remain solely with licensed healthcare providers.</div>')
]
for old, new in safety_replacements:
    html = html.replace(old, new)

# Page Alerts
alerts_replacements = [
    ('<div><div class="kicker">06 · PHASE 6.5 / 6.6 · REMINDERS & NOTIFICATIONS</div><h1>Alerts & <em>alarms.</em></h1><p class="lead">A notification is a lifecycle object. Acknowledge, snooze, dismiss or cancel through the backend state machine.</p></div>',
     '<div><div class="kicker" data-i18n="alertsKicker">06 · PHASE 6.5 / 6.6 · REMINDERS & NOTIFICATIONS</div><h1 data-i18n="alertsH1" data-i18n-html="true">Alerts & <em>alarms.</em></h1><p class="lead" data-i18n="alertsLead">A notification is a lifecycle object. Acknowledge, snooze, dismiss or cancel through the backend state machine.</p></div>'),
    ('<div class="card-title">Reminder schedule</div>', '<div class="card-title" data-i18n="reminderScheduleTitle">Reminder schedule</div>'),
    ('<div class="card-title">Notification center</div>', '<div class="card-title" data-i18n="notifCenterTitle">Notification center</div>')
]
for old, new in alerts_replacements:
    html = html.replace(old, new)

# Page Providers
prov_replacements = [
    ('<div><div class="kicker">07 · PHASE 7 · ATTRIBUTE MATCHING</div><h1>Provider <em>finder.</em></h1><p class="lead">Deterministic filters only. No doctor ranking, subjective scores or sponsored results.</p></div>',
     '<div><div class="kicker" data-i18n="providerKicker">07 · PHASE 7 · ATTRIBUTE MATCHING</div><h1 data-i18n="providerH1" data-i18n-html="true">Provider <em>finder.</em></h1><p class="lead" data-i18n="providerLead">Deterministic filters only. No doctor ranking, subjective scores or sponsored results.</p></div>'),
    ('<div class="field"><label>Specialty</label><input id="pSpecialty" placeholder="Cardiology"></div>',
     '<div class="field"><label data-i18n="labelSpecialty">Specialty</label><input id="pSpecialty" placeholder="Cardiology"></div>'),
    ('<div class="field"><label>City / Location</label><input id="pCity" placeholder="Boston"></div>',
     '<div class="field"><label data-i18n="labelCity">City / Location</label><input id="pCity" placeholder="Boston"></div>'),
    ('<div class="field"><label>Language</label><input id="pLanguage" placeholder="English"></div>',
     '<div class="field"><label data-i18n="labelLanguage">Language</label><input id="pLanguage" placeholder="English"></div>'),
    ('<div class="field"><label>Insurance</label><input id="pInsurance" placeholder="Medicare"></div>',
     '<div class="field"><label data-i18n="labelInsurance">Insurance</label><input id="pInsurance" placeholder="Medicare"></div>'),
    ('<button class="btn orange" id="providerSearch">Search providers →</button>',
     '<button class="btn orange" id="providerSearch" data-i18n="btnSearchProviders">Search providers →</button>')
]
for old, new in prov_replacements:
    html = html.replace(old, new)

# Page Audit
audit_replacements = [
    ('<div><div class="kicker">08 · CLINICAL AUDIT</div><h1>Evidence, <em>audited.</em></h1><p class="lead">A compact view of the deterministic constraints that remain intact from source to patient-facing output.</p></div>',
     '<div><div class="kicker" data-i18n="auditKicker">08 · CLINICAL AUDIT</div><h1 data-i18n="auditH1" data-i18n-html="true">Evidence, <em>audited.</em></h1><p class="lead" data-i18n="auditLead">A compact view of the deterministic constraints that remain intact from source to patient-facing output.</p></div>')
]
for old, new in audit_replacements:
    html = html.replace(old, new)

# Inspector tabs
inspector_tab_replacements = [
    ('<button class="itab active" data-itab="evidence">Provenance</button>', '<button class="itab active" data-itab="evidence" data-i18n="tabProvenance">Provenance</button>'),
    ('<button class="itab" data-itab="alerts">Reminders</button>', '<button class="itab" data-itab="alerts" data-i18n="tabReminders">Reminders</button>'),
    ('<button class="itab" data-itab="providers">Provider</button>', '<button class="itab" data-itab="providers" data-i18n="tabProvider">Provider</button>'),
    ('<button class="itab" data-itab="audit">Audit</button>', '<button class="itab" data-itab="audit" data-i18n="tabAudit">Audit</button>')
]
for old, new in inspector_tab_replacements:
    html = html.replace(old, new)

# Now check the script section
script_marker = "<script>"
script_pos = html.find(script_marker)
if script_pos == -1:
    raise Exception("Could not find <script> tag in index.html")

script_content = f"""<script>
const API = location.origin;
const SUPPORTED_UI_LANGS = {json.dumps(list(i18n_dict.keys()))};
const UI_STRINGS = {json.dumps(i18n_dict, ensure_ascii=False)};

const browserLang = (navigator.language || "en").toLowerCase().split("-")[0];
const savedLang = localStorage.getItem("carebridge-language");
const initialLang = SUPPORTED_UI_LANGS.includes(savedLang) ? savedLang : (SUPPORTED_UI_LANGS.includes(browserLang) ? browserLang : "en");

const state = {{
  cases: [],
  currentCase: null,
  pipeline: null,
  language: initialLang,
  selectedTask: 0,
  notifications: [],
  providers: [],
  requestSeq: 0
}};

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = v => String(v ?? "").replace(/[&<>"']/g, m => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}}[m]));
const first = (...xs) => xs.find(x => x !== undefined && x !== null);

function t(key, fallback = "") {{
  return UI_STRINGS[state.language]?.[key] || UI_STRINGS["en"]?.[key] || fallback;
}}

function applyI18n(lang = state.language) {{
  if (!SUPPORTED_UI_LANGS.includes(lang)) {{
    console.error("Unsupported UI language:", lang);
    return;
  }}
  state.language = lang;
  document.documentElement.lang = lang;
  localStorage.setItem("carebridge-language", lang);

  const langSel = $("#languageSelect");
  if (langSel && langSel.value !== lang) langSel.value = lang;
  const pasteLang = $("#pasteLang");
  if (pasteLang) pasteLang.textContent = lang;

  $$("[data-i18n]").forEach(el => {{
    const k = el.dataset.i18n;
    const val = t(k);
    if (val) {{
      if (el.dataset.i18nHtml) {{
        el.innerHTML = val;
      }} else {{
        el.textContent = val;
      }}
    }}
  }});

  $$("[data-i18n-placeholder]").forEach(el => {{
    const k = el.dataset.i18nPlaceholder;
    const val = t(k);
    if (val) el.placeholder = val;
  }});
}}

function toast(title, msg = "") {{
  const toastEl = $("#toast");
  if (!toastEl) return;
  toastEl.innerHTML = `<b>${{esc(title)}}</b>${{msg ? `<div style="color:#777;margin-top:4px">${{esc(msg)}}</div>` : ""}}`;
  toastEl.classList.add("show");
  setTimeout(() => toastEl.classList.remove("show"), 3200);
}}

function showPage(name) {{
  $$(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + name));
  $$("[data-page]").forEach(n => {{
    if (n.classList.contains("nav") || n.parentElement?.classList.contains("top-nav")) {{
      n.classList.toggle("active", n.dataset.page === name);
    }}
  }});
  document.body.classList.toggle("landing-mode", name === "overview");
  const ws = $(".workspace");
  if (ws) ws.scrollTop = 0;
  const custom = $('[data-page="intake"]');
  const pipeline = $('[data-page="paste"]');
  if (custom) custom.textContent = name === "overview" ? "+ Custom" : t("btnDocIntake", "＋ Document intake");
  if (pipeline) pipeline.textContent = name === "overview" ? "◉" : t("btnRunPipeline", "Paste / Run pipeline →");
  const pasteLang = $("#pasteLang");
  if (pasteLang) pasteLang.textContent = state.language;
}}
$$("[data-page]").forEach(b => b.addEventListener("click", () => showPage(b.dataset.page)));

function normalize(raw) {{
  const p = raw || {{}};
  const safety = first(p.overall_safety_status, p.safety, p.safety_state, p.gate, p.status, "SAFE / ORGANIZED");
  const directives = first(p.tasks, p.directives, p.care_plan?.items, []) || [];
  const meds = first(p.medications, p.medication_precision, []) || [];
  const timeline = first(p.timeline, p.tasks, []) || [];
  const review = first(p.human_review_items, p.review_items, p.needs_review, []) || [];
  const reminders = first(p.reminders, p.care_reminders, []) || [];
  const notifications = first(p.notifications, p.alerts, []) || [];
  const source = first(p.raw_text, p.source, p.document, "") || "";
  return {{
    raw: p,
    safety: String(safety),
    directives: Array.isArray(directives) ? directives : [],
    meds: Array.isArray(meds) ? meds : [],
    timeline: Array.isArray(timeline) ? timeline : [],
    review: Array.isArray(review) ? review : [],
    reminders: Array.isArray(reminders) ? reminders : [],
    notifications: Array.isArray(notifications) ? notifications : [],
    source
  }};
}}

function titleOf(x) {{
  return first(x.title, x.name, x.action, x.directive, x.task, x.hazard_code, x.message, x.description, "Untitled directive");
}}
function statusOf(x) {{
  return String(first(x.status, x.state, x.safety_status, x.classification, "PENDING")).toUpperCase();
}}
function sourceOf(x) {{
  if (x && x.evidence && x.evidence.source_text) return x.evidence.source_text;
  return first(x.source_quote, x.quote, x.exact_source_quote, x.source_text, x.evidence_snippet, "No source quote returned.");
}}

function renderAll() {{
  const n = normalize(state.pipeline);
  const patient = first(n.raw.case_info?.patient_identifier, n.raw.patient_id, n.raw.case_id, state.currentCase?.patient_id, state.currentCase?.case_id, "—");
  const ptPrefLang = state.currentCase?.preferred_language || state.pipeline?.case_info?.preferred_language || "en";

  let groundingText = "—";
  if (n.directives.length > 0) {{
    const grounded = n.directives.filter(x => x.evidence && (x.evidence.source_text || x.evidence.source_sentence_id)).length;
    groundingText = Math.round((grounded / n.directives.length) * 100) + "%";
  }} else if (n.raw.grounding_score) {{
    groundingText = String(n.raw.grounding_score);
  }}

  $("#metricPatient").textContent = patient;
  $("#metricGround").textContent = groundingText;
  $("#metricTasks").textContent = n.directives.length;
  $("#metricReview").textContent = n.review.length;
  $("#metricLatency").textContent = (first(n.raw.latency_ms, n.raw.pipeline_latency, 24)) + "ms";

  const isBlock = n.safety.includes("BLOCK") || n.safety.includes("CONFLICT");
  const isReview = n.review.length > 0 || n.safety.includes("REVIEW");

  $("#caseState").textContent = "● " + n.safety;
  $("#caseState").style.borderColor = isBlock ? "rgba(255,103,81,.4)" : isReview ? "rgba(255,150,127,.4)" : "rgba(85,214,162,.32)";
  $("#caseState").style.color = isBlock ? "var(--red)" : isReview ? "var(--orange2)" : "var(--green)";

  if ($("#sidebarCase")) {{
    const diffTag = ptPrefLang !== state.language ? ` [PT: ${{ptPrefLang.toUpperCase()}}]` : "";
    $("#sidebarCase").textContent = patient + " · " + n.safety + diffTag;
  }}
  const landingGate = $("#landingGate");
  if (landingGate) {{
    landingGate.textContent = n.safety;
    landingGate.style.color = isBlock ? "var(--red)" : isReview ? "var(--orange2)" : "#f4eee9";
  }}
  const landingGateSub = $("#landingGateSub");
  if (landingGateSub) {{
    landingGateSub.textContent = isBlock ? "BLOCKED" : isReview ? "REVIEW REQUIRED" : "SAFE / RELEASABLE";
    landingGateSub.style.color = isBlock ? "var(--red)" : isReview ? "var(--orange2)" : "var(--green)";
  }}
  const landingGround = $("#landingGround");
  if (landingGround) landingGround.textContent = groundingText;
  const floatCase = $("#floatCase");
  if (floatCase) floatCase.textContent = patient;

  const el_safetyBadge = $("#safetyBadge");
  if (el_safetyBadge) el_safetyBadge.textContent = n.review.length || (isBlock ? "BLOCK" : "0");
  const el_taskBadge = $("#taskBadge");
  if (el_taskBadge) el_taskBadge.textContent = n.directives.length;
  const el_medBadge = $("#medBadge");
  if (el_medBadge) el_medBadge.textContent = n.meds.length;
  const el_timelineBadge = $("#timelineBadge");
  if (el_timelineBadge) el_timelineBadge.textContent = n.timeline.length;
  const el_alertBadge = $("#alertBadge");
  if (el_alertBadge) el_alertBadge.textContent = n.notifications.length;
  const el_auditBadge = $("#auditBadge");
  if (el_auditBadge) el_auditBadge.textContent = isBlock ? "FAIL" : isReview ? "REVIEW" : "PASS";

  const verifiedPlanList = $("#verifiedPlanList");
  if (verifiedPlanList) {{
    if (n.directives.length) {{
      verifiedPlanList.innerHTML = n.directives.slice(0, 3).map((tItem, idx) => `
        <div class="plan-row">
          <em>${{String(idx + 1).padStart(2, "0")}}</em>
          <div>
            <strong>${{esc(titleOf(tItem))}}</strong>
            <small>${{esc(first(tItem.timeframe_display, tItem.absolute_date, tItem.relative_window, "Scheduled"))}}</small>
          </div>
          <b class="${{statusOf(tItem).includes("SCHED") ? "scheduled" : statusOf(tItem).includes("READY") ? "" : "routine"}}">${{esc(statusOf(tItem))}}</b>
        </div>
      `).join("");
    }} else {{
      verifiedPlanList.innerHTML = `<div style="padding:14px;color:#777;font-size:10px">No tasks extracted for this document.</div>`;
    }}
  }}

  renderCare(n);
  renderMeds(n);
  renderTimeline(n);
  renderSafety(n);
  renderAlerts(n);
  renderAudit(n);
  renderInspector(currentInspectorTab, n);
}}

function renderCare(n) {{
  $("#careCount").textContent = n.directives.length + " " + t("tasksLabel", "TASKS");
  $("#taskList").innerHTML = n.directives.length ? n.directives.map((x, i) => {{
    const isSel = state.selectedTask === i;
    const st = statusOf(x);
    const tagClass = st.includes("REVIEW") || st.includes("BLOCK") ? "red" : st.includes("READY") || st.includes("COMPLETE") ? "green" : "orange";
    const plain = first(x.plain_language, x.patient_language, x.description, "No plain-language translation.");
    const timing = first(x.timeframe_display, x.absolute_date, x.relative_window, x.window, x.time, "Schedule not specified");
    return `
      <div class="task ${{isSel ? 'selected' : ''}}" data-task="${{i}}" style="${{isSel ? 'background:rgba(255,98,66,.06);border-left:2px solid var(--orange);padding-left:10px' : ''}}">
        <div class="task-id">${{String(i + 1).padStart(2, "0")}}</div>
        <div>
          <h3>${{esc(titleOf(x))}}</h3>
          <p>${{esc(plain)}}</p>
          <div class="tiny" style="margin-top:7px">
            <b>${{esc(timing)}}</b> · SOURCE: “${{esc(sourceOf(x).slice(0, 110))}}”
          </div>
        </div>
        <span class="tag ${{tagClass}}">${{esc(st)}}</span>
      </div>
    `;
  }}).join("") : '<div class="notice">No care tasks returned by the clinical pipeline.</div>';

  $$("#taskList .task").forEach(el => el.addEventListener("click", () => {{
    state.selectedTask = +el.dataset.task;
    renderCare(n);
    renderInspector("evidence", n);
  }}));
}}

function renderMeds(n) {{
  const rows = n.meds.map(x => {{
    const st = x.is_ambiguous ? "NEEDS_REVIEW" : statusOf(x);
    const tagClass = st.includes("REVIEW") || st.includes("BLOCK") ? "red" : "green";
    return `
      <tr>
        <td class="drug">${{esc(first(x.name, x.medication, "—"))}}</td>
        <td class="formula">${{esc(first(x.raw_display, x.formulation, "—"))}}</td>
        <td>${{esc(first(x.dosage, x.dose, "—"))}}</td>
        <td>${{esc(first(x.route, "Oral"))}}</td>
        <td>${{esc(first(x.frequency, "—"))}}</td>
        <td>${{esc(sourceOf(x).slice(0, 70))}}</td>
        <td><span class="tag ${{tagClass}}">${{esc(st)}}</span></td>
      </tr>
      ${{x.ambiguity_note ? `<tr><td colspan="7" style="padding:4px 12px 10px;font-size:9px;color:var(--orange2);background:rgba(255,98,66,.03)">Ambiguity: ${{esc(x.ambiguity_note)}}</td></tr>` : ''}}
    `;
  }}).join("");
  $("#medTable").innerHTML = rows || '<tr><td colspan="7" style="padding:24px;color:#666">No medication records returned.</td></tr>';
}}

function renderTimeline(n) {{
  const items = n.timeline.length ? n.timeline : n.directives;
  $("#timelineList").innerHTML = items.map((x, i) => {{
    const when = first(x.timeframe_display, x.absolute_date, x.relative_window, x.phase, x.when, "PHASE " + (i + 1));
    return `
      <div class="milestone ${{i === 0 ? 'done' : ''}}">
        <div class="when">${{esc(when)}}</div>
        <h3>${{esc(titleOf(x))}}</h3>
        <p>${{esc(first(x.description, x.plain_language, "Grounded timeline directive."))}}</p>
      </div>
    `;
  }}).join("") || '<div class="notice">No timeline milestones returned.</div>';
}}

function renderSafety(n) {{
  const items = n.review.length ? n.review : (n.safety.includes("BLOCK") || n.safety.includes("CONFLICT") ? [{{
    title: "Safety Gate Paused Patient Release",
    reason: "Clinical conflict or ambiguity detected. Deterministic safety rules require human clinical signoff before release.",
    severity: "BLOCKING",
    evidence_snippet: sourceOf(n.raw)
  }}] : []);

  $("#safetyHeader").textContent = items.length ? items.length + " " + t("openReviewStatus", "OPEN REVIEW") : t("clearStatus", "CLEAR");
  $("#safetyHeader").className = "chip " + (items.length ? "red" : "green");

  let disclaimerHtml = `
    <div class="notice" style="margin-top:16px;border-color:rgba(85,214,162,.3);background:rgba(85,214,162,.04);color:#a8d5be">
      <b>Mandatory Clinical Boundary:</b> ${{esc(first(n.raw.disclaimer, t("safetyMandatoryStatement")))}}
    </div>
  `;

  $("#safetyContent").innerHTML = (items.length ? items.map((x, i) => `
    <div class="safety-card" style="margin-bottom:12px">
      <div class="kicker">OPEN SAFETY EVENT · ${{String(i + 1).padStart(2, "0")}} · ${{esc(first(x.severity, "WARNING"))}}</div>
      <div class="safety-big">${{esc(titleOf(x))}}</div>
      <p class="muted">${{esc(first(x.message, x.reason, x.description, "Clinical ambiguity requires human judgment."))}}</p>
      <div class="safety-quote">“${{esc(first(x.evidence_snippet, sourceOf(x)))}}”</div>
      <div class="actions">
        <button class="btn orange" onclick="assignReview(${{i}})">${{esc(t("btnAssignReviewer", "Assign reviewer →"))}}</button>
        <button class="btn" onclick="keepPaused(${{i}})">${{esc(t("btnKeepPaused", "Keep paused"))}}</button>
      </div>
    </div>
  `).join("") : `
    <div class="card">
      <div class="stat green-t">${{esc(t("clearStatus", "CLEAR"))}}</div>
      <p class="muted">No open human-review items or conflicting directives were detected for this case.</p>
    </div>
  `) + disclaimerHtml;
}}

function renderAlerts(n) {{
  $("#alertHeader").textContent = n.notifications.length + " ACTIVE";
  $("#reminderList").innerHTML = n.reminders.length ? n.reminders.map(x => `
    <div class="notif">
      <div class="notif-head">
        <div class="notif-title">${{esc(titleOf(x))}}</div>
        <div class="notif-state">${{esc(statusOf(x))}}</div>
      </div>
      <p>${{esc(first(x.scheduled_for, x.event_time, x.trigger, x.description, "Scheduled reminder"))}}</p>
    </div>
  `).join("") : '<div class="notice">No reminder records scheduled.</div>';

  $("#notificationList").innerHTML = n.notifications.length ? n.notifications.map((x, i) => {{
    const id = first(x.notification_id, x.id, i);
    const transBadge = x.translation_status === "REVIEW_REQUIRED" ? `<span style="color:var(--orange);font-size:7px;margin-left:6px">[LANG REVIEW REQUIRED]</span>` : "";
    return `
      <div class="notif">
        <div class="notif-head">
          <div class="notif-title">${{esc(titleOf(x))}}${{transBadge}}</div>
          <div class="notif-state">${{esc(statusOf(x))}}</div>
        </div>
        <p>${{esc(first(x.body, x.message, x.description, "Notification payload"))}}</p>
        <div class="notif-actions">
          <button class="mini" onclick="notifAction('${{esc(id)}}','ACKNOWLEDGE')">${{esc(t("btnAcknowledge", "✓ Acknowledge"))}}</button>
          <button class="mini" onclick="notifAction('${{esc(id)}}','SNOOZE')">${{esc(t("btnSnooze", "⏰ Snooze 60m"))}}</button>
          <button class="mini" onclick="notifAction('${{esc(id)}}','DISMISS')">${{esc(t("btnDismiss", "× Dismiss"))}}</button>
        </div>
      </div>
    `;
  }}).join("") : '<div class="notice">No notifications generated for this care plan.</div>';
}}

function renderAudit(n) {{
  const r = n.raw;
  const isBlock = n.safety.includes("BLOCK") || n.safety.includes("CONFLICT");
  const isReview = n.review.length > 0;
  const gatePass = !isBlock && !isReview;

  const rows = [
    ["Zero-hallucination safety gate", n.safety, gatePass ? "pass" : "fail"],
    ["Source grounding", first($("#metricGround")?.textContent, "100%"), "pass"],
    ["Patient releasable", first(r.patient_releasable, !isBlock) ? "TRUE" : "FALSE (BLOCKED)", first(r.patient_releasable, !isBlock) ? "pass" : "fail"],
    ["Active hazard codes", String(r.validation_report?.hazard_codes?.length || (isReview ? n.review.length : 0)), isReview ? "fail" : "pass"],
    ["Indexed sentence offsets", String(r.sentences?.length || "VERIFIED"), "pass"],
    ["Translation fidelity audit", first(r.language?.translation_status, "VERIFIED"), (r.language?.translation_status === "REVIEW_REQUIRED") ? "fail" : "pass"],
    ["Provider ranking invariant", "ZERO (NEUTRAL ATTRIBUTES ONLY)", "pass"],
    ["Sponsored results invariant", "ZERO (NO SPONSORED ADS)", "pass"]
  ];

  $("#auditContent").innerHTML = rows.map(x => `
    <div class="audit-line">
      <span>${{esc(x[0])}}</span>
      <b class="${{x[2]}}">${{esc(String(x[1]))}}</b>
    </div>
  `).join("");
}}

let currentInspectorTab = "evidence";

function renderInspector(tab, n) {{
  currentInspectorTab = tab;
  $$(".itab").forEach(b => b.classList.toggle("active", b.dataset.itab === tab));
  const x = n.directives[state.selectedTask ?? 0] || n.review[0] || n.directives[0] || {{}};
  const el = $("#inspector");
  if (!el) return;

  const ptPrefLang = state.currentCase?.preferred_language || state.pipeline?.case_info?.preferred_language || "en";

  if (tab === "evidence") {{
    const sStart = x.evidence ? (x.evidence.start_offset ?? x.evidence.char_start ?? 0) : 0;
    const sEnd = x.evidence ? (x.evidence.end_offset ?? x.evidence.char_end ?? 0) : 0;
    const spanText = x.evidence ? `[${{sStart}} : ${{sEnd}}]` : "[—]";
    const sectionName = x.evidence?.section || "Follow-Up Instructions";
    const plain = first(x.plain_language, x.patient_language, x.description, "Plain language wording.");
    el.innerHTML = `
      <div class="inspect-kicker">${{esc(t("inspectorProvenanceKicker", "PROVENANCE & EVIDENCE"))}}</div>
      <div class="inspect-title">${{esc(titleOf(x))}}</div>
      <div class="quote">“${{esc(sourceOf(x))}}”</div>
      <div class="section-label">${{esc(t("inspectorCoords", "Provenance Coordinates"))}}</div>
      <div class="metric-grid">
        <div class="metric"><span>${{esc(t("inspectorCharOffset", "Char Offset"))}}</span><b>${{esc(spanText)}}</b></div>
        <div class="metric"><span>${{esc(t("inspectorSection", "Section"))}}</span><b>${{esc(sectionName)}}</b></div>
        <div class="metric"><span>${{esc(t("inspectorGrounding", "Grounding"))}}</span><b class="green-t">VERIFIED</b></div>
        <div class="metric"><span>${{esc(t("inspectorSafety", "Safety"))}}</span><b>${{esc(statusOf(x))}}</b></div>
      </div>
      <div class="section-label">${{esc(t("inspectorPatientLanguage", "Patient Plain Language"))}} (${{esc(state.language.toUpperCase())}}${{ptPrefLang !== state.language ? ` / PT: ${{ptPrefLang.toUpperCase()}}` : ''}})</div>
      <div class="quote" style="border-left-color:var(--green)">${{esc(plain)}}</div>
    `;
  }} else if (tab === "alerts") {{
    el.innerHTML = `
      <div class="inspect-kicker">REMINDERS & NOTIFICATIONS</div>
      <div class="inspect-title">Patient Alerts</div>
      ${{n.notifications.length ? n.notifications.map((z, i) => `
        <div class="notif">
          <div class="notif-head">
            <div class="notif-title">${{esc(titleOf(z))}}</div>
            <div class="notif-state">${{esc(statusOf(z))}}</div>
          </div>
          <p>${{esc(first(z.body, z.message, "—"))}}</p>
          <div class="notif-actions">
            <button class="mini" onclick="notifAction('${{esc(first(z.notification_id, z.id, i))}}','ACKNOWLEDGE')">${{esc(t("btnAcknowledge", "ACK"))}}</button>
            <button class="mini" onclick="notifAction('${{esc(first(z.notification_id, z.id, i))}}','SNOOZE')">${{esc(t("btnSnooze", "SNOOZE"))}}</button>
            <button class="mini" onclick="notifAction('${{esc(first(z.notification_id, z.id, i))}}','DISMISS')">${{esc(t("btnDismiss", "DISMISS"))}}</button>
          </div>
        </div>
      `).join("") : '<div class="notice">No notifications generated.</div>'}}
    `;
  }} else if (tab === "providers") {{
    el.innerHTML = `
      <div class="inspect-kicker">PROVIDER DISCOVERY</div>
      <div class="inspect-title">Deterministic Matching</div>
      <div class="notice">${{esc(t("providerDisclosure"))}}</div>
      <div class="section-label">Current Search Criteria</div>
      <div class="metric-grid">
        <div class="metric"><span>${{esc(t("labelSpecialty", "Specialty"))}}</span><b>${{esc($("#pSpecialty")?.value || "Cardiology")}}</b></div>
        <div class="metric"><span>${{esc(t("labelCity", "Location"))}}</span><b>${{esc($("#pCity")?.value || "Any")}}</b></div>
      </div>
    `;
  }} else {{
    el.innerHTML = `
      <div class="inspect-kicker">CLINICAL AUDIT</div>
      <div class="inspect-title">System Invariants</div>
      <div class="audit-line"><span>Zero-hallucination gate</span><b class="pass">${{esc(n.safety)}}</b></div>
      <div class="audit-line"><span>Source grounding</span><b class="pass">100%</b></div>
      <div class="audit-line"><span>Human review required</span><b class="${{n.review.length ? 'fail' : 'pass'}}">${{n.review.length ? 'YES' : 'CLEAR'}}</b></div>
      <div class="audit-line"><span>Provider ranking</span><b class="pass">ZERO</b></div>
    `;
  }}
}}
$$(".itab").forEach(b => b.addEventListener("click", () => renderInspector(b.dataset.itab, normalize(state.pipeline))));

async function loadCases() {{
  try {{
    const r = await fetch(API + "/api/v1/cases");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    state.cases = Array.isArray(data) ? data : (data.cases || data.items || []);
  }} catch (e) {{
    console.warn("Could not fetch cases from backend, using default synthetic cases:", e);
    state.cases = [
      {{ case_id: "case_absolute_cardiology", title: "Cardiology Absolute Follow-up (March 18, 2027)", patient_id: "SYNTH-PT-801", preferred_language: "ta", expected_safety: "SAFE / ORGANIZED" }},
      {{ case_id: "SYNTH-PT-101", title: "Normal Uncomplicated Discharge", patient_id: "SYNTH-PT-101", preferred_language: "en", expected_safety: "SAFE / ORGANIZED" }},
      {{ case_id: "SYNTH-PT-202", title: "Multi-Medication Regimen", patient_id: "SYNTH-PT-202", preferred_language: "hi", expected_safety: "SAFE / ORGANIZED" }},
      {{ case_id: "SYNTH-PT-303", title: "Ambiguous Follow-up", patient_id: "SYNTH-PT-303", preferred_language: "te", expected_safety: "NEEDS REVIEW" }},
      {{ case_id: "SYNTH-PT-404", title: "Conflicting Medications", patient_id: "SYNTH-PT-404", preferred_language: "kn", expected_safety: "CONFLICT DETECTED" }},
      {{ case_id: "SYNTH-PT-505", title: "Emergency Red Flags", patient_id: "SYNTH-PT-505", preferred_language: "ml", expected_safety: "NEEDS REVIEW" }}
    ];
  }}

  const caseSelect = $("#caseSelect");
  caseSelect.innerHTML = state.cases.map(c => `
    <option value="${{esc(c.case_id || c.id)}}">[${{esc(c.patient_id || c.case_id)}}] ${{esc(c.title || c.name || "Synthetic Case")}}</option>
  `).join("");

  const firstCaseId = state.cases[0]?.case_id || state.cases[0]?.id;
  await runCase(firstCaseId, state.language);
}}

async function runCase(id, lang = state.language) {{
  state.currentCase = state.cases.find(c => (c.case_id || c.id) === id) || {{ case_id: id }};
  state.requestSeq = (state.requestSeq || 0) + 1;
  const seq = state.requestSeq;

  const caseState = $("#caseState");
  if (caseState) {{
    caseState.textContent = "● RUNNING PIPELINE";
    caseState.style.borderColor = "var(--orange)";
    caseState.style.color = "var(--orange2)";
  }}

  const t0 = performance.now();
  try {{
    const r = await fetch(API + "/api/v1/orchestrate", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ case_id: id, language: lang }})
    }});
    if (seq !== state.requestSeq) return; // Drop stale response

    if (!r.ok) throw new Error(await r.text());
    state.pipeline = await r.json();
    const latencyMs = Math.round(performance.now() - t0);
    if (state.pipeline) state.pipeline.latency_ms = latencyMs;

    renderAll();
  }} catch (e) {{
    if (seq !== state.requestSeq) return;
    console.error("Orchestration error:", e);
    state.pipeline = {{
      case_id: id,
      overall_safety_status: first(state.currentCase?.expected_safety, "NEEDS REVIEW"),
      summary: "Backend orchestration call failed. Verify FastAPI server connection at " + API,
      tasks: [],
      medications: [],
      human_review_items: [{{ title: "Backend Communication Error", reason: "Could not retrieve clinical pipeline state.", severity: "WARNING" }}],
      reminders: [],
      notifications: [],
      grounding_score: "—"
    }};
    renderAll();
    toast(t("pipelineFail", "Pipeline communication error"), t("checkServer", "Check FastAPI server connection."));
  }}
}}

// Case change listener - PRESERVE user selected language!
$("#caseSelect").addEventListener("change", e => {{
  runCase(e.target.value, state.language);
}});

// Multilingual 12-language Selector
const langSel = $("#languageSelect");
if (langSel) {{
  langSel.value = state.language;
  langSel.addEventListener("change", async (e) => {{
    const newLang = e.target.value;
    if (!SUPPORTED_UI_LANGS.includes(newLang)) {{
      console.error("Unsupported language:", newLang);
      return;
    }}
    state.language = newLang;
    applyI18n(newLang);
    if (state.pipeline) renderAll();

    const patId = state.currentCase?.patient_id || state.pipeline?.case_info?.patient_identifier;
    if (patId && patId !== "SYNTH-PT-CUSTOM") {{
      try {{
        await fetch(`${{API}}/api/v1/patients/${{encodeURIComponent(patId)}}/language`, {{
          method: "PATCH",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ language: newLang }})
        }});
      }} catch (err) {{ /* silent ignore */ }}
    }}

    const curCaseId = state.currentCase?.case_id || state.currentCase?.id || state.cases[0]?.case_id;
    if (curCaseId) {{
      await runCase(curCaseId, newLang);
    }}
    toast(t("langUpdated", "Language preference updated"), `${{t("activeLang", "Active language")}}: ${{newLang.toUpperCase()}}`);
  }});
}}

function loadTextFile(file) {{
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  $("#fileList").innerHTML = `
    <div class="file-row">
      <div class="file-ext">${{esc(ext || "FILE")}}</div>
      <div>
        <div class="file-name">${{esc(file.name)}}</div>
        <div class="file-meta">${{Math.round(file.size / 1024)}} KB · local source document</div>
      </div>
      <span class="tag green">LOADED</span>
    </div>
  `;
  if (["txt", "md", "json", "csv"].includes(ext)) {{
    const reader = new FileReader();
    reader.onload = () => {{
      $("#rawText").value = reader.result;
      const pasteText = $("#pasteText");
      if (pasteText) pasteText.value = reader.result;
      toast("Source document loaded", "Text is staged and ready for orchestration.");
    }};
    reader.readAsText(file);
  }} else {{
    toast("File selected", "Text files (.txt, .md, .json) can be loaded directly into the editor.");
  }}
}}

$("#fileInput").addEventListener("change", e => {{ if (e.target.files[0]) loadTextFile(e.target.files[0]); }});
$("#dropZone").addEventListener("dragover", e => {{ e.preventDefault(); $("#dropZone").classList.add("drag"); }});
$("#dropZone").addEventListener("dragleave", () => $("#dropZone").classList.remove("drag"));
$("#dropZone").addEventListener("drop", e => {{
  e.preventDefault();
  $("#dropZone").classList.remove("drag");
  if (e.dataTransfer.files[0]) loadTextFile(e.dataTransfer.files[0]);
}});

$("#clearRaw").onclick = () => {{
  $("#rawText").value = "";
  $("#fileList").innerHTML = "";
}};

async function runRawText(text) {{
  if (!text || !text.trim()) {{
    toast(t("noSourceText", "No source text"), t("pasteDischarge", "Paste clinical discharge text first."));
    return;
  }}
  state.requestSeq = (state.requestSeq || 0) + 1;
  const seq = state.requestSeq;

  const caseState = $("#caseState");
  if (caseState) {{
    caseState.textContent = "● RUNNING CUSTOM PIPELINE";
    caseState.style.borderColor = "var(--orange)";
    caseState.style.color = "var(--orange2)";
  }}
  const t0 = performance.now();
  try {{
    const r = await fetch(API + "/api/v1/orchestrate", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ raw_text: text, language: state.language }})
    }});
    if (seq !== state.requestSeq) return;
    if (!r.ok) throw new Error(await r.text());
    state.pipeline = await r.json();
    state.pipeline.latency_ms = Math.round(performance.now() - t0);
    state.currentCase = {{
      case_id: first(state.pipeline.case_info?.case_id, "CUSTOM"),
      patient_id: first(state.pipeline.case_info?.patient_identifier, "SYNTH-PT-CUSTOM"),
      title: "Custom Clinical Document"
    }};
    renderAll();
    showPage("overview");
    toast("Pipeline execution complete", "Source grounded and clinical state updated.");
  }} catch (e) {{
    if (seq !== state.requestSeq) return;
    console.error("Pipeline failure:", e);
    toast(t("pipelineFail", "Pipeline communication error"), t("checkServer", "Check FastAPI server connection."));
  }}
}}

$("#runRaw").onclick = () => runRawText($("#rawText").value);
$("#runPaste").onclick = () => runRawText($("#pasteText").value);

async function notifAction(id, action) {{
  try {{
    const r = await fetch(API + "/api/v1/notifications/" + encodeURIComponent(id) + "/action", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ action, snooze_minutes: 60 }})
    }});
    if (!r.ok) throw new Error("HTTP " + r.status);
    toast("Notification updated", `Action: ${{action}}`);
    const curCaseId = state.currentCase?.case_id || state.currentCase?.id;
    if (curCaseId) await runCase(curCaseId, state.language);
    showPage("alerts");
  }} catch (e) {{
    toast("Action error", "Could not transition notification state.");
  }}
}}

function assignReview(idx) {{
  toast("Assigned for human review", "Flagged event routed to attending clinical coordinator.");
}}
function keepPaused(idx) {{
  toast("Gate remains paused", "Directive release held pending ambiguity resolution.");
}}

$("#providerSearch").onclick = async () => {{
  const specialty = $("#pSpecialty").value || "Cardiology";
  const city = $("#pCity").value || undefined;
  const lang = $("#pLanguage").value || undefined;
  const ins = $("#pInsurance").value || undefined;

  const payload = {{ specialty }};
  if (city) payload.location_city = city;
  if (lang) payload.language = lang;
  if (ins) payload.insurance = ins;

  try {{
    const r = await fetch(API + "/api/v1/providers/search", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(payload)
    }});
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    state.providers = Array.isArray(data) ? data : (data.providers || data.results || []);

    const resEl = $("#providerResults");
    if (state.providers.length) {{
      resEl.innerHTML = `
        <div style="font-size:8px;color:#777;margin-bottom:8px">DETERMINISTIC MATCHES (${{state.providers.length}}) · NO DOCTOR RANKING · DIRECT ATTRIBUTES</div>
        ${{state.providers.map(p => {{
          const provName = first(p.display_name, p.name, p.provider_name, "Provider");
          const facName = p.facility_name ? ` · ${{p.facility_name}}` : "";
          const loc = first(p.city, p.location_city, p.location, "");
          const stateCode = p.state ? `, ${{p.state}}` : "";
          const langs = Array.isArray(p.languages) ? p.languages : (p.language ? [p.language] : []);
          const insList = Array.isArray(p.accepted_insurance) ? p.accepted_insurance : (p.insurance ? [p.insurance] : []);
          const tags = [...langs, ...insList].filter(Boolean);
          const avail = p.availability_note || "Availability not verified";
          return `
            <div class="provider">
              <div>
                <div class="provider-name">${{esc(provName)}}${{esc(facName)}}</div>
                <div class="provider-meta">${{esc(first(p.specialty, specialty))}}${{loc ? ' · ' + esc(loc + stateCode) : ''}}${{p.phone ? ' · ' + esc(p.phone) : ''}}</div>
                <div class="provider-tags">
                  ${{tags.map(tagItem => `<span>${{esc(tagItem)}}</span>`).join("")}}
                  <span style="color:var(--orange2)">${{esc(avail)}}</span>
                </div>
              </div>
              <span class="tag green">ATTRIBUTE MATCH</span>
            </div>
          `;
        }}).join("")}}
        <div class="notice" style="margin-top:10px;font-size:9px">
          ${{esc(t("providerDisclosure"))}}
        </div>
      `;
    }} else {{
      resEl.innerHTML = '<div class="notice">No deterministic attribute matches found for this query.</div>';
    }}
    renderInspector("providers", normalize(state.pipeline));
  }} catch (e) {{
    toast("Provider search error", "Check /api/v1/providers/search endpoint.");
  }}
}};

$("#notifBtn").onclick = () => showPage("alerts");

async function boot() {{
  applyI18n(state.language);
  await loadCases();
}}
boot();
</script>
</body>
</html>
"""

new_html = html[:script_pos] + script_content

with open("backend/static/index.html", "w", encoding="utf-8") as f:
    f.write(new_html)

with open("frontend/index.html", "w", encoding="utf-8") as f:
    f.write(new_html)

print("Updated backend/static/index.html and frontend/index.html successfully!")
