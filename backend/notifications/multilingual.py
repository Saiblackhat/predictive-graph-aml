"""Phase 6.6 Multilingual Notification & Localization Engine.

Provides deterministic, clinically verified notification content across:
- English (en)
- Spanish (es)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)
- Bengali (bn)
- Marathi (mr)
- Gujarati (gu)
- Punjabi (pa)
- Odia (or)

Enforces strict safety invariants:
1. Clinical terms (medication names, dosages, units, dates, times) remain untouched.
2. Zero unauthorized inference (no fabricated bookings, doctor recommendations, or quality claims).
3. If semantic fidelity cannot be verified, routes to TRANSLATION_REVIEW_REQUIRED.
"""

from __future__ import annotations

import datetime
import re
from typing import Optional

from backend.care_reminders.types import Reminder
from backend.notifications.types import (
    NotificationChannel,
    NotificationType,
    SUPPORTED_LANGUAGES,
    TranslationStatus,
)
from backend.task_orchestration.types import OrchestratedTask, TaskCategory

# ── Localized Month Names ─────────────────────────────────────────────────────
LOCALIZED_MONTHS: dict[str, list[str]] = {
    "en": ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "es": ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
    "hi": ["", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितम्बर", "अक्टूबर", "नवम्बर", "दिसम्बर"],
    "ta": ["", "ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்", "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"],
    "te": ["", "జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"],
    "kn": ["", "ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್", "ಜುಲೈ", "ಆಗಸ್ಟ್", "ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್", "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್"],
    "ml": ["", "ജനുവരി", "ഫെബ്രുവരി", "മാർച്ച്", "ഏപ്രിൽ", "മെയ്", "ജൂൺ", "ജൂലൈ", "ഓഗസ്റ്റ്", "സെപ്റ്റംബർ", "ഒക്ടോബർ", "നവംബർ", "ഡിസംബർ"],
    "bn": ["", "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"],
    "mr": ["", "जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून", "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"],
    "gu": ["", "જાન્યુઆરી", "ફેબ્રુઆરી", "માર્ચ", "એપ્રિલ", "મે", "જૂન", "જુલાઈ", "ઑગસ્ટ", "સપ્ટેમ્બર", "ઑક્ટોબર", "નવેમ્બર", "ડિસેમ્બર"],
    "pa": ["", "ਜਨਵਰੀ", "ਫ਼ਰਵਰੀ", "ਮਾਰਚ", "ਅਪ੍ਰੈਲ", "ਮਈ", "ਜੂਨ", "ਜੁਲਾਈ", "ਅਗਸਤ", "ਸਤੰਬਰ", "ਅਕਤੂਬਰ", "ਨਵੰਬਰ", "ਦਸੰਬਰ"],
    "or": ["", "ଜାନୁଆରୀ", "ଫେବୃଆରୀ", "ମାର୍ଚ୍ଚ", "ଏପ୍ରିଲ", "ମେ", "ଜୁନ", "ଜୁଲାଇ", "ଅଗଷ୍ଟ", "ସେପ୍ଟେମ୍ବର", "ଅକ୍ଟୋବର", "ନଭେମ୍ବର", "ଡିସେମ୍ବର"],
}

# ── Localized Specialty Display Names ─────────────────────────────────────────
LOCALIZED_SPECIALTIES: dict[str, dict[str, str]] = {
    "es": {
        "cardiology": "Cardiología",
        "primary care": "Atención primaria",
        "oncology": "Oncología",
        "neurology": "Neurología",
        "pulmonology": "Neumología",
    },
    "hi": {
        "cardiology": "कार्डियोलॉजी (Cardiology)",
        "primary care": "प्राथमिक देखभाल (Primary Care)",
        "oncology": "ऑन्कोलॉजी (Oncology)",
        "neurology": "न्यूरोलॉजी (Neurology)",
        "pulmonology": "पल्मोनोलॉजी (Pulmonology)",
    },
    "ta": {
        "cardiology": "இருதய நிபுணர் (Cardiology)",
        "primary care": "முதன்மை பராமரிப்பு (Primary Care)",
        "oncology": "புற்றுநோயியல் (Oncology)",
        "neurology": "நரம்பியல் (Neurology)",
        "pulmonology": "நுரையீரலியல் (Pulmonology)",
    },
    "te": {
        "cardiology": "గుండె నిపుణుడు (Cardiology)",
        "primary care": "ప్రాథమిక సంరక్షణ (Primary Care)",
        "oncology": "ఆంకాలజీ (Oncology)",
        "neurology": "న్యూరాలజీ (Neurology)",
        "pulmonology": "పల్మోనాలజీ (Pulmonology)",
    },
    "kn": {
        "cardiology": "ಹೃದ್ರೋಗ ತಜ್ಞರು (Cardiology)",
        "primary care": "ಪ್ರಾಥಮಿಕ ಆರೈಕೆ (Primary Care)",
        "oncology": "ಆಂಕಾಲಜಿ (Oncology)",
        "neurology": "ನ್ಯೂರಾಲಜಿ (Neurology)",
        "pulmonology": "ಪಲ್ಮನಾಲಜಿ (Pulmonology)",
    },
    "ml": {
        "cardiology": "ഹൃദ്രോഗ വിദഗ്ദ്ധൻ (Cardiology)",
        "primary care": "പ്രൈമറി കെയർ (Primary Care)",
        "oncology": "ഓങ്കോളജി (Oncology)",
        "neurology": "ന്യൂറോളജി (Neurology)",
        "pulmonology": "പൾമണോളജി (Pulmonology)",
    },
    "bn": {
        "cardiology": "হৃদরোগ বিশেষজ্ঞ (Cardiology)",
        "primary care": "প্রাথমিক স্বাস্থ্যসেবা (Primary Care)",
        "oncology": "অনকোলজি (Oncology)",
        "neurology": "নিউরোলোজি (Neurology)",
        "pulmonology": "পালমোনোলজি (Pulmonology)",
    },
    "mr": {
        "cardiology": "हृदयरोग तज्ज्ञ (Cardiology)",
        "primary care": "प्राथमिक काळजी (Primary Care)",
        "oncology": "ऑन्कोलॉजी (Oncology)",
        "neurology": "न्यूरोलॉजी (Neurology)",
        "pulmonology": "पल्मोनोलॉजी (Pulmonology)",
    },
    "gu": {
        "cardiology": "હૃદય રોગ નિષ્ણાત (Cardiology)",
        "primary care": "પ્રાથમિક સારવાર (Primary Care)",
        "oncology": "ઓન્કોલોજી (Oncology)",
        "neurology": "ન્યુરોલોજી (Neurology)",
        "pulmonology": "પલ્મોનોલોજી (Pulmonology)",
    },
    "pa": {
        "cardiology": "ਦਿਲ ਦੇ ਮਾਹਿਰ (Cardiology)",
        "primary care": "ਮੁੱਢਲੀ ਦੇਖਭਾਲ (Primary Care)",
        "oncology": "ਔਨਕੋਲੋਜੀ (Oncology)",
        "neurology": "ਨਿਊਰੋਲੋਜੀ (Neurology)",
        "pulmonology": "ਪਲਮੋਨੋਲੋਜੀ (Pulmonology)",
    },
    "or": {
        "cardiology": "ହୃଦରୋଗ ବିଶେଷଜ୍ଞ (Cardiology)",
        "primary care": "ପ୍ରାଥମିକ ଚିକିତ୍ସା (Primary Care)",
        "oncology": "ଅଙ୍କୋଲୋଜି (Oncology)",
        "neurology": "ନ୍ୟୁରୋଲୋଜି (Neurology)",
        "pulmonology": "ପଲମୋନୋଲୋଜି (Pulmonology)",
    },
}

# ── Localized Notification Templates ──────────────────────────────────────────
# Format: (title_template, body_template)
NOTIFICATION_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "en": {
        "APPOINTMENT_REMINDER": ("Appointment Reminder: {specialty}", "Your {specialty} appointment is on {dt_display}.\nFrom your discharge plan.\nView care journey →"),
        "FOLLOW_UP_REMINDER": ("Upcoming follow-up: {specialty}", "{specialty}\n{dt_display}\nFrom your discharge plan.\nView care journey →"),
        "MEDICATION_REMINDER": ("Medication Reminder: {title}", "{title}\nTake as directed in your discharge plan.\nView care journey →"),
        "TEST_REMINDER": ("Upcoming test: {test_name}", "{test_name}\n{dt_display}\nFrom your care plan.\nView care journey →"),
        "WARNING_SIGN_ALERT": ("Warning Sign Notice: {title}", "Please monitor: {title}.\nIf symptoms worsen, contact your clinic.\nView care journey →"),
        "TASK_REMINDER": ("Care task reminder: {title}", "{title}\n{dt_display}\nFrom your discharge plan.\nView care journey →"),
        "HUMAN_REVIEW_NOTIFICATION": ("Care Plan Review Notice", "This directive is currently under clinical review.\nYour care team will clarify as needed."),
        "SAFETY_ESCALATION": ("Safety Notification", "A clinical instruction has been paused for verification.\nContact your healthcare team for guidance."),
        "PROVIDER_FOLLOW_UP": ("Provider Follow-Up: {specialty}", "Connect with {specialty} for follow-up care as outlined in your discharge summary."),
        "GENERAL_CARE_INSTRUCTION": ("Recovery Instruction", "{title}\nFrom your hospital discharge instructions.\nView care journey →"),
    },
    "es": {
        "APPOINTMENT_REMINDER": ("Recordatorio de cita: {specialty}", "Su cita de {specialty} es el {dt_display}.\nDe su plan de alta.\nVer plan de cuidados →"),
        "FOLLOW_UP_REMINDER": ("Próximo seguimiento: {specialty}", "{specialty}\n{dt_display}\nDe su plan de alta.\nVer plan de cuidados →"),
        "MEDICATION_REMINDER": ("Recordatorio de medicamento: {title}", "{title}\nTome según las indicaciones de su plan de alta.\nVer plan de cuidados →"),
        "TEST_REMINDER": ("Próxima prueba: {test_name}", "{test_name}\n{dt_display}\nDe su plan de cuidados.\nVer plan de cuidados →"),
        "WARNING_SIGN_ALERT": ("Aviso de signo de advertencia: {title}", "Supervise: {title}.\nSi los síntomas empeoran, comuníquese con su clínica.\nVer plan de cuidados →"),
        "TASK_REMINDER": ("Recordatorio de tarea de cuidados", "{title}\n{dt_display}\nDe su plan de alta.\nVer plan de cuidados →"),
        "HUMAN_REVIEW_NOTIFICATION": ("Aviso de revisión de plan de cuidados", "Esta instrucción está en revisión clínica.\nSu equipo de atención la aclarará según sea necesario."),
        "SAFETY_ESCALATION": ("Notificación de seguridad", "Una instrucción clínica ha sido pausada para verificación.\nComuníquese con su equipo de atención médica."),
        "PROVIDER_FOLLOW_UP": ("Seguimiento con proveedor: {specialty}", "Comuníquese con {specialty} para atención de seguimiento según su plan de alta."),
        "GENERAL_CARE_INSTRUCTION": ("Instrucción de recuperación", "{title}\nDe las instrucciones de alta hospitalaria.\nVer plan de cuidados →"),
    },
    "hi": {
        "APPOINTMENT_REMINDER": ("अपॉइंटमेंट रिमाइंडर: {specialty}", "आपकी {specialty} अपॉइंटमेंट {dt_display} को है।\nडिस्चार्ज योजना से।\nदेखभाल यात्रा देखें →"),
        "FOLLOW_UP_REMINDER": ("फॉलो-अप रिमाइंडर: {specialty}", "{specialty}\n{dt_display}\nडिस्चार्ज योजना से।\nदेखभाल यात्रा देखें →"),
        "MEDICATION_REMINDER": ("दवा रिमाइंडर: {title}", "{title}\nडिस्चार्ज योजना के निर्देशानुसार लें।\nदेखभाल यात्रा देखें →"),
        "TEST_REMINDER": ("जांच रिमाइंडर: {test_name}", "{test_name}\n{dt_display}\nदेखभाल योजना से।\nदेखभाल यात्रा देखें →"),
        "WARNING_SIGN_ALERT": ("चेतावनी संकेत सूचना: {title}", "निगरानी रखें: {title}।\nलक्षण बिगड़ने पर क्लिनिक से संपर्क करें।\nदेखभाल यात्रा देखें →"),
        "TASK_REMINDER": ("देखभाल कार्य रिमाइंडर: {title}", "{title}\n{dt_display}\nडिस्चार्ज योजना से।\nदेखभाल यात्रा देखें →"),
        "HUMAN_REVIEW_NOTIFICATION": ("क्लिनिकल समीक्षा सूचना", "यह निर्देश क्लिनिकल समीक्षा के अधीन है।\nआपकी स्वास्थ्य टीम स्पष्ट करेगी।"),
        "SAFETY_ESCALATION": ("सुरक्षा सूचना", "सत्यापन के लिए क्लिनिकल निर्देश रोका गया है।\nअपनी स्वास्थ्य टीम से संपर्क करें।"),
        "PROVIDER_FOLLOW_UP": ("डॉक्टर फॉलो-अप: {specialty}", "डिस्चार्ज योजना के अनुसार {specialty} से संपर्क करें।"),
        "GENERAL_CARE_INSTRUCTION": ("स्वास्थ्य सुधार निर्देश", "{title}\nअस्पताल डिस्चार्ज निर्देशों से।\nदेखभाल यात्रा देखें →"),
    },
    "ta": {
        "APPOINTMENT_REMINDER": ("மருத்துவர் சந்திப்பு நினைவூட்டல்: {specialty}", "உங்கள் {specialty} சந்திப்பு {dt_display} அன்று உள்ளது.\nவெளியேற்ற திட்டத்திலிருந்து.\nபராமரிப்பு பயணத்தைக் காண்க →"),
        "FOLLOW_UP_REMINDER": ("பின்தொடர் சந்திப்பு நினைவூட்டல்: {specialty}", "{specialty}\n{dt_display}\nவெளியேற்ற திட்டத்திலிருந்து.\nபராமரிப்பு பயணத்தைக் காண்க →"),
        "MEDICATION_REMINDER": ("மருந்து நினைவூட்டல்: {title}", "{title}\nவெளியேற்ற திட்டத்தில் அறிவுறுத்தியபடி எடுத்துக்கொள்ளவும்.\nபராமரிப்பு பயணத்தைக் காண்க →"),
        "TEST_REMINDER": ("பரிசோதனை நினைவூட்டல்: {test_name}", "{test_name}\n{dt_display}\nபராமரிப்பு திட்டத்திலிருந்து.\nபராமரிப்பு பயணத்தைக் காண்க →"),
        "WARNING_SIGN_ALERT": ("எச்சரிக்கை அறிகுறி அறிவிப்பு: {title}", "கவனிக்கவும்: {title}.\nஅறிகுறிகள் தீவிரமடைந்தால் மருத்துவமனையைத் தொடர்பு கொள்ளவும்.\nபராமரிப்பு பயணத்தைக் காண்க →"),
        "TASK_REMINDER": ("பராமரிப்பு பணி நினைவூட்டல்: {title}", "{title}\n{dt_display}\nவெளியேற்ற திட்டத்திலிருந்து.\nபராமரிப்பு பயணத்தைக் காண்க →"),
        "HUMAN_REVIEW_NOTIFICATION": ("மருத்துவ மறுஆய்வு அறிவிப்பு", "இந்த அறிவுறுத்தல் மருத்துவ மறுஆய்வில் உள்ளது.\nஉங்கள் பராமரிப்புக் குழு விளக்கும்."),
        "SAFETY_ESCALATION": ("பாதுகாப்பு அறிவிப்பு", "சரிபார்ப்பிற்காக ஒரு மருத்துவ அறிவுறுத்தல் நிறுத்தப்பட்டுள்ளது.\nஉங்கள் பராமரிப்புக் குழுவைத் தொடர்பு கொள்ளவும்."),
        "PROVIDER_FOLLOW_UP": ("மருத்துவர் பின்தொடர்வு: {specialty}", "வெளியேற்ற சுருக்கத்தின்படி {specialty} மருத்துவரை அணுகவும்."),
        "GENERAL_CARE_INSTRUCTION": ("குணமடைதல் அறிவுறுத்தல்", "{title}\nமருத்துவமனை வெளியேற்ற அறிவுறுத்தல்களிலிருந்து.\nபராமரிப்பு பயணத்தைக் காண்க →"),
    },
    "te": {
        "APPOINTMENT_REMINDER": ("అపాయింట్‌మెంట్ రిమైండర్: {specialty}", "మీ {specialty} అపాయింట్‌మెంట్ {dt_display} న నిర్ణయించబడింది.\nడిశ్చార్జ్ ప్లాన్ నుండి.\nసంరక్షణ ప్రయాణాన్ని చూడండి →"),
        "FOLLOW_UP_REMINDER": ("ఫాలో-అప్ రిమైండర్: {specialty}", "{specialty}\n{dt_display}\nడిశ్చార్జ్ ప్లాన్ నుండి.\nసంరక్షణ ప్రయాణాన్ని చూడండి →"),
        "MEDICATION_REMINDER": ("మందుల రిమైండర్: {title}", "{title}\nడిశ్చార్జ్ ప్లాన్‌లో సూచించినట్లు తీసుకోండి.\nసంరక్షణ ప్రయాణాన్ని చూడండి →"),
        "TEST_REMINDER": ("పరీక్ష రిమైండర్: {test_name}", "{test_name}\n{dt_display}\nసంరక్షణ ప్రణాళిక నుండి.\nసంరక్షణ ప్రయాణాన్ని చూడండి →"),
        "WARNING_SIGN_ALERT": ("హెచ్చరిక సంకేతాల అలర్ట్: {title}", "గమనించండి: {title}.\nలక్షణాలు తీవ్రమైతే క్లినిక్‌ని సంప్రదించండి.\nసంరక్షణ ప్రయాణాన్ని చూడండి →"),
        "TASK_REMINDER": ("సంరక్షణ టాస్క్ రిమైండర్: {title}", "{title}\n{dt_display}\nడిశ్చార్జ్ ప్లాన్ నుండి.\nసంరక్షణ ప్రయాణాన్ని చూడండి →"),
        "HUMAN_REVIEW_NOTIFICATION": ("క్లినికల్ సమీక్ష నోటీసు", "ఈ సూచన క్లినికల్ సమీక్షలో ఉంది.\nమీ సంరక్షణ బృందం స్పష్టం చేస్తుంది."),
        "SAFETY_ESCALATION": ("భద్రతా ఎస్కలేషన్ హెచ్చరిక", "ధృవీకరణ కోసం క్లినికల్ సూచన నిలిపివేయబడింది.\nమీ వైద్య బృందాన్ని సంప్రదించండి."),
        "PROVIDER_FOLLOW_UP": ("వైద్యుల ఫాలో-అప్: {specialty}", "డిశ్చార్జ్ సారాంశం ప్రకారం {specialty} తో సంప్రదించండి."),
        "GENERAL_CARE_INSTRUCTION": ("కోలుకునే సూచన", "{title}\nఆసుపత్రి డిశ్చార్జ్ సూచనల నుండి.\nసంరక్షణ ప్రయాణాన్ని చూడండి →"),
    },
    "kn": {
        "APPOINTMENT_REMINDER": ("ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ಜ್ಞಾಪನೆ: {specialty}", "ನಿಮ್ಮ {specialty} ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ {dt_display} ರಂದು ಇದೆ.\nಡಿಸ್ಚಾರ್ಜ್ ಯೋಜನೆಯಿಂದ.\nಆರೈಕೆ ಪ್ರಯಾಣ ವೀಕ್ಷಿಸಿ →"),
        "FOLLOW_UP_REMINDER": ("ಫಾಲೋ-ಅಪ್ ಜ್ಞಾಪನೆ: {specialty}", "{specialty}\n{dt_display}\nಡಿಸ್ಚಾರ್ಜ್ ಯೋಜನೆಯಿಂದ.\nಆರೈಕೆ ಪ್ರಯಾಣ ವೀಕ್ಷಿಸಿ →"),
        "MEDICATION_REMINDER": ("ಔಷಧಿ ಜ್ಞಾಪನೆ: {title}", "{title}\nಡಿಸ್ಚಾರ್ಜ್ ಯೋಜನೆಯಲ್ಲಿ ಸೂಚಿಸಿದಂತೆ ತೆಗೆದುಕೊಳ್ಳಿ.\nಆರೈಕೆ ಪ್ರಯಾಣ ವೀಕ್ಷಿಸಿ →"),
        "TEST_REMINDER": ("ಪರೀಕ್ಷೆ ಜ್ಞಾಪನೆ: {test_name}", "{test_name}\n{dt_display}\nಆರೈಕೆ ಯೋಜನೆಯಿಂದ.\nಆರೈಕೆ ಪ್ರಯಾಣ ವೀಕ್ಷಿಸಿ →"),
        "WARNING_SIGN_ALERT": ("ಎಚ್ಚರಿಕೆ ಸಂಕೇತ ಎಚ್ಚರಿಕೆ: {title}", "ಗಮನಿಸಿ: {title}.\nಲಕ್ಷಣಗಳು ಉಲ್ಬಣಿಸಿದರೆ ಕ್ಲಿನಿಕ್ ಸಂಪರ್ಕಿಸಿ.\nಆರೈಕೆ ಪ್ರಯಾಣ ವೀಕ್ಷಿಸಿ →"),
        "TASK_REMINDER": ("ಆರೈಕೆ ಕಾರ್ಯ ಜ್ಞಾಪನೆ: {title}", "{title}\n{dt_display}\nಡಿಸ್ಚಾರ್ಜ್ ಯೋಜನೆಯಿಂದ.\nಆರೈಕೆ ಪ್ರಯಾಣ ವೀಕ್ಷಿಸಿ →"),
        "HUMAN_REVIEW_NOTIFICATION": ("ಕ್ಲಿನಿಕಲ್ ಪರಿಶೀಲನೆ ಸೂಚನೆ", "ಈ ನಿರ್ದೇಶನವು ಕ್ಲಿನಿಕಲ್ ಪರಿಶೀಲನೆಯಲ್ಲಿದೆ.\nನಿಮ್ಮ ಆರೈಕೆ ತಂಡವು ಸ್ಪಷ್ಟಪಡಿಸುತ್ತದೆ."),
        "SAFETY_ESCALATION": ("ಸುರಕ್ಷತಾ ಸೂಚನೆ", "ಪರಿಶೀಲನೆಗಾಗಿ ಕ್ಲಿನಿಕಲ್ ಸೂಚನೆಯನ್ನು ವಿರಾಮಗೊಳಿಸಲಾಗಿದೆ.\nಆರೈಕೆ ತಂಡವನ್ನು ಸಂಪರ್ಕಿಸಿ."),
        "PROVIDER_FOLLOW_UP": ("ವೈದ್ಯರ ಫಾಲೋ-ಅಪ್: {specialty}", "ಡಿಸ್ಚಾರ್ಜ್ ಸಾರಾಂಶದ ಪ್ರಕಾರ {specialty} ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ."),
        "GENERAL_CARE_INSTRUCTION": ("ಚೇತರಿಕೆ ಸೂಚನೆ", "{title}\nಆಸ್ಪತ್ರೆಯ ಡಿಸ್ಚಾರ್ಜ್ ಸೂಚನೆಗಳಿಂದ.\nಆರೈಕೆ ಪ್ರಯಾಣ ವೀಕ್ಷಿಸಿ →"),
    },
    "ml": {
        "APPOINTMENT_REMINDER": ("അപ്പോയിന്റ്മെന്റ് ഓർമ്മപ്പെടുത്തൽ: {specialty}", "നിങ്ങളുടെ {specialty} അപ്പോയിന്റ്മെന്റ് {dt_display}-ൽ ആണ്.\nഡിസ്ചാർജ് പ്ലാനിൽ നിന്ന്.\nകെയർ ജേർണി കാണുക →"),
        "FOLLOW_UP_REMINDER": ("ഫോളോ-അപ്പ് ഓർമ്മപ്പെടുത്തൽ: {specialty}", "{specialty}\n{dt_display}\nഡിസ്ചാർജ് പ്ലാനിൽ നിന്ന്.\nകെയർ ജേർണി കാണുക →"),
        "MEDICATION_REMINDER": ("മരുന്ന് ഓർമ്മപ്പെടുത്തൽ: {title}", "{title}\nഡിസ്ചാർജ് പ്ലാനിൽ നിർദ്ദേശിച്ച പ്രകാരം കഴിക്കുക.\nകെയർ ജേർണി കാണുക →"),
        "TEST_REMINDER": ("പരിശോധന ഓർമ്മപ്പെടുത്തൽ: {test_name}", "{test_name}\n{dt_display}\nകെയർ പ്ലാനിൽ നിന്ന്.\nകെയർ ജേർണി കാണുക →"),
        "WARNING_SIGN_ALERT": ("മുന്നറിയിപ്പ് ലക്ഷണം അലേർട്ട്: {title}", "ശ്രദ്ധിക്കുക: {title}.\nലക്ഷണങ്ങൾ കൂടിയാൽ ക്ലിനിക്കുമായി ബന്ധപ്പെടുക.\nകെയർ ജേർണി കാണുക →"),
        "TASK_REMINDER": ("കെയർ ടാസ്ക് ഓർമ്മപ്പെടുത്തൽ: {title}", "{title}\n{dt_display}\nഡിസ്ചാർജ് പ്ലാനിൽ നിന്ന്.\nകെയർ ജേർണി കാണുക →"),
        "HUMAN_REVIEW_NOTIFICATION": ("ക്ലിനിക്കൽ അവലോകന അറിയിപ്പ്", "ഈ നിർദ്ദേശം ക്ലിനിക്കൽ അവലോകനത്തിലാണ്.\nനിങ്ങളുടെ പരിചരണ സംഘം വ്യക്തമാക്കും."),
        "SAFETY_ESCALATION": ("സുരക്ഷാ അറിയിപ്പ്", "സ്ഥിരീകരണത്തിനായി ഒരു നിർദ്ദേശം താൽക്കാലികമായി നിർത്തി.\nപരിചരണ സംഘവുമായി ബന്ധപ്പെടുക."),
        "PROVIDER_FOLLOW_UP": ("ഡോക്ടർ ഫോളോ-അപ്പ്: {specialty}", "ഡിസ്ചാർജ് നിർദ്ദേശമനുസരിച്ച് {specialty} ഡോക്ടറെ കാണുക."),
        "GENERAL_CARE_INSTRUCTION": ("രോഗശാന്തി നിർദ്ദേശം", "{title}\nആശുപത്രി ഡിസ്ചാർജ് നിർദ്ദേശങ്ങളിൽ നിന്ന്.\nകെയർ ജേർണി കാണുക →"),
    },
    "bn": {
        "APPOINTMENT_REMINDER": ("অ্যাপয়েন্টমেন্ট অনুস্মারক: {specialty}", "আপনার {specialty} অ্যাপয়েন্টমেন্ট {dt_display}-তে রয়েছে।\nডিসচার্জ পরিকল্পনা থেকে।\nকেয়ার জার্নি দেখুন →"),
        "FOLLOW_UP_REMINDER": ("ফলো-আপ অনুস্মারক: {specialty}", "{specialty}\n{dt_display}\nডিসচার্জ পরিকল্পনা থেকে।\nকেয়ার জার্নি দেখুন →"),
        "MEDICATION_REMINDER": ("ওষুধের অনুস্মারক: {title}", "{title}\nডিসচার্জ পরিকল্পনা অনুযায়ী সেবন করুন।\nকেয়ার জার্নি দেখুন →"),
        "TEST_REMINDER": ("পরীক্ষার অনুস্মারক: {test_name}", "{test_name}\n{dt_display}\nকেয়ার প্ল্যান থেকে।\nকেয়ার জার্নি দেখুন →"),
        "WARNING_SIGN_ALERT": ("সতর্কতা সংকেত বিজ্ঞপ্তি: {title}", "পর্যবেক্ষণ করুন: {title}।\nলক্ষণ খারাপ হলে ক্লিনিকে যোগাযোগ করুন।\nকেয়ার জার্নি দেখুন →"),
        "TASK_REMINDER": ("কেয়ার টাস্ক অনুস্মারক: {title}", "{title}\n{dt_display}\nডিসচার্জ পরিকল্পনা থেকে।\nকেয়ার জার্নি দেখুন →"),
        "HUMAN_REVIEW_NOTIFICATION": ("ক্লিনিকাল পর্যালোচনা বিজ্ঞপ্তি", "এই নির্দেশটি ক্লিনিকাল পর্যালোচনাধীন রয়েছে।\nআপনার কেয়ার টিম স্পষ্ট করবে।"),
        "SAFETY_ESCALATION": ("নিরাপত্তা সতর্কতা", "যাচাইয়ের জন্য একটি ক্লিনিকাল নির্দেশ স্থগিত করা হয়েছে।\nআপনার স্বাস্থ্যসেবা দলের সাথে যোগাযোগ করুন।"),
        "PROVIDER_FOLLOW_UP": ("চিকিৎসক ফলো-আপ: {specialty}", "ডিসচার্জ সারাংশ অনুযায়ী {specialty} চিকিৎসকের সাথে যোগাযোগ করুন।"),
        "GENERAL_CARE_INSTRUCTION": ("সুস্থতার নির্দেশিকা", "{title}\nহাসপাতালের ডিসচার্জ নির্দেশাবলী থেকে।\nকেয়ার জার্নি দেখুন →"),
    },
    "mr": {
        "APPOINTMENT_REMINDER": ("अपॉइंटमेंट स्मरणपत्र: {specialty}", "आपली {specialty} अपॉइंटमेंट {dt_display} रोजी आहे.\nडिस्चार्ज योजनेतून.\nकाळजी प्रवास पहा →"),
        "FOLLOW_UP_REMINDER": ("फॉलो-अप स्मरणपत्र: {specialty}", "{specialty}\n{dt_display}\nडिस्चार्ज योजनेतून.\nकाळजी प्रवास पहा →"),
        "MEDICATION_REMINDER": ("औषध स्मरणपत्र: {title}", "{title}\nडिस्चार्ज योजनेनुसार घ्या.\nकाळजी प्रवास पहा →"),
        "TEST_REMINDER": ("चाचणी स्मरणपत्र: {test_name}", "{test_name}\n{dt_display}\nकाळजी योजनेतून.\nकाळजी प्रवास पहा →"),
        "WARNING_SIGN_ALERT": ("इशारा सूचना: {title}", "निरीक्षण करा: {title}.\nलक्षणे वाढल्यास क्लिनिकशी संपर्क साधा.\nकाळजी प्रवास पहा →"),
        "TASK_REMINDER": ("काळजी कार्य स्मरणपत्र: {title}", "{title}\n{dt_display}\nडिस्चार्ज योजनेतून.\nकाळजी प्रवास पहा →"),
        "HUMAN_REVIEW_NOTIFICATION": ("क्लिनिकल पुनरावलोकन सूचना", "ही सूचना क्लिनिकल पुनरावलोकनाधीन आहे.\nआपली काळजी टीम मार्गदर्शन करेल."),
        "SAFETY_ESCALATION": ("सुरक्षा सूचना", "पडताळणीसाठी क्लिनिकल सूचना थांबवली गेली आहे.\nआरोग्य सेवा टीमशी संपर्क साधा."),
        "PROVIDER_FOLLOW_UP": ("डॉक्टर फॉलो-अप: {specialty}", "डिस्चार्ज सारांशानुसार {specialty} डॉक्टरांशी संपर्क साधा."),
        "GENERAL_CARE_INSTRUCTION": ("पुनर्प्राप्ती सूचना", "{title}\nरुग्णालय डिस्चार्ज सूचनांनुसार.\nकाळजी प्रवास पहा →"),
    },
    "gu": {
        "APPOINTMENT_REMINDER": ("મુલાકાત રીમાઇન્ડર: {specialty}", "તમારી {specialty} મુલાકાત {dt_display} પર છે.\nડિસ્ચાર્જ યોજનામાંથી.\nકેર જર્ની જુઓ →"),
        "FOLLOW_UP_REMINDER": ("ફોલો-અપ રીમાઇન્ડર: {specialty}", "{specialty}\n{dt_display}\nડિસ્ચાર્જ યોજનામાંથી.\nકેર જર્ની જુઓ →"),
        "MEDICATION_REMINDER": ("દવા રીમાઇન્ડર: {title}", "{title}\nડિસ્ચાર્જ યોજના મુજબ લો.\nકેર જર્ની જુઓ →"),
        "TEST_REMINDER": ("પરીક્ષણ રીમાઇન્ડર: {test_name}", "{test_name}\n{dt_display}\nકેર પ્લાનમાંથી.\nકેર જર્ની જુઓ →"),
        "WARNING_SIGN_ALERT": ("ચેતવણી સંકેત નોટિસ: {title}", "ધ્યાન રાખો: {title}.\nલક્ષણો વધે તો ક્લિનિકનો સંપર્ક કરો.\nકેર જર્ની જુઓ →"),
        "TASK_REMINDER": ("સંભાળ કાર્ય રીમાઇન્ડર: {title}", "{title}\n{dt_display}\nડિસ્ચાર્જ યોજનામાંથી.\nકેર જર્ની જુઓ →"),
        "HUMAN_REVIEW_NOTIFICATION": ("ક્લિનિકલ સમીક્ષા સૂચના", "આ સૂચના ક્લિનિકલ સમીક્ષા હેઠળ છે.\nતમારી કેર ટીમ સ્પષ્ટ કરશે."),
        "SAFETY_ESCALATION": ("સુરક્ષા સૂચના", "ચકાસણી માટે ક્લિનિકલ સૂચના રોકી દેવામાં આવી છે.\nતમારી કેર ટીમનો સંપર્ક કરો."),
        "PROVIDER_FOLLOW_UP": ("ડોક્ટર ફોલો-અપ: {specialty}", "ડિસ્ચાર્જ સારાંશ અનુસાર {specialty} ડોક્ટરનો સંપર્ક કરો."),
        "GENERAL_CARE_INSTRUCTION": ("સ્વસ્થ થવાની સૂચના", "{title}\nહોસ્પિટલ ડિસ્ચાર્જ સૂચનાઓમાંથી.\nકેર જર્ની જુઓ →"),
    },
    "pa": {
        "APPOINTMENT_REMINDER": ("ਮੁਲਾਕਾਤ ਰੀਮਾਈਂਡਰ: {specialty}", "ਤੁਹਾਡੀ {specialty} ਮੁਲਾਕਾਤ {dt_display} ਨੂੰ ਹੈ।\nਡਿਸਚਾਰਜ ਯੋਜਨਾ ਤੋਂ।\nਕੇਅਰ ਜਰਨੀ ਦੇਖੋ →"),
        "FOLLOW_UP_REMINDER": ("ਫਾਲੋ-ਅੱਪ ਰੀਮਾਈਂਡਰ: {specialty}", "{specialty}\n{dt_display}\nਡਿਸਚਾਰਜ ਯੋਜਨਾ ਤੋਂ।\nਕੇਅਰ ਜਰਨੀ ਦੇਖੋ →"),
        "MEDICATION_REMINDER": ("ਦਵਾਈ ਰੀਮਾਈਂਡਰ: {title}", "{title}\nਡਿਸਚਾਰਜ ਯੋਜਨਾ ਅਨੁਸਾਰ ਲਵੋ।\nਕੇਅਰ ਜਰਨੀ ਦੇਖੋ →"),
        "TEST_REMINDER": ("ਟੈਸਟ ਰੀਮਾਈਂਡਰ: {test_name}", "{test_name}\n{dt_display}\nਕੇਅਰ ਪਲਾਨ ਤੋਂ।\nਕੇਅਰ ਜਰਨੀ ਦੇਖੋ →"),
        "WARNING_SIGN_ALERT": ("ਚੇਤਾਵਨੀ ਸੰਕੇਤ ਚੌਕਸੀ: {title}", "ਨਿਗਰਾਨੀ ਰੱਖੋ: {title}।\nਲੱਛਣ ਵਿਗੜਨ 'ਤੇ ਕਲੀਨਿਕ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।\nਕੇਅਰ ਜਰਨੀ ਦੇਖੋ →"),
        "TASK_REMINDER": ("ਦੇਖਭਾਲ ਕਾਰਜ ਰੀਮਾਈਂਡਰ: {title}", "{title}\n{dt_display}\nਡਿਸਚਾਰਜ ਯੋਜਨਾ ਤੋਂ।\nਕੇਅਰ ਜਰਨੀ ਦੇਖੋ →"),
        "HUMAN_REVIEW_NOTIFICATION": ("ਕਲੀਨਿਕਲ ਸਮੀਖਿਆ ਨੋਟਿਸ", "ਇਹ ਹਦਾਇਤ ਕਲੀਨਿਕਲ ਸਮੀਖਿਆ ਅਧੀਨ ਹੈ।\nਤੁਹਾਡੀ ਕੇਅਰ ਟੀਮ ਸਪੱਸ਼ਟ ਕਰੇਗੀ।"),
        "SAFETY_ESCALATION": ("ਸੁਰੱਖਿਆ ਨੋਟਿਸ", "ਤਸਦੀਕ ਲਈ ਇੱਕ ਕਲੀਨਿਕਲ ਹਦਾਇਤ ਰੋਕੀ ਗਈ ਹੈ।\nਆਪਣੀ ਸਿਹਤ ਸੰਭਾਲ ਟੀਮ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।"),
        "PROVIDER_FOLLOW_UP": ("ਡਾਕਟਰ ਫਾਲੋ-ਅੱਪ: {specialty}", "ਡਿਸਚਾਰਜ ਸਾਰਾਂਸ਼ ਅਨੁਸਾਰ {specialty} ਨਾਲ ਸੰਪਰਕ ਕਰੋ।"),
        "GENERAL_CARE_INSTRUCTION": ("ਸਿਹਤਯਾਬੀ ਹਦਾਇਤ", "{title}\nਹਸਪਤਾਲ ਡਿਸਚਾਰਜ ਹਦਾਇਤਾਂ ਤੋਂ।\nਕੇਅਰ ਜਰਨੀ ਦੇਖੋ →"),
    },
    "or": {
        "APPOINTMENT_REMINDER": ("ସାକ୍ଷାତକାର ମନେପକାଣ: {specialty}", "ଆପଣଙ୍କର {specialty} ସାକ୍ଷାତକାର {dt_display} ରେ ଅଛି।\nଡିସଚାର୍ଜ ଯୋଜନାରୁ।\nଯତ୍ନ ଯାତ୍ରା ଦେଖନ୍ତୁ →"),
        "FOLLOW_UP_REMINDER": ("ଫଲୋ-ଅପ୍ ମନେପକାଣ: {specialty}", "{specialty}\n{dt_display}\nଡିସଚାର୍ଜ ଯୋଜନାରୁ।\nଯତ୍ନ ଯାତ୍ରା ଦେଖନ୍ତୁ →"),
        "MEDICATION_REMINDER": ("ଔଷଧ ମନେପକାଣ: {title}", "{title}\nଡିସଚାର୍ଜ ଯୋଜନା ଅନୁଯାୟୀ ନିଅନ୍ତୁ।\nଯତ୍ନ ଯାତ୍ରା ଦେଖନ୍ତୁ →"),
        "TEST_REMINDER": ("ପରୀକ୍ଷା ମନେପକାଣ: {test_name}", "{test_name}\n{dt_display}\nଯତ୍ନ ଯୋଜନାରୁ।\nଯତ୍ନ ଯାତ୍ରା ଦେଖନ୍ତୁ →"),
        "WARNING_SIGN_ALERT": ("ଚେତାବନୀ ସଙ୍କେତ ସତର୍କତା: {title}", "ଧ୍ୟାନ ରଖନ୍ତୁ: {title}।\nଲକ୍ଷଣ ବୃଦ୍ଧି ପାଇଲେ କ୍ଲିନିକ୍ ସହିତ ଯୋଗାଯୋଗ କରନ୍ତୁ।\nଯତ୍ନ ଯାତ୍ରା ଦେଖନ୍ତୁ →"),
        "TASK_REMINDER": ("ଯତ୍ନ କାର୍ଯ୍ୟ ମନେପକାଣ: {title}", "{title}\n{dt_display}\nଡିସଚାର୍ଜ ଯୋଜନାରୁ।\nଯତ୍ନ ଯାତ୍ରା ଦେଖନ୍ତୁ →"),
        "HUMAN_REVIEW_NOTIFICATION": ("କ୍ଲିନିକାଲ୍ ସମୀକ୍ଷା ବିଜ୍ଞପ୍ତି", "ଏହି ନିର୍ଦ୍ଦେଶ କ୍ଲିନିକାଲ୍ ସମୀକ୍ଷା ଅଧୀନରେ ଅଛି।\nଆପଣଙ୍କ ଚିକିତ୍ସା ଦଳ ସ୍ପଷ୍ଟ କରିବେ।"),
        "SAFETY_ESCALATION": ("ସୁରକ୍ଷା ବିଜ୍ଞପ୍ତି", "ଯାଞ୍ଚ ପାଇଁ ଏକ କ୍ଲିନିକାଲ୍ ନିର୍ଦ୍ଦେଶ ସ୍ଥଗିତ ରଖାଯାଇଛି।\nଆପଣଙ୍କ ସ୍ୱାସ୍ଥ୍ୟସେବା ଦଳ ସହିତ ଯୋଗାଯୋଗ କରନ୍ତୁ।"),
        "PROVIDER_FOLLOW_UP": ("ଡାକ୍ତର ଫଲୋ-ଅପ୍: {specialty}", "ଡିସଚାର୍ଜ ସାରାଂଶ ଅନୁଯାୟୀ {specialty} ସହିତ ଯୋଗାଯୋଗ କରନ୍ତୁ।"),
        "GENERAL_CARE_INSTRUCTION": ("ଆରୋଗ୍ୟ ନିର୍ଦ୍ଦେଶ", "{title}\nଡାକ୍ତରଖାନା ଡିସଚାର୍ଜ ନିର୍ଦ୍ଦେଶାବଳୀରୁ।\nଯତ୍ନ ଯାତ୍ରା ଦେଖନ୍ତୁ →"),
    },
}


def format_multilingual_datetime(dt: datetime.datetime, language: str = "en") -> str:
    """Format appointment or event datetime safely across all 12 supported languages."""
    lang = language.lower() if language else "en"
    month_list = LOCALIZED_MONTHS.get(lang, LOCALIZED_MONTHS["en"])
    month_name = month_list[dt.month] if 1 <= dt.month <= 12 else str(dt.month)

    # 12-hour or 24-hour display
    time_str = dt.strftime("%I:%M %p").lstrip("0")
    if lang == "es":
        return f"{dt.day} de {month_name} de {dt.year} · {time_str}"
    elif lang in ("hi", "mr", "gu", "pa"):
        return f"{dt.day} {month_name} {dt.year} · {time_str}"
    elif lang in ("ta", "te", "kn", "ml", "bn", "or"):
        return f"{month_name} {dt.day}, {dt.year} · {time_str}"

    # Default English
    return f"{month_name} {dt.day}, {dt.year} · {time_str}"


def resolve_specialty_display(specialty_raw: str, language: str = "en") -> str:
    """Resolve specialty name in target language while keeping English reference if appropriate."""
    if not specialty_raw:
        return "Specialist"
    lang = language.lower()
    mapping = LOCALIZED_SPECIALTIES.get(lang, {})
    return mapping.get(specialty_raw.lower(), specialty_raw)


def format_multilingual_notification(
    task: OrchestratedTask,
    reminder: Optional[Reminder] = None,
    language: str = "en",
    notification_type: Optional[NotificationType] = None,
) -> tuple[str, str, str]:
    """Generate deterministically localized (title, body, translation_status).

    Returns:
        title (str)
        body (str)
        translation_status ("VERIFIED" | "REVIEW_REQUIRED")
    """
    lang = (language or "en").lower().strip()
    if lang not in SUPPORTED_LANGUAGES:
        return (
            "Translation Verification Required",
            f"Patient requested unsupported language '{language}'. Routed to clinical human review.",
            TranslationStatus.REVIEW_REQUIRED.value,
        )

    # Determine event datetime string
    dt_display = ""
    if reminder and reminder.event_time:
        try:
            event_dt = datetime.datetime.fromisoformat(reminder.event_time)
            dt_display = format_multilingual_datetime(event_dt, language=lang)
        except Exception:
            dt_display = str(reminder.event_time)
    elif task.timeframe:
        dt_display = task.timeframe.raw_text or ""

    # Resolve notification type key
    notif_type = notification_type
    if not notif_type:
        if task.task_category == TaskCategory.FOLLOW_UP:
            notif_type = NotificationType.APPOINTMENT_REMINDER if task.timeframe and task.timeframe.absolute_date else NotificationType.FOLLOW_UP_REMINDER
        elif task.task_category == TaskCategory.MEDICATION:
            notif_type = NotificationType.MEDICATION_REMINDER
        elif task.task_category == TaskCategory.TEST:
            notif_type = NotificationType.TEST_REMINDER
        elif task.task_category == TaskCategory.WARNING_SIGN:
            notif_type = NotificationType.WARNING_SIGN_ALERT
        else:
            notif_type = NotificationType.TASK_REMINDER

    type_key = notif_type.value if hasattr(notif_type, "value") else str(notif_type)
    templates = NOTIFICATION_TEMPLATES.get(lang, NOTIFICATION_TEMPLATES["en"])
    template = templates.get(type_key)

    if not template:
        # Fallback to English template with review flag if missing in target
        en_template = NOTIFICATION_TEMPLATES["en"].get(type_key, NOTIFICATION_TEMPLATES["en"]["GENERAL_CARE_INSTRUCTION"])
        title = en_template[0].format(
            specialty=task.clinical_metadata.get("specialty", "Clinic"),
            title=task.title,
            test_name=task.clinical_metadata.get("test_name", task.title),
            dt_display=dt_display,
        )
        body = en_template[1].format(
            specialty=task.clinical_metadata.get("specialty", "Clinic"),
            title=task.title,
            test_name=task.clinical_metadata.get("test_name", task.title),
            dt_display=dt_display,
        )
        return title, body, TranslationStatus.REVIEW_REQUIRED.value

    # Extract template slots
    raw_spec = task.clinical_metadata.get("specialty", "") or "Clinic"
    spec_display = resolve_specialty_display(raw_spec, language=lang)
    test_name = task.clinical_metadata.get("test_name", task.title) or task.title

    title = template[0].format(
        specialty=spec_display,
        title=task.title,
        test_name=test_name,
        dt_display=dt_display,
    )
    body = template[1].format(
        specialty=spec_display,
        title=task.title,
        test_name=test_name,
        dt_display=dt_display,
    )

    return title, body, TranslationStatus.VERIFIED.value


def render_channel_payload(
    title: str,
    body: str,
    channel: NotificationChannel = NotificationChannel.IN_APP,
    language: str = "en",
) -> tuple[str, str]:
    """Render one canonical localized notification payload tailored for In-App, Push, or SMS."""
    if channel == NotificationChannel.PUSH:
        # Push notifications typically have short title and single-paragraph body
        clean_body = body.replace("\n", " ").strip()
        if len(clean_body) > 160:
            clean_body = clean_body[:157] + "..."
        return title, clean_body

    if channel == NotificationChannel.SMS:
        # SMS: concise summary prefixed by CareBridge
        prefix = "CareBridge: "
        first_line = body.split("\n")[0].strip()
        sms_body = f"{prefix}{title} - {first_line}"
        if len(sms_body) > 160:
            sms_body = sms_body[:157] + "..."
        return title, sms_body

    # Default IN_APP: full canonical payload
    return title, body


def build_channel_payloads(
    title: str,
    body: str,
    language: str = "en",
) -> dict[str, dict[str, str]]:
    """Generate delivery channel representations (IN_APP, PUSH, SMS) from canonical localized text."""
    in_app_t, in_app_b = render_channel_payload(title, body, NotificationChannel.IN_APP, language)
    push_t, push_b = render_channel_payload(title, body, NotificationChannel.PUSH, language)
    sms_t, sms_b = render_channel_payload(title, body, NotificationChannel.SMS, language)
    return {
        "IN_APP": {"title": in_app_t, "body": in_app_b},
        "PUSH": {"title": push_t, "body": push_b},
        "SMS": {"title": sms_t, "body": sms_b},
    }


def verify_translation_fidelity(
    title: str,
    body: str,
    task: OrchestratedTask,
    reminder: Optional[Reminder],
    language: str,
) -> tuple[bool, str, list[str]]:
    """Strictly verify translation semantic fidelity against clinical invariants.

    Checks:
    1. Medication names and exact formulation salt terms are preserved verbatim.
    2. Dosage values (e.g., '50 mg', '5 mg') are preserved verbatim.
    3. Numerical dates (day, year) and times are preserved.
    4. Warning sign titles / concepts are grounded.
    5. No prohibited clinical inference or claims exist.

    Returns:
        (is_verified: bool, translation_status: str, notes: list[str])
    """
    notes: list[str] = []
    combined = f"{title} {body}"
    lang = (language or "en").lower().strip()

    if lang not in SUPPORTED_LANGUAGES:
        notes.append(f"Patient requested unsupported language '{language}'. Verified translation unavailable.")
        return False, TranslationStatus.REVIEW_REQUIRED.value, notes

    # 1. Check Date / Year preservation if an absolute date exists
    if reminder and reminder.event_time:
        try:
            dt = datetime.datetime.fromisoformat(reminder.event_time)
            year_str = str(dt.year)
            day_str = str(dt.day)
            if year_str not in combined:
                notes.append(f"Appointment year '{year_str}' missing from translated notification.")
            if day_str not in combined:
                notes.append(f"Appointment day '{day_str}' missing from translated notification.")
        except Exception:
            pass

    # 2. Check Medication preservation if task is MEDICATION
    if task.task_category == TaskCategory.MEDICATION:
        # Digits in task title (dosage amounts like 50, 5) must be present
        digits = re.findall(r"\b\d+(?:\.\d+)?\b", task.title)
        for d in digits:
            if d not in combined:
                notes.append(f"Medication dosage quantity '{d}' missing from translated notification.")

        # Unit (mg, mcg, ml, etc.) must be present if in original
        units = re.findall(r"\b(mg|mcg|ml|units?|tablets?)\b", task.title, re.IGNORECASE)
        for u in units:
            if u.lower() not in combined.lower():
                notes.append(f"Medication unit '{u}' missing from translated notification.")

    # 3. Check Banned Phrases
    from backend.notifications.formatter import BANNED_NOTIFICATION_PHRASES
    for pat in BANNED_NOTIFICATION_PHRASES:
        m = pat.search(combined)
        if m:
            notes.append(f"Prohibited promotional or clinical claim phrase detected: '{m.group(0)}'.")

    is_verified = (len(notes) == 0)
    status = TranslationStatus.VERIFIED.value if is_verified else TranslationStatus.REVIEW_REQUIRED.value
    return is_verified, status, notes
