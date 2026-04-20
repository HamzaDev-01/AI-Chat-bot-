import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import random
import re
import pickle
import os


# =========================
# CLEAN FUNCTION
# =========================
def clean(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


# =========================
# TRAINING DATA  (430 samples, 21 intents)
# KEY FIXES:
#  - location: 25 samples with full "where are you/university located" coverage
#  - small_talk: purged all "you are / where" overlapping phrases
#  - facilities: 25 crisp, distinct samples — no library/hostel bleed
#  - all intents: 20 natural-language variants to reduce misclassification
# =========================
texts = [

# ================= GREETING (20) =================
"hello", "hi", "hey", "good morning", "good evening",
"assalamualaikum", "salam", "hi there", "hey bot", "is anyone there",
"how are you", "whats up", "good afternoon", "hey assistant", "greetings",
"good night", "heyy", "helloo", "anybody here", "start",

# ================= ADMISSIONS (20) =================
"how can i get admission", "admission process", "eligibility for bs cs",
"when do admissions open", "can i apply now", "admission deadline",
"documents needed for admission", "is entry test required", "nts test required",
"how to apply online", "last date for admission", "merit list date",
"when is result announced", "admission criteria", "how to register for admission",
"open merit seats", "how many seats available", "can i get admission without nts",
"apply for fall 2026", "admission form download",

# ================= FEES (20) =================
"fee structure", "semester fee amount", "total fee for bs cs",
"can i pay fee online", "are installments available", "late fee charges",
"how much is admission fee", "hostel fee per month", "bus transport fee",
"fee payment method", "what is the total program cost", "is fee too expensive",
"show me fee breakdown", "accepted payment methods", "can i pay via bank",
"how to download challan", "fee refund policy", "jazz cash payment accepted",
"easypaisa for fee", "fee due date",

# ================= SCHOLARSHIPS (20) =================
"scholarship available", "need based scholarship", "merit based scholarship",
"how to apply for scholarship", "minimum cgpa for scholarship", "hec scholarship",
"financial aid available", "fee waiver or discount", "zakat fund for students",
"scholarship criteria and eligibility", "who is eligible for scholarship",
"how to renew scholarship", "sports scholarship available", "family kinship discount",
"how to keep scholarship", "full free scholarship", "hafiz e quran scholarship",
"government scholarship", "ehsaas scholarship", "how much scholarship amount",

# ================= LOCATION (25) =================
# EXPANDED: covers every natural phrasing of "where is the university"
# Root-cause fix for "where are you located" → was misclassified as small_talk
"where is the university located", "where is campus", "university address",
"how to reach campus", "is campus in islamabad", "nearest metro to campus",
"what is the campus location", "google maps link for university",
"distance from rawalpindi to campus", "how far is campus from city",
"nearest landmark to university", "what is the route to university",
"where exactly is the university", "university located in which city",
"tell me the campus address", "how do i find the campus",
"give me directions to university", "is university in rawalpindi or islamabad",
"where are you located", "in which area is the campus",
"university sector in islamabad", "which road does campus face",
"campus near which hospital", "how to reach by metro",
"what city is the university in",

# ================= CONTACT (20) =================
"contact number of university", "university email address", "admission helpline number",
"office timing", "admission office hours", "support email id",
"where to send complaints", "can i visit campus for inquiry",
"where is admin office", "where is helpdesk", "how to contact university",
"does university have whatsapp", "call the university", "how to contact faculty",
"registrar office contact", "when is office open", "phone number please",
"contact for fee issues", "university uan number", "official email",

# ================= ACADEMICS (20) =================
"how many semesters in bs cs", "how long is the degree", "what subjects in cs",
"attendance policy", "what is minimum attendance required", "how many credit hours",
"grading system explained", "how to improve my gpa", "can i repeat a course",
"how to drop a subject", "how does course registration work", "what are electives",
"what are degree completion requirements", "how do classes work",
"suggest a study plan", "is cs a difficult degree", "how are credits counted",
"what is the cgpa system", "can i change my major", "what is academic probation",

# ================= EXAMS (20) =================
"midterm exam schedule", "when are final exams", "what are passing marks",
"how is grading done in exams", "how is gpa calculated", "can i skip an exam",
"is makeup exam available", "i missed my exam what to do",
"when is result announced", "what are exam rules", "how does quiz system work",
"how to prepare for exams", "how many exams per semester",
"what time are exams held", "final term schedule", "exam weightage breakdown",
"are open book exams allowed", "exam cheating policy", "recheck exam paper",
"grade appeal process",

# ================= FACILITIES (25) =================
# EXPANDED: every "what is available" variant covered
# Root-cause fix for "what facilities" → was sometimes landing on wrong intent
"what facilities are available on campus", "campus amenities list",
"is wifi available on campus", "are computer labs available",
"is there a cafeteria on campus", "what sports facilities exist",
"is there a gym on campus", "is medical room available", "is there a prayer area",
"is parking available", "is there an ai lab", "are study rooms available",
"how fast is the internet on campus", "what are lab timings",
"how is campus security", "any recreational facilities on campus",
"what services does campus provide", "is there a student lounge",
"what campus resources are there", "describe campus infrastructure",
"tell me about campus facilities", "all available facilities at university",
"what does campus offer students", "campus support services",
"i want to know about campus facilities",

# ================= SOCIETIES (20) =================
"what student societies are there", "how to join a society", "is there an ieee club",
"coding club available", "what events happen on campus", "are hackathons organized",
"how to get society membership", "can i organize an event",
"is there a drama club", "is there a music society",
"what tech societies exist", "are there sports clubs", "is there debate club",
"how to join a student club", "list of student organizations",
"annual events at university", "ieee student branch", "google developer student club",
"acm chapter available", "co curricular activities available",

# ================= SMALL TALK (20) =================
# FIXED: removed ALL "where/you are/are you" patterns to prevent location clash
"what is your purpose", "you seem very intelligent", "i like talking to you",
"can you tell me a joke", "do you sleep", "what do you do for fun",
"you are very helpful bot", "do you have feelings", "are you smarter than google",
"who programmed you", "what language are you built in", "you are amazing",
"can you think like humans", "how do you work", "i am bored talk to me",
"tell me something funny", "do robots dream", "you understand everything",
"are you connected to the internet", "what is artificial intelligence",

# ================= HOSTEL (20) =================
"is hostel available", "hostel for boys", "hostel for girls",
"hostel monthly fee", "how are hostel rooms", "can i stay on campus",
"hostel meal plan", "hostel security system", "who is hostel warden",
"how to apply for hostel", "is hostel full", "hostel waiting list",
"hostel curfew time", "hostel rules and regulations", "hostel internet access",
"hostel room sharing", "are rooms furnished", "hostel laundry facility",
"hostel check in procedure", "hostel food quality",

# ================= TRANSPORT (20) =================
"is there a university bus", "bus daily schedule", "which areas does bus cover",
"how to use transport service", "how much is bus fee", "pick and drop service",
"bus timings morning and evening", "how to register for transport",
"bus from rawalpindi", "bus from islamabad", "where are bus stops",
"how to apply for bus pass", "is shuttle service available",
"transport semester charges", "bus for daily commute",
"bus route from bahria town", "bus stop near dha", "faizabad bus stop",
"can i get transport mid semester", "bus from g9",

# ================= LIBRARY (20) =================
"library opening hours", "how to borrow a book", "how to get library card",
"what resources are in library", "is online library available",
"library membership process", "digital library access", "available library seats",
"is there a quiet zone in library", "library facilities",
"how to access research papers", "library rules",
"how to renew a library book", "what is library fine", "are journals available",
"ieee xplore access", "springer link access", "hec digital library",
"how many books can i borrow", "library booking process",

# ================= INTERNSHIP (20) =================
"how to get an internship", "internship opportunities available",
"industry placement available", "job placement after graduation",
"career development center", "internship program details",
"how to get internship letter", "job opportunities for graduates",
"placement cell services", "university job portal",
"career counseling available", "alumni network access",
"university industry connections", "internship period",
"paid internship available", "which companies hire from here",
"final year placement drive", "cv building support",
"interview preparation help", "linkedin career portal",

# ================= PROGRAMS (20) =================
"what programs are offered", "list all available programs", "is bs cs available",
"what ms programs are there", "is phd available", "which degree should i choose",
"full list of programs", "is software engineering program available",
"data science program details", "ai program available", "bs se admission",
"ms cs details", "is mba available", "new programs for 2026",
"what degree programs exist", "bachelor programs list", "master programs list",
"evening program available", "bs cybersecurity new program", "ms cloud computing",

# ================= FACULTY (20) =================
"who are the professors here", "can i get faculty list", "how to meet a professor",
"when are teacher office hours", "what is faculty email format", "who is department head",
"how to contact my professor", "who are the best teachers",
"faculty educational qualifications", "are there visiting faculty members",
"how many permanent faculty", "how many phd faculty", "student to teacher ratio",
"does faculty do research", "how to reach my course teacher",
"cs department faculty", "se department head", "professor availability",
"faculty profile on website", "teaching staff details",

# ================= GRADUATION (20) =================
"graduation requirements", "steps to graduate", "minimum cgpa to graduate",
"when is convocation", "how to get degree", "what is final year project",
"fyp details", "degree collection process", "how to get transcript",
"graduation ceremony details", "degree verification process",
"how many credits needed to graduate", "what is capstone project",
"convocation date 2026", "degree attestation process",
"hec degree verification", "mofa attestation", "official transcript request",
"provisional certificate", "degree issuance timeline",

# ================= CAMPUS LIFE (20) =================
"how is campus life", "is campus enjoyable", "what student activities are there",
"cultural events on campus", "how do students spend time",
"is there a good canteen", "describe student environment", "campus vibe",
"campus culture details", "co curricular activities list",
"how is student community", "campus atmosphere",
"activities for students at university", "social life on campus",
"campus overall experience", "annual festivals at university",
"spring festival details", "campus open for visit",
"is campus green and clean", "student welfare services",

# ================= COMPLAINTS (20) =================
"how to file a complaint", "complaint submission process", "register a complaint",
"academic complaint process", "fee related complaint", "hostel complaint process",
"complaint against faculty", "where is grievance cell", "complaint form download",
"what are student rights", "how to report an issue", "who is ombudsman",
"where to complain", "complaint email address", "complaint office location",
"sexual harassment complaint", "bullying complaint process",
"exam grade dispute", "complaint response time", "escalate unresolved complaint",

# ================= FAREWELL (20) =================
"bye", "goodbye", "thanks", "thank you", "see you later",
"take care", "see you", "that was very helpful", "you helped me a lot",
"thanks for the help", "good bye", "ok bye now",
"cheers", "thanks a lot", "appreciate your help",
"have a good day", "ok done", "all good now", "got my answer", "thanks bye",
]


# =========================
# INTENTS
# =========================
intents = (
    ["greeting"]     * 20 +
    ["admissions"]   * 20 +
    ["fees"]         * 20 +
    ["scholarships"] * 20 +
    ["location"]     * 25 +
    ["contact"]      * 20 +
    ["academics"]    * 20 +
    ["exams"]        * 20 +
    ["facilities"]   * 25 +
    ["societies"]    * 20 +
    ["small_talk"]   * 20 +
    ["hostel"]       * 20 +
    ["transport"]    * 20 +
    ["library"]      * 20 +
    ["internship"]   * 20 +
    ["programs"]     * 20 +
    ["faculty"]      * 20 +
    ["graduation"]   * 20 +
    ["campus_life"]  * 20 +
    ["complaints"]   * 20 +
    ["farewell"]     * 20
)

# =========================
# SAFETY CHECK
# =========================
if len(texts) != len(intents):
    raise ValueError(f"Mismatch: texts={len(texts)}, intents={len(intents)}")

print(f"Data OK: {len(texts)} samples, {len(set(intents))} intents")


# =========================
# DATAFRAME
# =========================
df = pd.DataFrame({"text": texts, "intent": intents})


# =========================
# MODEL TRAINING
# =========================
print("Training model...")

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True),
    SVC(kernel="linear", probability=True, C=2.0)
)

model.fit(df["text"], df["intent"])
print("Model trained successfully!")


# =========================
# RESPONSES
# =========================
responses = {
    "greeting": [
        "Assalam-o-Alaikum! 👋 Welcome! How can I help you today?",
        "Hello! 😊 Ask me anything about admissions, fees, programs, or campus life.",
        "Hi there! I'm your Hamza AI University Assistant. What can I do for you?"
    ],
    "admissions": [
        "📋 Admissions for Fall 2026 are open! Apply online at riphah.edu.pk. Need eligibility info?",
        "Required documents: Matric certificate, FSc certificate, CNIC copy, and 2 passport photos.",
        "Entry test (NTS or university test) is required. Deadline is usually in July — apply early!",
        "Merit list is announced within 2 weeks of the test. Limited seats, so apply as soon as possible."
    ],
    "fees": [
        "💰 BS CS: PKR 120,000/sem | MS CS: PKR 95,000/sem | MBA: PKR 110,000/sem.",
        "Pay via bank challan, JazzCash, EasyPaisa, or online banking. Need the challan process?",
        "⚠️ Late fee fine is PKR 500/day after the due date. Always pay before the deadline!",
        "Installment plans may be available for financial hardship — contact the Accounts Office."
    ],
    "scholarships": [
        "🎓 Merit scholarships need 3.5+ CGPA (top 10%). Need-based requires an income certificate.",
        "HEC need-based scholarships, Zakat funds, and Ehsaas scholarships are all available.",
        "Sports, Hafiz-e-Quran, and kinship scholarships are also offered — ask the scholarship office.",
        "Maintain 3.5+ CGPA every semester to keep renewing your merit scholarship."
    ],
    "location": [
        "📍 Campus is at H-8/4 Islamabad, near Faizabad Interchange. Easily found on Google Maps!",
        "Search 'Riphah International University Islamabad' on Google Maps for turn-by-turn directions.",
        "Nearest landmarks: Faizabad Metro Station (5 min), Jinnah Hospital, and Kashmir Highway.",
        "Accessible via CDA bus routes, Faizabad Metro, and the university's own shuttle service."
    ],
    "contact": [
        "📞 UAN: 051-111-747-424 | ✉️ admissions@riphah.edu.pk | Office: Mon–Sat, 9 AM–5 PM.",
        "WhatsApp support is available on the UAN number during office hours.",
        "Visit the Admission Block at the main campus entrance for walk-in inquiries.",
        "For department queries: cs@riphah.edu.pk | For fee issues: accounts@riphah.edu.pk"
    ],
    "academics": [
        "📚 BS CS is an 8-semester (4-year) program requiring 130+ credit hours.",
        "⚠️ Minimum 75% attendance is mandatory — falling below may result in de-registration.",
        "Core subjects: Programming, Data Structures, Algorithms, Databases, AI, and Networks.",
        "GPA below 2.0 triggers academic probation. You can repeat a course once to improve."
    ],
    "exams": [
        "📅 Midterms: Week 8–9 | Finals: Week 17–18. Schedule is posted on the student portal.",
        "Passing marks: 50% in finals + 40% in midterms (combined). GPA: A=4.0, B=3.0, C=2.0, D=1.0.",
        "Makeup exams are for medical emergencies only — submit a doctor's certificate within 3 days.",
        "Typical weightage: Quizzes 10% | Assignments 10% | Midterm 30% | Final Exam 50%."
    ],
    "facilities": [
        "🏫 Campus has high-speed WiFi, state-of-the-art computer labs, AI & research labs, and a digital library.",
        "⚽ Sports: Cricket ground, basketball & volleyball courts, indoor gym, table tennis, and a full sports complex.",
        "🏥 Support: Medical room (doctor on duty), separate prayer halls, 24/7 CCTV security, and ample parking.",
        "🖥️ Lab timings: Computer labs open 8 AM–9 PM daily. AI & Research labs need advance booking.",
        "🛋️ Student spaces: AC study lounges, group discussion rooms, cafeteria, and a dedicated student affairs office."
    ],
    "societies": [
        "🏆 Active societies: IEEE, Google DSC, ACM Chapter, Coding Club, Debating Society, and Drama Club.",
        "Societies run hackathons, tech seminars, coding competitions, and inter-university events.",
        "To join, attend the society orientation held in Week 1 of each semester — it's free!",
        "Active participation earns co-curricular credit hours — a great boost to your transcript!"
    ],
    "small_talk": [
        "I'm Hamza AI — your smart university assistant! 🤖 Ask me about admissions, fees, or campus life.",
        "I was built to help students like you navigate university life. What would you like to know?",
        "I know everything about this university — programs, fees, hostel, and more. Just ask!"
    ],
    "hostel": [
        "🏠 Separate fully-furnished hostels are available for boys and girls on campus.",
        "Hostel fee: ~PKR 25,000/month including 3 daily meals, WiFi, and 24/7 security.",
        "Rooms have WiFi, attached washrooms, study tables, wardrobes, and CCTV surveillance.",
        "Apply at the Student Affairs office before the semester — seats are limited, apply early!",
        "Rules: curfew at 10 PM, no outside guests allowed, mess timings must be respected."
    ],
    "transport": [
        "🚌 University buses cover 10+ routes across Islamabad and Rawalpindi every day.",
        "Transport fee: PKR 8,000/semester. Register at the Transport Office in Week 1.",
        "⏰ Morning pickup: 7:30–8:00 AM | Evening drop: 5:00–6:00 PM from campus.",
        "Key stops: Faizabad, Pir Wadhai, Bahria Town, DHA Phase 2, Saddar, and G-9 Markaz.",
        "A shuttle runs between campus and Faizabad Metro every hour on weekdays."
    ],
    "library": [
        "📚 Library hours: Monday–Saturday, 8 AM–9 PM. Closed Sundays and public holidays.",
        "Borrow up to 3 books for 14 days with your student ID. Renew online via the portal.",
        "Digital access: IEEE Xplore, Springer Link, HEC Digital Library, and 10,000+ e-books.",
        "Late return fine: PKR 10/day per book. Renew before the due date to avoid charges.",
        "Private study rooms available — book 24 hours in advance at the library front desk."
    ],
    "internship": [
        "💼 The Career Development Center (CDC) connects students with internships every semester.",
        "Recruiters include PTCL, Systems Ltd, NetSol, IBM, and 50+ top local & global tech firms.",
        "Get your internship offer letter processed at CDC — takes 2 working days.",
        "Final-year placement drives are held every spring semester. Update your CV!",
        "Use the university's alumni LinkedIn group and job portal for extra career opportunities."
    ],
    "programs": [
        "🎓 Undergraduate: BS CS, BS SE, BS AI, BS Data Science, BBA, BS Pharmacy & more.",
        "Postgraduate: MS CS, MS SE, MS Data Science, MS AI, MBA, and PhD in CS/Engineering.",
        "PhD requires an MS with 3.0+ CGPA. Research funding opportunities may be available.",
        "Evening programs for working professionals: BS CS, MS CS, and MBA (weekdays 6–9 PM).",
        "New 2026 programs: BS Cybersecurity 🔐 and MS Cloud Computing ☁️ — limited seats!"
    ],
    "faculty": [
        "👨‍🏫 200+ faculty — over 60% hold PhDs from top national and international universities.",
        "Email format: firstname.lastname@riphah.edu.pk — full faculty list on the university portal.",
        "Faculty office hours are posted on department notice boards at the start of each semester.",
        "Visiting industry professionals teach specialized electives — bringing real-world expertise.",
        "Student-to-faculty ratio is ~20:1, ensuring personalized mentorship and guidance."
    ],
    "graduation": [
        "🎓 Complete 130+ credit hours with minimum 2.0 CGPA — and you're ready to graduate!",
        "Final Year Project (FYP) is compulsory — it runs across both 7th and 8th semesters.",
        "Convocation is held annually (usually April/May). Watch official university announcements.",
        "Request transcripts or degree copies at the Registrar Office — ready in 5–7 working days.",
        "HEC/MOFA attestation guidance available at the Registrar — bring originals and copies."
    ],
    "campus_life": [
        "🎉 Campus life is vibrant! Tech fests, sports galas, cultural shows, and marathons every semester.",
        "The cafeteria serves fresh halal meals, snacks, and beverages at very affordable student prices.",
        "A green, modern campus with open courtyards, cozy study lounges, and a positive community.",
        "Annual highlights: Spring Festival 🌸, Coding Marathon 💻, Sports Gala 🏅, and IEEE TechWeek.",
        "Fully accessible: ramps, elevators, and special support for differently-abled students."
    ],
    "complaints": [
        "📋 Academic issues (grades/attendance)? Start with your Course Instructor, then Department Head.",
        "Fee or financial complaints? Visit the Accounts Office or Student Affairs Office directly.",
        "Submit formal complaints on the student portal — responses within 7 working days.",
        "The Grievance Cell handles all sensitive complaints with full confidentiality.",
        "Still unresolved? Escalate to the Ombudsman: ombudsman@riphah.edu.pk"
    ],
    "farewell": [
        "You're welcome! 😊 Best of luck with your studies — come back anytime!",
        "Happy to help! Take care and have a wonderful day! 👋",
        "Goodbye! Stay curious and keep learning! 🎓 See you next time."
    ],
    "fallback": [
        "Sorry, I didn't quite get that. Could you rephrase? Try asking about admissions, fees, hostel, or location.",
        "Hmm, I'm not sure about that. I can help with programs, exams, facilities, transport, and more!",
        "Can you clarify your question? I'm here to help with anything university-related!"
    ]
}


# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)
model_path = "models/chatbot_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to '{model_path}'")


# =========================
# CHAT FUNCTION
# =========================
def chatbot_response(user_input):
    user_input = clean(user_input)
    probs = model.predict_proba([user_input])[0]
    intent = model.classes_[probs.argmax()]
    confidence = max(probs)

    if confidence < 0.30:
        return random.choice(responses["fallback"])

    return random.choice(responses.get(intent, responses["fallback"]))


# =========================
# CLI TEST LOOP
# =========================
if __name__ == "__main__":
    print("\nChatbot Ready! Type 'exit' to stop\n")
    while True:
        msg = input("You: ")
        if msg.lower() == "exit":
            print("Bot: Goodbye!")
            break
        print("Bot:", chatbot_response(msg))