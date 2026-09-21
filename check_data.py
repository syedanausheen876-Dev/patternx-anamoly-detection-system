import os
import sqlite3
import smtplib
from email.message import EmailMessage
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PatternX",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .stApp {
        background: #f7f9fc;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #14213d;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-bottom: 30px;
    }

    .event-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e6eaf0;
        box-shadow: 0 5px 18px rgba(20, 33, 61, 0.06);
        margin-bottom: 20px;
    }

    .event-title {
        font-size: 23px;
        font-weight: 650;
        color: #14213d;
    }

    .event-subtitle {
        color: #667085;
        margin-top: 5px;
        margin-bottom: 12px;
    }

    .info-box {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e6eaf0;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        color: #14213d;
        margin-bottom: 15px;
    }

    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e6eaf0;
        text-align: center;
    }

    .metric-number {
        font-size: 30px;
        font-weight: 700;
        color: #14213d;
    }

    .metric-label {
        color: #667085;
        font-size: 14px;
    }

    .footer-note {
        text-align: center;
        color: #7b8494;
        font-size: 13px;
        margin-top: 45px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

MODEL_FILE = "patternx_model.pkl"
METADATA_FILE = "patternx_metadata.pkl"
DB_FILE = "patternx.db"


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    metadata = None

    if os.path.exists(METADATA_FILE):
        metadata = joblib.load(METADATA_FILE)

    return model, metadata


model, metadata = load_model()


# =========================================================
# EVENTS
# =========================================================

EVENTS = {

    "AI Nexus 2026": {
        "category": "Artificial Intelligence",
        "date": "26 September 2026",
        "time": "10:00 AM – 1:00 PM",
        "location": "PatternX Innovation Center",
        "description":
            "A fictional academic conference focused on artificial intelligence, "
            "innovation and emerging intelligent technologies."
    },

    "CodeCraft Summit 2026": {
        "category": "Software Development",
        "date": "3 October 2026",
        "time": "10:00 AM – 2:00 PM",
        "location": "PatternX Innovation Center",
        "description":
            "A fictional software development summit covering modern programming, "
            "software engineering and development practices."
    },

    "ML Vision 2026": {
        "category": "Machine Learning",
        "date": "10 October 2026",
        "time": "10:30 AM – 1:30 PM",
        "location": "PatternX Innovation Center",
        "description":
            "A fictional workshop focused on machine learning, data intelligence "
            "and practical data-driven applications."
    },

    "CyberShield 2026": {
        "category": "Cybersecurity",
        "date": "17 October 2026",
        "time": "10:00 AM – 1:00 PM",
        "location": "PatternX Innovation Center",
        "description":
            "A fictional cybersecurity conference covering digital safety, "
            "security awareness and modern cybersecurity practices."
    }
}


# =========================================================
# DATABASE
# =========================================================

def init_db():

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registrations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            full_name TEXT,
            email TEXT,
            phone TEXT,
            college TEXT,
            department TEXT,
            year TEXT,

            event_name TEXT,
            event_category TEXT,
            event_date TEXT,

            registration_time TEXT,

            prediction TEXT,
            confidence REAL,

            notification_status TEXT

        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(data):

    df = data.copy()

    # -----------------------------------------------------
    # IP FEATURES
    # -----------------------------------------------------

    def ip_valid(x):

        try:
            parts = str(x).split(".")

            if len(parts) != 4:
                return 0

            nums = [int(p) for p in parts]

            return int(all(0 <= n <= 255 for n in nums))

        except:
            return 0

    def ip_first(x):

        try:
            return int(str(x).split(".")[0])
        except:
            return 0

    def ip_second(x):

        try:
            return int(str(x).split(".")[1])
        except:
            return 0

    df["ip_valid"] = df["ip_address"].apply(ip_valid)
    df["ip_first_octet"] = df["ip_address"].apply(ip_first)
    df["ip_second_octet"] = df["ip_address"].apply(ip_second)


    # -----------------------------------------------------
    # EMAIL FEATURES
    # -----------------------------------------------------

    df["email_address"] = df["email_address"].fillna("").astype(str)

    df["email_length"] = df["email_address"].str.len()

    df["email_digits"] = df["email_address"].str.count(r"\d")

    df["email_specials"] = df["email_address"].str.count(
        r"[^A-Za-z0-9]"
    )

    df["email_has_plus"] = (
        df["email_address"].str.contains(
            r"\+", regex=True
        ).astype(int)
    )

    df["email_domain"] = (
        df["email_address"]
        .str.split("@")
        .str[-1]
    )

    df["email_domain_length"] = (
        df["email_domain"].str.len()
    )

    df["email_local"] = (
        df["email_address"]
        .str.split("@")
        .str[0]
    )

    df["email_local_length"] = (
        df["email_local"].str.len()
    )


    # -----------------------------------------------------
    # BILLING STATE
    # -----------------------------------------------------

    df["billing_state"] = (
        df["billing_state"]
        .fillna("unknown")
        .astype(str)
    )


    # -----------------------------------------------------
    # POSTAL CODE
    # -----------------------------------------------------

    df["billing_postal"] = (
        df["billing_postal"]
        .fillna("")
        .astype(str)
    )

    df["postal_prefix"] = (
        df["billing_postal"]
        .str[:3]
    )


    # -----------------------------------------------------
    # PHONE
    # -----------------------------------------------------

    df["phone_number"] = (
        df["phone_number"]
        .fillna("")
        .astype(str)
    )

    df["phone_digit_length"] = (
        df["phone_number"]
        .str.replace(r"\D", "", regex=True)
        .str.len()
    )

    df["phone_has_valid_length"] = (
        df["phone_digit_length"]
        .between(10, 15)
        .astype(int)
    )


    # -----------------------------------------------------
    # USER AGENT
    # -----------------------------------------------------

    df["user_agent"] = (
        df["user_agent"]
        .fillna("")
        .astype(str)
    )

    ua = df["user_agent"].str.lower()

    df["user_agent_length"] = (
        df["user_agent"].str.len()
    )

    df["ua_chrome"] = (
        ua.str.contains("chrome")
        .astype(int)
    )

    df["ua_firefox"] = (
        ua.str.contains("firefox")
        .astype(int)
    )

    df["ua_safari"] = (
        ua.str.contains("safari")
        .astype(int)
    )

    df["ua_windows"] = (
        ua.str.contains("windows")
        .astype(int)
    )

    df["ua_linux"] = (
        ua.str.contains("linux")
        .astype(int)
    )

    df["ua_mac"] = (
        ua.str.contains("mac")
        .astype(int)
    )

    df["ua_mobile"] = (
        ua.str.contains(
            "mobile|android|iphone|ipad"
        )
        .astype(int)
    )


    # -----------------------------------------------------
    # ADDRESS
    # -----------------------------------------------------

    df["billing_address"] = (
        df["billing_address"]
        .fillna("")
        .astype(str)
    )

    df["address_length"] = (
        df["billing_address"].str.len()
    )

    df["address_digits"] = (
        df["billing_address"]
        .str.count(r"\d")
    )

    df["address_has_street"] = (
        df["billing_address"]
        .str.lower()
        .str.contains(
            "street|st |road|rd |avenue|ave|lane|ln "
        )
        .astype(int)
    )


    # -----------------------------------------------------
    # TIMESTAMP
    # -----------------------------------------------------

    timestamp = pd.to_datetime(
        df["EVENT_TIMESTAMP"],
        errors="coerce"
    )

    df["event_hour"] = (
        timestamp.dt.hour.fillna(0).astype(int)
    )

    df["event_month"] = (
        timestamp.dt.month.fillna(0).astype(int)
    )

    df["event_dayofweek"] = (
        timestamp.dt.dayofweek.fillna(0).astype(int)
    )


    return df


# =========================================================
# PREDICTION
# =========================================================

def predict_registration(
    email,
    phone,
    event_timestamp
):

    raw = pd.DataFrame([{

        "ip_address": "unknown",
        "email_address": email,
        "billing_state": "unknown",
        "user_agent": "unknown",
        "billing_postal": "unknown",
        "phone_number": phone,
        "EVENT_TIMESTAMP": event_timestamp,
        "billing_address": "unknown"

    }])


    features = create_features(raw)


    # The saved model is a complete Pipeline.
    # It performs the required preprocessing internally.

    prediction = model.predict(features)[0]

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(features)[0]

        probability = float(
            max(probabilities)
        )


    if prediction == 1:

        result = "Suspicious"

    else:

        result = "Normal"


    return result, probability


# =========================================================
# EMAIL
# =========================================================

def send_suspicious_email(
    recipient,
    name,
    event_name
):

    try:

        sender = st.secrets["EMAIL_ADDRESS"]
        password = st.secrets["EMAIL_PASSWORD"]

        msg = EmailMessage()

        msg["Subject"] = (
            "PatternX Registration Analysis"
        )

        msg["From"] = sender
        msg["To"] = recipient

        msg.set_content(
            f"""
Hello {name},

Your registration for {event_name} has been analyzed by PatternX.

The system detected suspicious indicators in the submitted registration information.

This notification is part of the PatternX academic demonstration system.

Regards,
PatternX Security Platform
"""
        )

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                sender,
                password
            )

            smtp.send_message(msg)

        return "Sent"

    except Exception:

        return "Not configured"


# =========================================================
# SAVE REGISTRATION
# =========================================================

def save_registration(
    full_name,
    email,
    phone,
    college,
    department,
    year,
    event_name,
    event_category,
    event_date,
    prediction,
    confidence,
    notification_status
):

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO registrations (

            full_name,
            email,
            phone,
            college,
            department,
            year,

            event_name,
            event_category,
            event_date,

            registration_time,

            prediction,
            confidence,

            notification_status

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        full_name,
        email,
        phone,
        college,
        department,
        year,

        event_name,
        event_category,
        event_date,

        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        prediction,
        confidence,

        notification_status

    ))

    conn.commit()
    conn.close()


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:

    st.session_state.page = "home"


if "selected_event" not in st.session_state:

    st.session_state.selected_event = None


# =========================================================
# NAVIGATION
# =========================================================

def go_home():

    st.session_state.page = "home"


def show_home():

    st.markdown(
        '<div class="main-title">PatternX</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Enterprise Registration Security Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-box">'
        '<b>PatternX</b> analyzes event registration information '
        'using a trained machine learning model and identifies '
        'registrations that show suspicious patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Upcoming Events</div>',
        unsafe_allow_html=True
    )


    for event_name, details in EVENTS.items():

        st.markdown(
            f"""
            <div class="event-card">

                <div class="event-title">
                    {event_name}
                </div>

                <div class="event-subtitle">
                    {details["category"]}
                </div>

                <b>Date:</b> {details["date"]}<br>
                <b>Time:</b> {details["time"]}<br>
                <b>Location:</b> {details["location"]}

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "View Event",
            key=f"view_{event_name}",
            use_container_width=True
        ):

            st.session_state.selected_event = event_name
            st.session_state.page = "event"


    st.markdown(
        '<div class="footer-note">'
        'All events shown on this platform are fictional events '
        'created for the PatternX academic demonstration.'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# EVENT PAGE
# =========================================================

def show_event():

    event_name = st.session_state.selected_event

    details = EVENTS[event_name]

    if st.button("Back to Events"):

        go_home()

        st.rerun()


    st.markdown(
        f'<div class="main-title">{event_name}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">'
        f'{details["category"]}'
        f'</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="info-box">

        <div class="section-title">
        Event Details
        </div>

        <b>Date:</b> {details["date"]}<br><br>

        <b>Time:</b> {details["time"]}<br><br>

        <b>Location:</b> {details["location"]}<br><br>

        <b>Description:</b><br>
        {details["description"]}

        </div>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        "Register Now",
        type="primary",
        use_container_width=True
    ):

        st.session_state.page = "register"

        st.rerun()


# =========================================================
# REGISTRATION PAGE
# =========================================================

def show_registration():

    event_name = st.session_state.selected_event

    details = EVENTS[event_name]


    if st.button("Back to Event"):

        st.session_state.page = "event"

        st.rerun()


    st.markdown(
        '<div class="main-title">'
        'Registration Form'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">'
        f'Register for {event_name}'
        f'</div>',
        unsafe_allow_html=True
    )


    with st.form("registration_form"):

        full_name = st.text_input(
            "Full Name"
        )

        email = st.text_input(
            "Email Address"
        )

        phone = st.text_input(
            "Phone Number"
        )

        college = st.text_input(
            "College / Organization"
        )

        department = st.text_input(
            "Department / Course"
        )

        year = st.selectbox(
            "Year",
            [
                "1st Year",
                "2nd Year",
                "3rd Year",
                "4th Year"
            ]
        )


        submitted = st.form_submit_button(
            "Submit Registration",
            use_container_width=True
        )


    if submitted:

        if not full_name.strip():

            st.error(
                "Please enter your full name."
            )

            return


        if not email.strip():

            st.error(
                "Please enter your email address."
            )

            return


        if not phone.strip():

            st.error(
                "Please enter your phone number."
            )

            return


        if not college.strip():

            st.error(
                "Please enter your college / organization."
            )

            return


        if not department.strip():

            st.error(
                "Please enter your department / course."
            )

            return


        event_timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        try:

            prediction, confidence = predict_registration(
                email=email,
                phone=phone,
                event_timestamp=event_timestamp
            )

        except Exception as e:

            st.error(
                "PatternX could not analyze this registration."
            )

            st.caption(
                f"Technical error: {e}"
            )

            return


        notification_status = "Not required"


        if prediction == "Suspicious":

            notification_status = (
                send_suspicious_email(
                    email,
                    full_name,
                    event_name
                )
            )


        save_registration(

            full_name,
            email,
            phone,
            college,
            department,
            year,

            event_name,
            details["category"],
            details["date"],

            prediction,
            confidence,

            notification_status

        )


        st.session_state.last_prediction = prediction
        st.session_state.last_confidence = confidence
        st.session_state.last_notification = notification_status
        st.session_state.page = "result"

        st.rerun()


# =========================================================
# RESULT PAGE
# =========================================================

def show_result():

    prediction = st.session_state.last_prediction

    confidence = st.session_state.last_confidence

    notification = st.session_state.last_notification


    st.markdown(
        '<div class="main-title">'
        'Registration Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    if prediction == "Normal":

        st.success(
            "Registration analyzed successfully. "
            "No suspicious pattern was detected."
        )

    else:

        st.warning(
            "PatternX detected suspicious indicators "
            "in this registration."
        )


    if confidence is not None:

        st.write(
            f"**Model confidence:** "
            f"{confidence * 100:.2f}%"
        )


    if prediction == "Suspicious":

        if notification == "Sent":

            st.info(
                "A notification email was sent to the "
                "registered email address."
            )

        else:

            st.info(
                "Email notification is not configured "
                "in this local demonstration."
            )


    st.markdown(
        '<div class="info-box">'
        'PatternX uses a trained machine learning model '
        'to identify suspicious registration patterns. '
        'A suspicious prediction is an indicator for '
        'further review and does not by itself prove fraud.'
        '</div>',
        unsafe_allow_html=True
    )


    if st.button(
        "Back to Events",
        use_container_width=True
    ):

        go_home()

        st.rerun()


# =========================================================
# ADMIN LOGIN
# =========================================================

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "patternx123"


def show_admin_login():

    st.markdown(
        '<div class="main-title">'
        'PatternX Admin'
        '</div>',
        unsafe_allow_html=True
    )

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )


    if st.button(
        "Login",
        use_container_width=True
    ):

        if (
            username == ADMIN_USERNAME
            and password == ADMIN_PASSWORD
        ):

            st.session_state.admin_logged_in = True

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

def show_admin_dashboard():

    if not st.session_state.get(
        "admin_logged_in",
        False
    ):

        show_admin_login()

        return


    st.markdown(
        '<div class="main-title">'
        'Admin Dashboard'
        '</div>',
        unsafe_allow_html=True
    )


    if st.button("Logout"):

        st.session_state.admin_logged_in = False

        st.rerun()


    conn = sqlite3.connect(DB_FILE)

    df = pd.read_sql_query(
        "SELECT * FROM registrations ORDER BY id DESC",
        conn
    )

    conn.close()


    total = len(df)

    normal = (
        len(df[df["prediction"] == "Normal"])
        if total
        else 0
    )

    suspicious = (
        len(df[df["prediction"] == "Suspicious"])
        if total
        else 0
    )

    total_events = len(EVENTS)


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-number">
                    {total}
                </div>
                <div class="metric-label">
                    Total Registered
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-number">
                    {normal}
                </div>
                <div class="metric-label">
                    Normal Registrations
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-number">
                    {suspicious}
                </div>
                <div class="metric-label">
                    Suspicious Registrations
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-number">
                    {total_events}
                </div>
                <div class="metric-label">
                    Total Events
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">'
        'Registered Users'
        '</div>',
        unsafe_allow_html=True
    )


    if df.empty:

        st.info(
            "No registrations available yet."
        )

        return


    display_df = df[
        [
            "full_name",
            "email",
            "phone",
            "event_name",
            "registration_time",
            "prediction"
        ]
    ]


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        '<div class="section-title">'
        'Registration Details'
        '</div>',
        unsafe_allow_html=True
    )


    selected_id = st.selectbox(
        "Select Registration",
        df["id"].tolist()
    )


    selected = df[
        df["id"] == selected_id
    ].iloc[0]


    st.markdown(
        f"""
        <div class="info-box">

        <b>Name:</b> {selected["full_name"]}<br>
        <b>Email:</b> {selected["email"]}<br>
        <b>Phone:</b> {selected["phone"]}<br>
        <b>College:</b> {selected["college"]}<br>
        <b>Department:</b> {selected["department"]}<br>
        <b>Year:</b> {selected["year"]}<br><br>

        <b>Event:</b> {selected["event_name"]}<br>
        <b>Category:</b> {selected["event_category"]}<br>
        <b>Event Date:</b> {selected["event_date"]}<br>
        <b>Registration Time:</b> {selected["registration_time"]}<br><br>

        <b>PatternX Analysis:</b> {selected["prediction"]}<br>
        <b>Model Confidence:</b> {selected["confidence"] * 100:.2f}%<br>
        <b>Notification Status:</b> {selected["notification_status"]}

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("### PatternX")

    if st.button(
        "Home",
        use_container_width=True
    ):

        st.session_state.page = "home"

        st.rerun()


    if st.button(
        "Admin Login",
        use_container_width=True
    ):

        st.session_state.page = "admin"

        st.rerun()


# =========================================================
# PAGE ROUTING
# =========================================================

if st.session_state.page == "home":

    show_home()

elif st.session_state.page == "event":

    show_event()

elif st.session_state.page == "register":

    show_registration()

elif st.session_state.page == "result":

    show_result()

elif st.session_state.page == "admin":

    show_admin_dashboard()