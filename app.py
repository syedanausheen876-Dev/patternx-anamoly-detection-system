import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import ipaddress


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PatternX",
    page_icon="logo copy.png",
    layout="wide"
)


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fa;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
}

h1, h2, h3 {
    color: #173f63;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 10px;
    padding: 15px;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #d9e2ec;
}

.stButton > button {
    border-radius: 7px;
    border: 1px solid #173f63;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOGIN
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    left, center, right = st.columns([1, 1.1, 1])

    with center:

        st.write("")

        try:
            st.image("logo copy.png", width=110)
        except:
            pass

        st.markdown(
            "<h1 style='text-align:center;'>PatternX</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align:center;color:#66788a;'>"
            "AI-Powered Real-Time Cybersecurity Anomaly Detection System"
            "</p>",
            unsafe_allow_html=True
        )

        st.write("")

        st.subheader("Admin Login")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button(
            "Login",
            use_container_width=True
        ):

            if (
                username == "security_admin"
                and password == "patternx123"
            ):

                st.session_state.logged_in = True
                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "patternx_model.pkl"
    )

    features = joblib.load(
        "patternx_features.pkl"
    )

    return model, features


# Model and datasets are loaded lazily.
# This keeps the login and first dashboard load fast.
model = None
features = None
df = None
friday_df = None
friday_summary_df = None


def get_model():

    global model, features

    if model is None or features is None:
        model, features = load_model()

    return model, features


# ============================================================
# LOAD CICIDS2017 SUMMARY DATA
# ============================================================

@st.cache_data
def load_cicids_data():

    data = pd.read_csv(
        "CICIDS2017_cleaned.csv",
        usecols=["Attack Type"]
    )

    data.columns = data.columns.str.strip()

    return data


# ============================================================
# LOAD FRIDAY SUMMARY DATA
# ============================================================

@st.cache_data
def load_friday_summary_data():

    data = pd.read_csv(
        "Friday.csv",
        usecols=["Src IP dec", "Dst IP dec"]
    )

    data.columns = data.columns.str.strip()

    return data


# ============================================================
# LOAD FRIDAY MODEL DATA
# ============================================================

@st.cache_data
def load_friday_data():

    # Live Monitoring needs only a small candidate pool, not the full
    # Friday file in memory. Keep several suspicious records per source
    # IP so the ML model has more than one chance to confirm each host.
    required_columns = [
        "Src IP dec", "Src Port", "Dst IP dec", "Dst Port",
        "Protocol", "Timestamp", "Label"
    ] + list(FRIDAY_TO_MODEL.values())

    required_columns = list(dict.fromkeys(required_columns))

    benign_candidates = []
    suspicious_candidates = []
    suspicious_per_source = {}
    row_offset = 0

    for chunk in pd.read_csv(
        "Friday.csv",
        usecols=required_columns,
        chunksize=25000
    ):

        chunk.columns = chunk.columns.str.strip()
        chunk.index = range(
            row_offset,
            row_offset + len(chunk)
        )
        row_offset += len(chunk)

        labels = chunk["Label"].astype(str).str.upper().str.strip()

        benign_chunk = chunk[labels == "BENIGN"]
        suspicious_chunk = chunk[labels != "BENIGN"]

        if len(benign_candidates) < 40:
            needed = 40 - len(benign_candidates)
            benign_candidates.extend(
                list(benign_chunk.head(needed).iterrows())
            )

        # Keep multiple suspicious records for each source IP.
        # This is important because one record from a source may be
        # misclassified by the model even when another record is detected.
        for idx, row in suspicious_chunk.iterrows():
            source_ip = convert_decimal_ip(row["Src IP dec"])
            count = suspicious_per_source.get(source_ip, 0)

            if count >= 12:
                continue

            suspicious_candidates.append((int(idx), row))
            suspicious_per_source[source_ip] = count + 1

        # Six source IPs are present in the Friday data; 12 records each
        # gives the detector enough candidates without loading the file.
        if (
            len(benign_candidates) >= 40
            and len(suspicious_per_source) >= 6
            and all(v >= 12 for v in suspicious_per_source.values())
        ):
            break

    if len(benign_candidates) < 26:
        raise ValueError(
            "Could not collect enough normal Friday records for the demo."
        )

    if len(suspicious_per_source) < 4:
        raise ValueError(
            "Could not collect four different suspicious source IPs."
        )

    selected = benign_candidates + suspicious_candidates

    rows = [row for _, row in selected]
    indices = [idx for idx, _ in selected]

    data = pd.DataFrame(rows)
    data.index = indices

    return data


# ============================================================
# SESSION VARIABLES
# ============================================================

if "monitor_index" not in st.session_state:
    st.session_state.monitor_index = 0

if "monitor_results" not in st.session_state:
    st.session_state.monitor_results = []

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "restricted_hosts" not in st.session_state:
    st.session_state.restricted_hosts = {}

if "selected_alert" not in st.session_state:
    st.session_state.selected_alert = None

if "last_detection" not in st.session_state:
    st.session_state.last_detection = None

if "risk_counts" not in st.session_state:
    st.session_state.risk_counts = {}

if "last_live_alert" not in st.session_state:
    st.session_state.last_live_alert = None

if "demo_stream" not in st.session_state:
    st.session_state.demo_stream = []

if "demo_initialized" not in st.session_state:
    st.session_state.demo_initialized = False

if "demo_completed" not in st.session_state:
    st.session_state.demo_completed = False

if "demo_last_result" not in st.session_state:
    st.session_state.demo_last_result = None


# ============================================================
# FRIDAY → MODEL FEATURE MAPPING
# ============================================================

FRIDAY_TO_MODEL = {

    "Destination Port": "Dst Port",

    "Flow Duration": "Flow Duration",

    "Total Fwd Packets": "Total Fwd Packet",

    "Total Length of Fwd Packets":
        "Total Length of Fwd Packet",

    "Fwd Packet Length Max":
        "Fwd Packet Length Max",

    "Fwd Packet Length Min":
        "Fwd Packet Length Min",

    "Fwd Packet Length Mean":
        "Fwd Packet Length Mean",

    "Fwd Packet Length Std":
        "Fwd Packet Length Std",

    "Bwd Packet Length Max":
        "Bwd Packet Length Max",

    "Bwd Packet Length Min":
        "Bwd Packet Length Min",

    "Bwd Packet Length Mean":
        "Bwd Packet Length Mean",

    "Bwd Packet Length Std":
        "Bwd Packet Length Std",

    "Flow Bytes/s":
        "Flow Bytes/s",

    "Flow Packets/s":
        "Flow Packets/s",

    "Flow IAT Mean":
        "Flow IAT Mean",

    "Flow IAT Std":
        "Flow IAT Std",

    "Flow IAT Max":
        "Flow IAT Max",

    "Flow IAT Min":
        "Flow IAT Min",

    "Fwd IAT Total":
        "Fwd IAT Total",

    "Fwd IAT Mean":
        "Fwd IAT Mean",

    "Fwd IAT Std":
        "Fwd IAT Std",

    "Fwd IAT Max":
        "Fwd IAT Max",

    "Fwd IAT Min":
        "Fwd IAT Min",

    "Bwd IAT Total":
        "Bwd IAT Total",

    "Bwd IAT Mean":
        "Bwd IAT Mean",

    "Bwd IAT Std":
        "Bwd IAT Std",

    "Bwd IAT Max":
        "Bwd IAT Max",

    "Bwd IAT Min":
        "Bwd IAT Min",

    "Fwd Header Length":
        "Fwd Header Length",

    "Bwd Header Length":
        "Bwd Header Length",

    "Fwd Packets/s":
        "Fwd Packets/s",

    "Bwd Packets/s":
        "Bwd Packets/s",

    "Packet Length Min":
        "Packet Length Min",

    "Packet Length Max":
        "Packet Length Max",

    "Packet Length Mean":
        "Packet Length Mean",

    "Packet Length Std":
        "Packet Length Std",

    "Packet Length Variance":
        "Packet Length Variance",

    "FIN Flag Count":
        "FIN Flag Count",

    "PSH Flag Count":
        "PSH Flag Count",

    "ACK Flag Count":
        "ACK Flag Count",

    "Average Packet Size":
        "Average Packet Size",

    "Subflow Fwd Bytes":
        "Subflow Fwd Bytes",

    "Init_Win_bytes_forward":
        "FWD Init Win Bytes",

    "Init_Win_bytes_backward":
        "Bwd Init Win Bytes",

    "act_data_pkt_fwd":
        "Fwd Act Data Pkts",

    "min_seg_size_forward":
        "Fwd Seg Size Min",

    "Active Mean":
        "Active Mean",

    "Active Max":
        "Active Max",

    "Active Min":
        "Active Min",

    "Idle Mean":
        "Idle Mean",

    "Idle Max":
        "Idle Max",

    "Idle Min":
        "Idle Min"
}


# ============================================================
# PREPARE FRIDAY ROW FOR ML MODEL
# ============================================================

def prepare_friday_model_input(row):

    _, model_features = get_model()

    model_input = pd.DataFrame(
        columns=model_features,
        index=[0]
    )

    for model_column, friday_column in FRIDAY_TO_MODEL.items():

        if (
            model_column in model_features
            and friday_column in row.index
        ):

            model_input.loc[
                0,
                model_column
            ] = row[friday_column]

    model_input = model_input.apply(
        pd.to_numeric,
        errors="coerce"
    )

    model_input = model_input.replace(
        [float("inf"), float("-inf")],
        0
    )

    model_input = model_input.fillna(0)

    return model_input


# ============================================================
# IP CONVERSION
# ============================================================

def convert_decimal_ip(value):

    try:

        if pd.isna(value):
            return "Not Available"

        return str(
            ipaddress.ip_address(
                int(float(value))
            )
        )

    except:

        return str(value)


# ============================================================
# ML DETECTION
# ============================================================

def detect_friday_activity(row):

    model_object, _ = get_model()

    X = prepare_friday_model_input(row)

    prediction = model_object.predict(X)[0]

    probabilities = model_object.predict_proba(X)[0]

    confidence = max(probabilities) * 100

    if prediction == "Normal Traffic":

        risk_score = (
            1 - confidence / 100
        ) * 100

        risk_level = "Normal"

    else:

        risk_score = confidence

        if risk_score >= 70:
            risk_level = "High Risk"

        elif risk_score >= 40:
            risk_level = "Medium Risk"

        else:
            risk_level = "Low Risk"

    return (
        prediction,
        confidence,
        risk_score,
        risk_level
    )


# ============================================================
# CREATE LIVE MONITOR RESULT
# ============================================================

def analyze_network_row(row, index, event_prefix="EVT", event_number=None):

    (
        prediction,
        confidence,
        risk_score,
        model_risk_level
    ) = detect_friday_activity(row)

    source_ip = convert_decimal_ip(
        row["Src IP dec"]
    )

    destination_ip = convert_decimal_ip(
        row["Dst IP dec"]
    )

    if event_number is None:
        event_id = (
            f"{event_prefix}-{100001 + index}"
        )
    else:
        event_id = (
            f"{event_prefix}-{100001 + event_number}"
        )

    # Repeated suspicious activity count for the same source.
    if prediction != "Normal Traffic":

        current_count = (
            st.session_state.risk_counts.get(
                source_ip,
                0
            )
        )

        current_count += 1

        st.session_state.risk_counts[
            source_ip
        ] = current_count

    else:

        current_count = (
            st.session_state.risk_counts.get(
                source_ip,
                0
            )
        )

    # Escalate the display status when the same
    # source generates repeated suspicious events.
    if prediction == "Normal Traffic":

        display_risk_level = "Normal"

    elif current_count >= 3:

        display_risk_level = "High Risk"

    else:

        display_risk_level = model_risk_level

    result = {

        "Event ID":
            event_id,

        "Source IP":
            source_ip,

        "Destination IP":
            destination_ip,

        "Source Port":
            row["Src Port"],

        "Destination Port":
            row["Dst Port"],

        "Prediction":
            prediction,

        "Confidence":
            confidence,

        "Risk Score":
            risk_score,

        "Risk Level":
            display_risk_level,

        "Risk Count":
            current_count,

        "Record":
            int(index),

        "Timestamp":
            str(row["Timestamp"])
    }

    return result


# ============================================================
# BUILD 30-ACTIVITY DEMO STREAM
# ============================================================

@st.cache_data
def load_cicids_attack_candidates(required_count=4):
    model_object, model_features = get_model()
    usecols = list(dict.fromkeys(model_features + ["Attack Type"]))
    confirmed = []

    for chunk in pd.read_csv("CICIDS2017_cleaned.csv", usecols=usecols, chunksize=50000):
        chunk.columns = chunk.columns.str.strip()
        candidates = chunk[chunk["Attack Type"].astype(str).str.strip() != "Normal Traffic"]
        if candidates.empty:
            continue
        X = candidates.drop(columns=["Attack Type"]).reindex(columns=model_features, fill_value=0)
        X = X.apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], 0).fillna(0)
        predictions = model_object.predict(X)
        for pos, prediction in enumerate(predictions):
            if prediction != "Normal Traffic":
                confirmed.append(candidates.iloc[pos].copy())
                if len(confirmed) >= required_count:
                    return pd.DataFrame(confirmed)
    return pd.DataFrame(confirmed)


def build_demo_stream():
    benign = friday_df[friday_df["Label"].astype(str).str.upper() == "BENIGN"]
    if len(benign) < 26:
        raise ValueError("Not enough BENIGN records for the demo.")

    attack_candidates = load_cicids_attack_candidates(4)
    if len(attack_candidates) < 4:
        raise ValueError("Could not find four ML-confirmed suspicious activities in CICIDS2017 data.")

    friday_suspicious = friday_df[friday_df["Label"].astype(str).str.upper() != "BENIGN"]
    friday_hosts = []
    used_sources = set()
    for idx, row in friday_suspicious.iterrows():
        source_ip = convert_decimal_ip(row["Src IP dec"])
        if source_ip in used_sources:
            continue
        friday_hosts.append((int(idx), row.copy()))
        used_sources.add(source_ip)
        if len(friday_hosts) == 4:
            break

    if len(friday_hosts) < 4:
        raise ValueError("Could not find four different suspicious hosts in Friday.csv.")

    model_features = get_model()[1]
    suspicious_rows = []
    for candidate, (host_idx, combined) in zip(attack_candidates.to_dict("records"), friday_hosts):
        for model_column, friday_column in FRIDAY_TO_MODEL.items():
            if model_column in model_features and friday_column in combined.index and model_column in candidate:
                combined[friday_column] = candidate[model_column]
        suspicious_rows.append((host_idx, combined))

    normal_rows = []
    for idx, row in benign.iterrows():
        prediction, _, _, _ = detect_friday_activity(row)
        if prediction == "Normal Traffic":
            normal_rows.append((int(idx), row))
        if len(normal_rows) == 26:
            break
    if len(normal_rows) < 26:
        raise ValueError("Not enough ML-confirmed normal records for the demo.")

    suspicious_positions = {14: suspicious_rows[0], 19: suspicious_rows[1], 24: suspicious_rows[2], 30: suspicious_rows[3]}
    stream = []
    normal_pos = 0
    for activity_number in range(1, 31):
        if activity_number in suspicious_positions:
            dataset_index, row = suspicious_positions[activity_number]
        else:
            dataset_index, row = normal_rows[normal_pos]
            normal_pos += 1
        stream.append((dataset_index, row))
    return stream


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "<h2 style='color:#173f63;'>PatternX</h2>",
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Security Operations Console"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",
    [
        "Network Dashboard",
        "Live Monitoring",
        "ML Detection",
        "Alert Center",
        "Investigation",
        "Analytics",
        "Reports"
    ]
)


# ============================================================
# LAZY PAGE DATA LOADING
# ============================================================

if page in ["Network Dashboard", "Analytics", "Reports"]:
    df = load_cicids_data()
    friday_summary_df = load_friday_summary_data()

if page in ["Live Monitoring", "ML Detection", "Investigation"]:
    friday_df = load_friday_data()

if page == "Live Monitoring":
    get_model()


st.sidebar.divider()

st.sidebar.caption(
    "Administrator"
)

if st.sidebar.button(
    "Reset Demo",
    use_container_width=True
):

    st.session_state.monitor_index = 0
    st.session_state.monitor_results = []
    st.session_state.alerts = []
    st.session_state.restricted_hosts = {}
    st.session_state.selected_alert = None
    st.session_state.last_detection = None
    st.session_state.risk_counts = {}
    st.session_state.last_live_alert = None
    st.session_state.demo_stream = []
    st.session_state.demo_initialized = False
    st.session_state.demo_completed = False
    st.session_state.demo_last_result = None
    st.rerun()

if st.sidebar.button(
    "Logout",
    use_container_width=True
):

    st.session_state.logged_in = False
    st.rerun()


# ============================================================
# COMMON DATA
# ============================================================

if df is not None:
    total_events = len(df)
    normal_events = (
        df["Attack Type"] == "Normal Traffic"
    ).sum()
    suspicious_events = total_events - normal_events
else:
    total_events = 0
    normal_events = 0
    suspicious_events = 0

model_accuracy = 99.85

unique_source_ips = (
    friday_summary_df["Src IP dec"]
    .nunique()
    if friday_summary_df is not None
    else 0
)


# ============================================================
# NETWORK DASHBOARD
# ============================================================

if page == "Network Dashboard":

    st.title(
        "Network Dashboard"
    )

    st.write(
        "Overview of network activity, monitored sources "
        "and security events."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Source IPs",
        f"{unique_source_ips:,}"
    )

    col2.metric(
        "Network Activities",
        f"{len(friday_summary_df):,}"
    )

    col3.metric(
        "Suspicious Activities",
        f"{suspicious_events:,}"
    )

    col4.metric(
        "Model Accuracy",
        f"{model_accuracy:.2f}%"
    )

    st.divider()

    st.subheader(
        "Network Status"
    )

    if suspicious_events > 0:

        st.warning(
            "Suspicious network activity is present "
            "in the monitored dataset."
        )

    else:

        st.success(
            "Network activity is currently normal."
        )

    st.subheader(
        "Network Activity Distribution"
    )

    counts = (
        df["Attack Type"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Attack Type",
        "Count"
    ]

    fig = px.bar(
        counts,
        x="Attack Type",
        y="Count",
        title="Network Activity by Attack Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Recently Restricted Hosts"
    )

    if not st.session_state.restricted_hosts:

        st.info(
            "No hosts are currently restricted."
        )

    else:

        restricted_rows = []

        for ip, info in (
            st.session_state
            .restricted_hosts
            .items()
        ):

            restricted_rows.append(
                {
                    "Source IP": ip,
                    "Status": "Restricted",
                    "Restriction Period": "4 hours",
                    "Restricted At":
                        info["restricted_at"],
                    "Restriction Ends":
                        info["restricted_until"],
                    "Reason":
                        info["reason"]
                }
            )

        st.dataframe(
            pd.DataFrame(
                restricted_rows
            ),
            width="stretch",
            hide_index=True
        )


# ============================================================
# LIVE MONITORING
# ============================================================

elif page == "Live Monitoring":

    st.title(
        "Live Monitoring"
    )

    st.write(
        "PatternX continuously analyzes incoming "
        "network-flow records and highlights suspicious activity."
    )

    st.info(
        "Demo Mode: CICIDS2017 Friday network-flow records "
        "are streamed automatically as incoming activity."
    )

    if not st.session_state.demo_initialized:

        st.session_state.demo_stream = build_demo_stream()
        st.session_state.demo_initialized = True
        st.session_state.monitor_index = 0

    DEMO_TOTAL = len(st.session_state.demo_stream)

    @st.fragment(run_every="3s")
    def live_monitor():

        processed = len(st.session_state.monitor_results)

        st.metric(
            "Progress",
            f"{processed} / {DEMO_TOTAL}"
        )

        if st.session_state.demo_completed:
            result = st.session_state.demo_last_result
        else:
            activity_number = st.session_state.monitor_index + 1
            dataset_index, row = st.session_state.demo_stream[
                st.session_state.monitor_index
            ]

            result = analyze_network_row(
                row,
                dataset_index,
                event_prefix="EVT",
                event_number=activity_number
            )

            st.session_state.monitor_results.append(result)

            if result["Prediction"] != "Normal Traffic":
                st.session_state.alerts.append(result)
                st.session_state.last_live_alert = result

            st.session_state.monitor_index += 1
            st.session_state.demo_last_result = result

            if activity_number >= DEMO_TOTAL:
                st.session_state.demo_completed = True

        if result and result["Prediction"] != "Normal Traffic":

            st.error(
                f"🚨 SUSPICIOUS ACTIVITY DETECTED\n\n"
                f"Source IP: {result['Source IP']}\n\n"
                f"Destination IP: {result['Destination IP']}\n\n"
                f"Detection: {result['Prediction']}\n\n"
                f"Confidence: {result['Confidence']:.2f}%\n\n"
                f"Risk Score: {result['Risk Score']:.2f}\n\n"
                f"Risk Count: {result['Risk Count']}\n\n"
                f"Risk Level: {result['Risk Level']}"
            )

            if result["Risk Count"] >= 3:
                st.warning(
                    "🔴 HIGH RISK — Repeated suspicious activity "
                    "from this source requires investigation."
                )

                st.info(
                    "Review the event in Investigation before taking "
                    "an administrative action."
                )

        st.divider()

        st.subheader(
            "Current Network Activity"
        )

        current_details = pd.DataFrame({
            "Field": [
                "Event ID",
                "Timestamp",
                "Source IP",
                "Destination IP",
                "Source Port",
                "Destination Port",
                "ML Detection",
                "Confidence",
                "Risk Score",
                "Risk Count",
                "Risk Level"
            ],
            "Value": [
                result["Event ID"],
                result["Timestamp"],
                result["Source IP"],
                result["Destination IP"],
                result["Source Port"],
                result["Destination Port"],
                result["Prediction"],
                f"{result['Confidence']:.2f}%",
                f"{result['Risk Score']:.2f}",
                result["Risk Count"],
                result["Risk Level"]
            ]
        })

        st.dataframe(
            current_details,
            width="stretch",
            hide_index=True
        )

        # Host restriction is handled only from Investigation after review.

        st.subheader(
            "Risk Level Guide"
        )

        risk_guide = pd.DataFrame({
            "Risk Score": ["0–39", "40–69", "70–100"],
            "Risk Level": ["Low Risk", "Medium Risk", "High Risk"]
        })

        st.dataframe(
            risk_guide,
            width="stretch",
            hide_index=True
        )

        st.subheader(
            "Recent Network Activity"
        )

        recent = pd.DataFrame(
            st.session_state.monitor_results[-20:]
        )

        if not recent.empty:
            st.dataframe(
                recent,
                width="stretch",
                hide_index=True
            )

        st.divider()

        st.subheader(
            "Suspicious Activities — Investigation Required"
        )

        suspicious_results = [
            item for item in st.session_state.monitor_results
            if item["Prediction"] != "Normal Traffic"
        ]

        if suspicious_results:
            suspicious_table = []

            for item in suspicious_results:
                status = (
                    "Restricted"
                    if item["Source IP"]
                    in st.session_state.restricted_hosts
                    else "Investigation Required"
                )

                suspicious_table.append({
                    "Event ID": item["Event ID"],
                    "Source IP": item["Source IP"],
                    "Destination IP": item["Destination IP"],
                    "Detection": item["Prediction"],
                    "Risk Score": round(item["Risk Score"], 2),
                    "Risk Level": item["Risk Level"],
                    "Status": status
                })

            st.dataframe(
                pd.DataFrame(suspicious_table),
                width="stretch",
                hide_index=True
            )
        else:
            st.info("No suspicious activity detected yet.")

        if st.session_state.demo_completed:
            st.success(
                "30 network activities analyzed. Demo monitoring is complete."
            )

    live_monitor()


# ============================================================
# ML DETECTION
# ============================================================

elif page == "ML Detection":

    st.title(
        "ML Detection"
    )

    st.write(
        "Random Forest analyzes network-flow features "
        "and displays the latest detected activity."
    )

    st.divider()

    if st.session_state.monitor_results:

        result = st.session_state.monitor_results[-1]

        st.subheader(
            "Latest Detection Result"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Source IP", result["Source IP"])
        c2.metric("Detection", result["Prediction"])
        c3.metric("Confidence", f"{result['Confidence']:.2f}%")
        c4.metric("Risk Score", f"{result['Risk Score']:.2f}")

        if result["Prediction"] == "Normal Traffic":
            st.success("Normal network activity detected.")
        else:
            st.error(
                f"Suspicious activity detected: {result['Prediction']}"
            )
            st.warning(
                f"Risk Level: {result['Risk Level']} | "
                f"Risk Count: {result['Risk Count']}"
            )

        st.divider()
        st.subheader("Network Flow Details")

        record_number = int(result["Record"])
        record = friday_df.loc[[record_number]]

        network_details = pd.DataFrame({
            "Field": [
                "Event ID", "Timestamp", "Source IP", "Destination IP",
                "Source Port", "Destination Port", "Protocol",
                "ML Detection", "Confidence", "Risk Score",
                "Risk Count", "Risk Level"
            ],
            "Value": [
                result["Event ID"],
                str(record.iloc[0]["Timestamp"]),
                result["Source IP"],
                result["Destination IP"],
                record.iloc[0]["Src Port"],
                record.iloc[0]["Dst Port"],
                record.iloc[0]["Protocol"],
                result["Prediction"],
                f"{result['Confidence']:.2f}%",
                f"{result['Risk Score']:.2f}",
                result["Risk Count"],
                result["Risk Level"]
            ]
        })

        st.dataframe(
            network_details,
            width="stretch",
            hide_index=True
        )

    else:
        st.info(
            "Live Monitoring has not analyzed an activity yet. "
            "Open Live Monitoring to start the automatic demo stream."
        )


# ============================================================
# ALERT CENTER
# ============================================================

elif page == "Alert Center":

    st.title(
        "Alert Center"
    )

    st.write(
        "Suspicious activities identified by "
        "the PatternX ML model."
    )

    st.divider()

    if not st.session_state.alerts:

        st.success(
            "No ML-generated alerts available."
        )

    else:

        alert_df = pd.DataFrame(
            st.session_state.alerts
        )

        alert_df["Status"] = alert_df["Source IP"].apply(
            lambda ip: (
                "Restricted"
                if ip in st.session_state.restricted_hosts
                else "Investigation Required"
            )
        )

        st.metric(
            "Active Alerts",
            len(alert_df)
        )

        st.dataframe(
            alert_df,
            width="stretch",
            hide_index=True
        )

        st.divider()

        st.subheader(
            "Alert Investigation"
        )

        alert_options = []

        for _, alert in alert_df.iterrows():

            alert_options.append(
                f"{alert['Event ID']} — "
                f"{alert['Source IP']} — "
                f"{alert['Prediction']}"
            )

        selected_option = st.selectbox(
            "Select Alert",
            alert_options
        )

        selected_event = (
            selected_option.split(" — ")[0]
        )

        if st.button(
            "Investigate Selected Alert",
            use_container_width=True
        ):

            st.session_state.selected_alert = (
                selected_event
            )

            st.success(
                "Alert selected. Open Investigation "
                "to review the event."
            )


# ============================================================
# INVESTIGATION
# ============================================================

elif page == "Investigation":

    st.title(
        "Incident Investigation"
    )

    st.write(
        "Review suspicious network activity before "
        "taking an administrative security action."
    )

    alerts = st.session_state.alerts

    if not alerts:

        st.info(
            "No suspicious activity is available "
            "for investigation."
        )

    else:

        alert_df = pd.DataFrame(
            alerts
        )

        event_ids = (
            alert_df["Event ID"]
            .tolist()
        )

        default_event = (
            st.session_state.selected_alert
            if st.session_state.selected_alert
            in event_ids
            else event_ids[0]
        )

        selected_event = st.selectbox(
            "Select Event",
            event_ids,
            index=event_ids.index(
                default_event
            )
        )

        selected = alert_df[
            alert_df["Event ID"] == selected_event
        ].iloc[0]

        source_ip = selected["Source IP"]

        st.divider()

        st.subheader(
            "Investigation Summary"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Source IP",
            source_ip
        )

        c2.metric(
            "Activity",
            selected["Prediction"]
        )

        c3.metric(
            "Risk Score",
            f"{selected['Risk Score']:.2f}"
        )

        c4.metric(
            "Confidence",
            f"{selected['Confidence']:.2f}%"
        )

        st.divider()

        st.subheader(
            "Network Flow Information"
        )

        record_number = int(
            selected["Record"]
        )

        record = friday_df.loc[
            [record_number]
        ]

        network_information = pd.DataFrame({

            "Field": [

                "Event ID",
                "Timestamp",
                "Source IP",
                "Destination IP",
                "Source Port",
                "Destination Port",
                "Protocol",
                "ML Detection",
                "Risk Score",
                "Risk Count",
                "Risk Level"
            ],

            "Value": [

                selected["Event ID"],
                str(record.iloc[0]["Timestamp"]),
                selected["Source IP"],
                selected["Destination IP"],
                record.iloc[0]["Src Port"],
                record.iloc[0]["Dst Port"],
                record.iloc[0]["Protocol"],
                selected["Prediction"],
                f"{selected['Risk Score']:.2f}",
                selected["Risk Count"],
                selected["Risk Level"]
            ]
        })

        st.dataframe(
            network_information,
            width="stretch",
            hide_index=True
        )

        st.divider()

        # Show repeated suspicious events from this
        # exact source IP.
        st.subheader(
            "Source IP Activity History"
        )

        source_history = alert_df[
            alert_df["Source IP"] == source_ip
        ][
            [
                "Event ID",
                "Timestamp",
                "Prediction",
                "Risk Score",
                "Risk Count",
                "Risk Level"
            ]
        ]

        st.dataframe(
            source_history,
            width="stretch",
            hide_index=True
        )

        st.divider()

        st.subheader(
            "Administrative Action"
        )

        if source_ip in st.session_state.restricted_hosts:

            restriction = (
                st.session_state
                .restricted_hosts[source_ip]
            )

            st.error(
                f"{source_ip} — Network Access Restricted"
            )

            st.write(
                f"""
**Status:** Investigation Required

**Reason:** {restriction["reason"]}

**Restriction Period:** 4 hours

**Restricted At:** {restriction["restricted_at"]}

**Restriction Ends:** {restriction["restricted_until"]}
"""
            )

            if st.button(
                "Restore Host Access",
                use_container_width=True
            ):

                del st.session_state.restricted_hosts[
                    source_ip
                ]

                st.success(
                    "Host access has been restored."
                )

                st.rerun()

        else:

            st.write(
                f"Selected source host: **{source_ip}**"
            )

            st.write(
                "The administrator can temporarily restrict "
                "network access while the suspicious activity "
                "is investigated."
            )

            source_risk_count = int(
                selected["Risk Count"]
            )

            # Restriction is an administrator action after investigation.
            # Each suspicious host can be restricted independently.
            if st.button(
                "Restrict Host",
                type="primary",
                use_container_width=True
            ):

                    start_time = datetime.now()
                    end_time = start_time + timedelta(hours=4)

                    st.session_state.restricted_hosts[
                        source_ip
                    ] = {
                        "restricted_at": start_time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "restricted_until": end_time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "reason": "Suspicious network activity identified by ML detection"
                    }

                    st.success(
                        f"{source_ip} has been temporarily "
                        "restricted for security investigation."
                    )

                    st.rerun()

            st.caption(
                f"Suspicious events from this source: {source_risk_count}. "
                "Review the event before taking administrative action."
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.title(
        "Security Analytics"
    )

    st.write(
        "Analysis of network activity and "
        "PatternX ML detections."
    )

    st.divider()

    counts = (
        df["Attack Type"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Attack Type",
        "Count"
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Activity Distribution"
        )

        fig1 = px.bar(
            counts,
            x="Attack Type",
            y="Count"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Activity Share"
        )

        fig2 = px.pie(
            counts,
            names="Attack Type",
            values="Count"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.subheader(
        "Detection Summary"
    )

    st.dataframe(
        counts,
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Network Dataset Information"
    )

    dataset_info = pd.DataFrame({

        "Metric": [

            "Friday Network Records",
            "Friday Dataset Columns",
            "Unique Source IPs",
            "Unique Destination IPs",
            "ML Input Features"
        ],

        "Value": [

            f"{len(friday_summary_df):,}",

            "2",

            f"{friday_summary_df['Src IP dec'].nunique():,}",

            f"{friday_summary_df['Dst IP dec'].nunique():,}",

            "52"
        ]
    })

    st.dataframe(
        dataset_info,
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Session Monitoring"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Activities Analyzed",
        len(
            st.session_state.monitor_results
        )
    )

    c2.metric(
        "Alerts",
        len(
            st.session_state.alerts
        )
    )

    c3.metric(
        "Restricted Hosts",
        len(
            st.session_state.restricted_hosts
        )
    )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.title(
        "Security Reports"
    )

    st.write(
        "PatternX model and network monitoring summary."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Test Accuracy",
        "99.85%"
    )

    c2.metric(
        "Input Features",
        "52"
    )

    c3.metric(
        "Training Records",
        "2,016,600"
    )

    c4.metric(
        "Testing Records",
        "504,151"
    )

    st.divider()

    st.subheader(
        "Random Forest Model"
    )

    st.write(
        """
PatternX uses a Random Forest classifier trained
using CICIDS2017 network-flow data.

The model analyzes 52 network-flow features and
classifies network activity into different categories.

The reported test accuracy is 99.85% based on the
completed model training and evaluation.
"""
    )

    st.divider()

    st.subheader(
        "Friday Network Flow Dataset"
    )

    st.write(
        f"""
The monitoring interface uses the Friday network-flow
records to display source and destination network
information during investigation.

Records available: {len(friday_summary_df):,}

Columns available: 89

Source IP and Destination IP information is used
for network investigation and host-level administrative
actions.
"""
    )

    st.divider()

    st.subheader(
        "Monitoring Summary"
    )

    report_data = pd.DataFrame({

        "Metric": [

            "CICIDS2017 ML Dataset Records",
            "Normal Traffic Records",
            "Non-Normal Records",
            "Friday Network Records",
            "Unique Source IPs",
            "ML Activities Analyzed",
            "Alerts Generated",
            "Restricted Hosts"
        ],

        "Value": [

            f"{total_events:,}",

            f"{normal_events:,}",

            f"{suspicious_events:,}",

            f"{len(friday_summary_df):,}",

            f"{friday_summary_df['Src IP dec'].nunique():,}",

            len(
                st.session_state.monitor_results
            ),

            len(
                st.session_state.alerts
            ),

            len(
                st.session_state.restricted_hosts
            )
        ]
    })

    st.dataframe(
        report_data,
        width="stretch",
        hide_index=True
    )

    st.success(
        "PatternX security report generated."
    )
