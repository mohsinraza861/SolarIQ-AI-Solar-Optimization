
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# ==========================================================
# SOLARIQ - SOLAR PANEL ENERGY OPTIMIZATION + AI
# ==========================================================

st.set_page_config(
    page_title="SolarIQ | AI Solar Optimization",
    page_icon="☀️",
    layout="wide"
)

# -------------------- THEME --------------------

BG = "#07111f"
PANEL = "#10243a"
BORDER = "#294663"
TEXT = "#f4f8ff"
MUTED = "#b5c7dc"
CYAN = "#38bdf8"
YELLOW = "#fbbf24"
GREEN = "#34d399"
RED = "#fb7185"
PURPLE = "#c4b5fd"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Manrope', sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at 10% 0%, #183858 0%, transparent 35%),
        linear-gradient(135deg, #07111f, #0a1728);
    color: {TEXT};
}}

[data-testid="stSidebar"] {{
    background: #091827;
    border-right: 1px solid {BORDER};
}}

[data-testid="stHeader"] {{
    background: rgba(7,17,31,0.8);
}}

h1, h2, h3, h4, p, label, li {{
    color: {TEXT};
}}

h1 {{
    font-weight: 800 !important;
}}

.stMarkdown, .stText, .stCaption {{
    color: {TEXT};
}}

.stCaption, small {{
    color: {MUTED} !important;
}}

.hero {{
    padding: 30px;
    border: 1px solid #315477;
    border-radius: 20px;
    background: linear-gradient(120deg, #153754, #10243a 65%, #102d42);
    margin-bottom: 24px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.22);
}}

.hero h1 {{
    color: #ffffff !important;
    font-size: 39px !important;
    font-weight: 800 !important;
    margin-bottom: 8px;
}}

.hero p {{
    color: #d6e8fa !important;
    font-size: 16px;
}}

.hero-tag {{
    color: #65d5ff;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 12px;
}}

.metric {{
    background: linear-gradient(145deg, #17334e, #10243a);
    border: 1px solid #315477;
    border-radius: 17px;
    padding: 19px;
    min-height: 130px;
    box-shadow: 0 7px 20px rgba(0,0,0,0.18);
}}

.metric-label {{
    color: #c6d9ee;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.5px;
}}

.metric-value {{
    color: #ffffff;
    font-size: 31px;
    font-weight: 800;
    margin-top: 9px;
}}

.metric-note {{
    color: #b5c7dc;
    font-size: 12px;
    margin-top: 5px;
}}

.section-title {{
    color: #ffffff;
    font-size: 23px;
    font-weight: 800;
    margin: 10px 0 12px 0;
}}

.info-card {{
    background: #112941;
    border: 1px solid #315477;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}}

.info-card h4 {{
    color: #ffffff;
    font-size: 17px;
    font-weight: 800;
}}

.info-card p {{
    color: #c6d9ee;
    font-size: 14px;
}}

div[data-testid="stTabs"] button {{
    color: #d9eaff !important;
    font-size: 14px !important;
    font-weight: 800 !important;
}}

div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: #5fd0ff !important;
    border-bottom: 3px solid #38bdf8 !important;
}}

.stButton button {{
    background: linear-gradient(90deg, #0284c7, #0ea5e9);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 800;
    min-height: 42px;
}}

.stButton button:hover {{
    border: 1px solid #7dd3fc;
    color: white;
}}

div[data-testid="stMetric"] {{
    background: #112941;
    border: 1px solid #315477;
    padding: 13px;
    border-radius: 12px;
}}

div[data-testid="stMetricLabel"] {{
    color: #c6d9ee !important;
}}

div[data-testid="stMetricValue"] {{
    color: #ffffff !important;
}}

.stAlert {{
    border-radius: 12px;
}}

hr {{
    border-color: #315477;
}}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 1. MATHEMATICAL FUNCTIONS
# ==========================================================

P_RATED = 300.0


def sun_vector(elevation, azimuth):
    el = np.radians(elevation)
    az = np.radians(azimuth)

    return np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el)
    ])


def panel_normal(tilt, azimuth):
    t = np.radians(tilt)
    a = np.radians(azimuth)

    return np.array([
        np.sin(t) * np.sin(a),
        np.sin(t) * np.cos(a),
        np.cos(t)
    ])


def calculate_power(
    sun_el,
    sun_az,
    tilt,
    panel_az,
    irradiance,
    cloud,
    rated_power
):
    sun = sun_vector(sun_el, sun_az)
    normal = panel_normal(tilt, panel_az)

    cosine = np.dot(sun, normal)
    effective_cosine = max(0, cosine)

    incidence = np.degrees(
        np.arccos(np.clip(effective_cosine, 0, 1))
    )

    power = (
        rated_power
        * irradiance
        * cloud
        * effective_cosine
    )

    return power, incidence, cosine


# ==========================================================
# 2. PARTIAL DERIVATIVES
# ==========================================================

def partial_derivatives(
    sun_el,
    sun_az,
    tilt,
    panel_az,
    irradiance,
    cloud,
    rated_power
):
    el = np.radians(sun_el)
    az = np.radians(sun_az)

    t = np.radians(tilt)
    pa = np.radians(panel_az)

    base = rated_power * irradiance * cloud

    q = (
        np.sin(el) * np.cos(t)
        +
        np.cos(el) * np.sin(t) * np.cos(az - pa)
    )

    if q <= 0:
        return 0.0, 0.0

    dp_dt = base * (
        -np.sin(el) * np.sin(t)
        +
        np.cos(el) * np.cos(t) * np.cos(az - pa)
    ) * np.pi / 180

    dp_da = base * (
        np.cos(el)
        * np.sin(t)
        * np.sin(az - pa)
    ) * np.pi / 180

    return dp_dt, dp_da


# ==========================================================
# 3. OPTIMIZATION ENGINE
# ==========================================================

def optimize_panel(
    sun_el,
    sun_az,
    irradiance,
    cloud,
    rated_power
):
    tilts = np.linspace(0, 90, 91)
    azimuths = np.arange(0, 360, 1)

    T, A = np.meshgrid(tilts, azimuths)

    el = np.radians(sun_el)

    q = (
        np.sin(el) * np.cos(np.radians(T))
        +
        np.cos(el) * np.sin(np.radians(T))
        * np.cos(np.radians(sun_az - A))
    )

    Z = (
        rated_power
        * irradiance
        * cloud
        * np.maximum(0, q)
    )

    idx = np.unravel_index(np.argmax(Z), Z.shape)

    best_tilt = T[idx]
    best_az = A[idx]
    best_power = Z[idx]

    return T, A, Z, best_tilt, best_az, best_power


# ==========================================================
# 4. 3D SOLAR PANEL VISUALIZATION
# ==========================================================

def create_solar_scene(tilt, panel_az, sun_el, sun_az):

    normal = panel_normal(tilt, panel_az)

    az = np.radians(panel_az)

    u = np.array([
        np.cos(az),
        -np.sin(az),
        0
    ])

    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    center = np.array([0, 0, 1.3])

    width = 2.4
    height = 1.5

    corners = np.array([
        center - width/2*u - height/2*v,
        center + width/2*u - height/2*v,
        center + width/2*u + height/2*v,
        center - width/2*u + height/2*v
    ])

    sun = sun_vector(sun_el, sun_az)
    sun_position = center + sun * 3.5

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=corners[:, 0],
        y=corners[:, 1],
        z=corners[:, 2],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color="#087cae",
        opacity=0.98,
        name="Solar Panel"
    ))

    for f in np.linspace(-0.4, 0.4, 5):
        p1 = center + f*width*u - height/2*v
        p2 = center + f*width*u + height/2*v

        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode="lines",
            line=dict(color="#b8ecff", width=4),
            showlegend=False
        ))

    for f in [-0.25, 0, 0.25]:
        p1 = center - width/2*u + f*height*v
        p2 = center + width/2*u + f*height*v

        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode="lines",
            line=dict(color="#b8ecff", width=4),
            showlegend=False
        ))

    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 1.3],
        mode="lines",
        line=dict(color="#b5c7dc", width=8),
        name="Panel Stand"
    ))

    fig.add_trace(go.Scatter3d(
        x=[sun_position[0]],
        y=[sun_position[1]],
        z=[sun_position[2]],
        mode="markers+text",
        marker=dict(size=17, color=YELLOW),
        text=["SUN"],
        textposition="top center",
        textfont=dict(color="#ffffff", size=14),
        name="Sun"
    ))

    fig.add_trace(go.Scatter3d(
        x=[sun_position[0], center[0]],
        y=[sun_position[1], center[1]],
        z=[sun_position[2], center[2]],
        mode="lines",
        line=dict(color=YELLOW, width=5, dash="dash"),
        name="Sunlight"
    ))

    fig.update_layout(
        height=490,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        legend=dict(font=dict(color=TEXT)),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis_title="East / West",
            yaxis_title="North / South",
            zaxis_title="Height",
            xaxis=dict(color=MUTED, gridcolor=BORDER),
            yaxis=dict(color=MUTED, gridcolor=BORDER),
            zaxis=dict(color=MUTED, gridcolor=BORDER),
            aspectmode="cube"
        )
    )

    return fig


# ==========================================================
# 5. AI MODEL TRAINING DATA
# ==========================================================

@st.cache_resource(show_spinner="Training AI prediction model...")
def train_ai_models():

    rng = np.random.default_rng(42)

    n = 7000

    # Generate simulated solar conditions

    sun_el = rng.uniform(5, 90, n)
    sun_az = rng.uniform(0, 359, n)

    tilt = rng.uniform(0, 90, n)
    panel_az = rng.uniform(0, 359, n)

    irradiance = rng.uniform(0.1, 1.0, n)
    cloud = rng.uniform(0.2, 1.0, n)

    rated = rng.choice(
        [150, 200, 250, 300, 350, 400, 500],
        n
    )

    el = np.radians(sun_el)
    az = np.radians(sun_az)
    t = np.radians(tilt)
    pa = np.radians(panel_az)

    cosine = (
        np.sin(el) * np.cos(t)
        +
        np.cos(el) * np.sin(t) * np.cos(az - pa)
    )

    cosine = np.maximum(0, cosine)

    theoretical_power = (
        rated * irradiance * cloud * cosine
    )

    # Small noise makes the training data more realistic

    noise = rng.normal(0, 1.2, n)

    target_power = np.maximum(
        0,
        theoretical_power + noise
    )

    X = np.column_stack([
        sun_el,
        sun_az,
        tilt,
        panel_az,
        irradiance,
        cloud,
        rated
    ])

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, target_power)

    # Train anomaly detection model on normal operating data

    normal_noise = rng.normal(0, 0.025, n)

    normal_actual = np.maximum(
        0,
        theoretical_power * (1 + normal_noise)
    )

    deviation = np.abs(
        normal_actual - theoretical_power
    )

    anomaly_X = np.column_stack([
        theoretical_power,
        normal_actual,
        deviation,
        irradiance,
        cloud
    ])

    anomaly_model = IsolationForest(
        n_estimators=100,
        contamination=0.04,
        random_state=42
    )

    anomaly_model.fit(anomaly_X)

    return model, anomaly_model


# ==========================================================
# 6. AI PREDICTION FUNCTION
# ==========================================================

def ai_predict(
    model,
    sun_el,
    sun_az,
    tilt,
    panel_az,
    irradiance,
    cloud,
    rated_power
):

    features = np.array([[
        sun_el,
        sun_az,
        tilt,
        panel_az,
        irradiance,
        cloud,
        rated_power
    ]])

    prediction = float(model.predict(features)[0])

    maximum_possible = rated_power * irradiance * cloud

    return float(np.clip(
        prediction,
        0,
        maximum_possible
    ))


# ==========================================================
# 7. SIDEBAR CONTROLS
# ==========================================================

st.sidebar.title("☀️ SolarIQ Controls")

st.sidebar.caption("Live solar optimization controls")

st.sidebar.divider()

st.sidebar.subheader("Sun Position")

sun_elevation = st.sidebar.slider(
    "Sun Elevation",
    1, 90, 48
)

sun_azimuth = st.sidebar.slider(
    "Sun Azimuth",
    0, 359, 180
)

st.sidebar.divider()

st.sidebar.subheader("Solar Panel Position")

panel_tilt = st.sidebar.slider(
    "Panel Tilt",
    0, 90, 30
)

panel_azimuth = st.sidebar.slider(
    "Panel Azimuth",
    0, 359, 180
)

st.sidebar.divider()

st.sidebar.subheader("Environmental Conditions")

irradiance = st.sidebar.slider(
    "Irradiance Factor",
    0.1, 1.0, 0.9,
    step=0.05
)

cloud = st.sidebar.slider(
    "Cloud Factor",
    0.1, 1.0, 0.95,
    step=0.05
)

rated_power = st.sidebar.number_input(
    "Panel Rated Power (W)",
    min_value=50,
    max_value=2000,
    value=300,
    step=50
)

P_RATED = float(rated_power)


# ==========================================================
# 8. LIVE CALCULATIONS
# ==========================================================

power, incidence, cosine = calculate_power(
    sun_elevation,
    sun_azimuth,
    panel_tilt,
    panel_azimuth,
    irradiance,
    cloud,
    P_RATED
)

dp_dt, dp_da = partial_derivatives(
    sun_elevation,
    sun_azimuth,
    panel_tilt,
    panel_azimuth,
    irradiance,
    cloud,
    P_RATED
)

gradient = np.sqrt(dp_dt**2 + dp_da**2)

T, A, Z, best_tilt, best_az, best_power = optimize_panel(
    sun_elevation,
    sun_azimuth,
    irradiance,
    cloud,
    P_RATED
)

gain = max(0, best_power - power)

efficiency = (
    power / P_RATED * 100
    if P_RATED else 0
)


# ==========================================================
# 9. LOAD AI MODELS
# ==========================================================

ai_model, anomaly_model = train_ai_models()

ai_power = ai_predict(
    ai_model,
    sun_elevation,
    sun_azimuth,
    panel_tilt,
    panel_azimuth,
    irradiance,
    cloud,
    P_RATED
)

feature_names = [
    "Sun Elevation",
    "Sun Azimuth",
    "Panel Tilt",
    "Panel Azimuth",
    "Irradiance",
    "Cloud Factor",
    "Rated Power"
]

feature_importance = ai_model.feature_importances_


# ==========================================================
# 10. POWER HISTORY
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

signature = (
    sun_elevation,
    sun_azimuth,
    panel_tilt,
    panel_azimuth,
    irradiance,
    cloud,
    rated_power
)

if st.session_state.get("last_signature") != signature:

    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Power": power,
        "Tilt": panel_tilt,
        "Azimuth": panel_azimuth
    })

    st.session_state.history = st.session_state.history[-50:]

    st.session_state.last_signature = signature


# ==========================================================
# 12. KPI CARDS
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div class="metric">
    <div class="metric-label">LIVE POWER OUTPUT</div>
    <div class="metric-value" style="color:{YELLOW}">
        {power:.1f} W
    </div>
    <div class="metric-note">{efficiency:.1f}% panel capacity</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric">
    <div class="metric-label">OPTIMIZED POWER</div>
    <div class="metric-value" style="color:{GREEN}">
        {best_power:.1f} W
    </div>
    <div class="metric-note">Best tested orientation</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric">
    <div class="metric-label">AI POWER PREDICTION</div>
    <div class="metric-value" style="color:{CYAN}">
        {ai_power:.1f} W
    </div>
    <div class="metric-note">Random Forest prediction</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="metric">
    <div class="metric-label">POTENTIAL POWER GAIN</div>
    <div class="metric-value" style="color:{PURPLE}">
        {gain:.1f} W
    </div>
    <div class="metric-note">Available after optimization</div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# 13. DASHBOARD TABS
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "☀️ 3D Live Simulator",
    "∂ Calculus Laboratory",
    "⌖ Optimization Center",
    "🤖 AI Intelligence"
])


# ==========================================================
# TAB 1 - LIVE SIMULATOR
# ==========================================================

with tab1:

    left, right = st.columns([1.25, 0.75])

    with left:

        st.markdown(
            '<div class="section-title">3D Solar Panel Visualization</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Adjust the sidebar controls to change panel orientation and sunlight direction."
        )

        fig = create_solar_scene(
            panel_tilt,
            panel_azimuth,
            sun_elevation,
            sun_azimuth
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="section-title">Live Operating Conditions</div>',
            unsafe_allow_html=True
        )

        st.metric("Panel Tilt", f"{panel_tilt}°")
        st.metric("Panel Azimuth", f"{panel_azimuth}°")
        st.metric("Sun Elevation", f"{sun_elevation}°")
        st.metric("Sun Azimuth", f"{sun_azimuth}°")

        if cosine > 0:
            st.success("Direct sunlight is reaching the panel.")
        else:
            st.warning("The panel front is not facing the sun.")

        st.markdown("### Power History")

        history_df = pd.DataFrame(st.session_state.history)

        if len(history_df) > 1:

            history_fig = go.Figure()

            history_fig.add_trace(go.Scatter(
                x=history_df["Time"],
                y=history_df["Power"],
                mode="lines+markers",
                line=dict(color=GREEN, width=3),
                marker=dict(size=7),
                name="Power"
            ))

            history_fig.update_layout(
                height=290,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT),
                xaxis_title="Time",
                yaxis_title="Power (W)"
            )

            st.plotly_chart(
                history_fig,
                use_container_width=True
            )

        else:
            st.info("Change a control to build your power history.")


# ==========================================================
# TAB 2 - CALCULUS LABORATORY
# ==========================================================

with tab2:

    st.markdown(
        '<div class="section-title">Multivariable Calculus Laboratory</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Solar power is modelled as a function of two independent variables: panel tilt and panel azimuth."
    )

    st.latex(
        r"P(x,y)=P_{rated}IC\max(0,"
        r"\sin(e)\cos(x)+\cos(e)\sin(x)\cos(a-y))"
    )

    st.caption(
        "x = panel tilt, y = panel azimuth, e = sun elevation, "
        "a = sun azimuth. The mathematical formula uses radians."
    )

    st.divider()

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "∂P / ∂x",
        f"{dp_dt:.3f} W/degree"
    )

    m2.metric(
        "∂P / ∂y",
        f"{dp_da:.3f} W/degree"
    )

    m3.metric(
        "Gradient Magnitude",
        f"{gradient:.3f}"
    )

    st.markdown("### Gradient Vector")

    st.latex(
        r"\nabla P(x,y)="
        r"\left(\frac{\partial P}{\partial x},"
        r"\frac{\partial P}{\partial y}\right)"
    )

    st.write(
        f"Gradient = ({dp_dt:.4f}, {dp_da:.4f})"
    )

    st.info(
        "The gradient indicates the direction of the steepest local increase in power."
    )

    st.divider()

    st.markdown("### Derivative Interpretation")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Tilt Derivative")

        st.metric(
            "∂P / ∂x",
            f"{dp_dt:.4f} W/degree"
        )

        if dp_dt > 0:
            st.success("Increasing tilt locally increases power.")
        elif dp_dt < 0:
            st.warning("Decreasing tilt locally increases power.")
        else:
            st.info("Power is locally stationary with respect to tilt.")

    with col2:

        st.markdown("#### Azimuth Derivative")

        st.metric(
            "∂P / ∂y",
            f"{dp_da:.4f} W/degree"
        )

        if dp_da > 0:
            st.success("Increasing azimuth locally increases power.")
        elif dp_da < 0:
            st.warning("Decreasing azimuth locally increases power.")
        else:
            st.info("Power is locally stationary with respect to azimuth.")

    st.caption(
        "Derivatives are expressed in watts per degree. A zero gradient alone does not prove a global maximum."
    )


# ==========================================================
# TAB 3 - OPTIMIZATION CENTER
# ==========================================================

with tab3:

    st.markdown(
        '<div class="section-title">3D Power Optimization Surface</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Explore how different tilt and azimuth combinations affect solar power."
    )

    surface = go.Figure(data=[
        go.Surface(
            x=T,
            y=A,
            z=Z,
            colorscale="Turbo",
            colorbar=dict(
                title=dict(text="Power (W)", font=dict(color=TEXT)),
                tickfont=dict(color=TEXT)
            )
        )
    ])

    surface.add_trace(go.Scatter3d(
        x=[panel_tilt],
        y=[panel_azimuth],
        z=[power],
        mode="markers",
        marker=dict(size=7, color=YELLOW),
        name="Current Position"
    ))

    surface.add_trace(go.Scatter3d(
        x=[best_tilt],
        y=[best_az],
        z=[best_power],
        mode="markers+text",
        text=["MAXIMUM"],
        textposition="top center",
        textfont=dict(color="#ffffff"),
        marker=dict(size=9, color=RED, symbol="diamond"),
        name="Optimal Position"
    ))

    surface.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        scene=dict(
            xaxis_title="Panel Tilt",
            yaxis_title="Panel Azimuth",
            zaxis_title="Power Output",
            xaxis=dict(color=MUTED, gridcolor=BORDER),
            yaxis=dict(color=MUTED, gridcolor=BORDER),
            zaxis=dict(color=MUTED, gridcolor=BORDER)
        )
    )

    st.plotly_chart(
        surface,
        use_container_width=True
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Level Curves / Contour Plot</div>',
        unsafe_allow_html=True
    )

    contour = go.Figure(data=go.Contour(
        x=np.linspace(0, 90, 91),
        y=np.arange(360),
        z=Z,
        colorscale="Turbo",
        contours=dict(showlabels=True),
        colorbar=dict(title="Power (W)")
    ))

    contour.add_trace(go.Scatter(
        x=[panel_tilt],
        y=[panel_azimuth],
        mode="markers",
        marker=dict(size=13, color=YELLOW),
        name="Current Position"
    ))

    contour.add_trace(go.Scatter(
        x=[best_tilt],
        y=[best_az],
        mode="markers",
        marker=dict(size=14, color=RED, symbol="star"),
        name="Maximum Position"
    ))

    contour.update_layout(
        height=470,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis_title="Panel Tilt",
        yaxis_title="Panel Azimuth"
    )

    st.plotly_chart(
        contour,
        use_container_width=True
    )

    st.divider()

    st.markdown("### Optimization Results")

    r1, r2, r3 = st.columns(3)

    r1.metric("Optimal Tilt", f"{best_tilt:.1f}°")
    r2.metric("Optimal Azimuth", f"{best_az:.1f}°")
    r3.metric("Maximum Power", f"{best_power:.1f} W")

    st.success(
        f"Best tested orientation: tilt {best_tilt:.1f}° and azimuth {best_az:.1f}°."
    )


# ==========================================================
# TAB 4 - AI INTELLIGENCE
# ==========================================================

with tab4:

    st.markdown(
        '<div class="section-title">AI Solar Intelligence</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Machine Learning models analyse solar operating conditions, predict power output and identify unusual performance."
    )

    st.divider()

    # ---------------- AI POWER PREDICTION ----------------

    st.markdown("### 1. AI Power Prediction")

    ai_col1, ai_col2 = st.columns(2)

    with ai_col1:

        st.markdown(
            '<div class="info-card"><h4>Machine Learning Prediction</h4>'
            '<p>Random Forest Regression estimates solar power using seven input features.</p></div>',
            unsafe_allow_html=True
        )

        st.metric(
            "AI Predicted Power",
            f"{ai_power:.2f} W"
        )

        st.metric(
            "Mathematical Model Power",
            f"{power:.2f} W"
        )

        prediction_difference = abs(ai_power - power)

        st.metric(
            "Prediction Difference",
            f"{prediction_difference:.2f} W"
        )

        if prediction_difference < max(5, P_RATED * 0.05):
            st.success("AI prediction is close to the mathematical model.")
        else:
            st.info("AI prediction differs from the mathematical model.")

    with ai_col2:

        comparison = go.Figure()

        comparison.add_trace(go.Bar(
            x=["Mathematical Model", "AI Prediction"],
            y=[power, ai_power],
            marker_color=[CYAN, PURPLE],
            text=[f"{power:.1f} W", f"{ai_power:.1f} W"],
            textposition="outside"
        ))

        comparison.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
            yaxis_title="Power (W)",
            showlegend=False
        )

        st.plotly_chart(
            comparison,
            use_container_width=True
        )

    st.divider()

    # ---------------- ANOMALY DETECTION ----------------

    st.markdown("### 2. AI Anomaly Detection")

    st.write(
        "Enter a measured power value. The AI model will check whether it looks unusual compared with expected production."
    )

    default_actual = min(
        int(P_RATED),
        max(0, int(round(power)))
    )

    actual_power = st.number_input(
        "Measured Power Output (W)",
        min_value=0.0,
        max_value=float(P_RATED),
        value=float(default_actual),
        step=1.0,
        key="measured_power"
    )

    deviation = abs(actual_power - power)

    anomaly_features = np.array([[
        power,
        actual_power,
        deviation,
        irradiance,
        cloud
    ]])

    anomaly_result = anomaly_model.predict(anomaly_features)[0]

    anomaly_score = anomaly_model.decision_function(
        anomaly_features
    )[0]

    anomaly_col1, anomaly_col2, anomaly_col3 = st.columns(3)

    anomaly_col1.metric(
        "Expected Power",
        f"{power:.1f} W"
    )

    anomaly_col2.metric(
        "Measured Power",
        f"{actual_power:.1f} W"
    )

    anomaly_col3.metric(
        "Power Difference",
        f"{deviation:.1f} W"
    )

    if anomaly_result == -1:
        st.error(
            "AI Alert: Unusual power behaviour detected. Check the panel conditions and measurement."
        )
    else:
        st.success(
            "AI Status: Power measurement appears within the learned normal operating pattern."
        )

    st.caption(
        f"Isolation Forest anomaly score: {anomaly_score:.4f}"
    )

    st.divider()

    # ---------------- FEATURE IMPORTANCE ----------------

    st.markdown("### 3. AI Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    }).sort_values(
        "Importance",
        ascending=True
    )

    importance_fig = go.Figure(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h",
        marker=dict(color=CYAN)
    ))

    importance_fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis_title="Relative Feature Importance",
        yaxis_title="Input Feature"
    )

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )

    st.divider()

    # ---------------- AI MODEL INFORMATION ----------------

    st.markdown("### 4. Active AI Models")

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.markdown("""
        <div class="info-card">
            <h4>Random Forest Regressor</h4>
            <p><b>Task:</b> Solar power prediction</p>
            <p><b>Input:</b> Sun position, panel orientation, irradiance, cloud factor and rated power</p>
            <p><b>Output:</b> Predicted power in watts</p>
            <p><b>Status:</b> Model trained and active</p>
        </div>
        """, unsafe_allow_html=True)

    with model_col2:

        st.markdown("""
        <div class="info-card">
            <h4>Isolation Forest</h4>
            <p><b>Task:</b> Anomaly detection</p>
            <p><b>Input:</b> Expected power, measured power, deviation and environmental conditions</p>
            <p><b>Output:</b> Normal or unusual measurement</p>
            <p><b>Status:</b> Model trained and active</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "These AI models are trained on simulated data for this prototype. Real-world accuracy requires measured solar production data."
    )


# ==========================================================
# END
# ==========================================================
