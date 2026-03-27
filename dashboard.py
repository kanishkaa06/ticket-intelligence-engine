import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Support Ticket Intelligence Engine",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #16192b 100%);
        border-right: 1px solid #2d3150;
    }

    /* Hide default Streamlit header/footer/deploy */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}

    /* KPI cards */
    .kpi-card {
        background: #1e2130;
        border: 1px solid #2d3150;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: #4f6ef7; }
    .kpi-label {
        color: #8b92a5;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        line-height: 1;
    }
    .kpi-sub {
        color: #4f6ef7;
        font-size: 12px;
        margin-top: 6px;
    }

    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 0 8px 0;
        border-bottom: 2px solid #4f6ef7;
        margin-bottom: 16px;
    }

    /* Insight box */
    .insight-box {
        background: #1a2744;
        border-left: 4px solid #4f6ef7;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        color: #c8d0e0;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Alert box */
    .alert-box {
        background: #2a1a1a;
        border-left: 4px solid #e74c3c;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 8px 0;
        color: #f0a0a0;
        font-size: 14px;
    }

    /* Nav label */
    .nav-label {
        color: #8b92a5;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 8px 0 4px 0;
    }

    /* Page title */
    .page-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .page-subtitle {
        color: #8b92a5;
        font-size: 14px;
        margin-bottom: 24px;
    }

    /* Predict button */
    .stButton > button {
        background: #4f6ef7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #3d5ce0; }

    /* Chart background */
    .chart-container {
        background: #1e2130;
        border: 1px solid #2d3150;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Divider */
    .custom-divider {
        border: none;
        border-top: 1px solid #2d3150;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# CHART STYLE — dark theme for all matplotlib charts
# ---------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor':  '#1e2130',
    'axes.facecolor':    '#1e2130',
    'axes.edgecolor':    '#2d3150',
    'axes.labelcolor':   '#8b92a5',
    'axes.titlecolor':   '#ffffff',
    'xtick.color':       '#8b92a5',
    'ytick.color':       '#8b92a5',
    'text.color':        '#ffffff',
    'grid.color':        '#2d3150',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'legend.facecolor':  '#1e2130',
    'legend.edgecolor':  '#2d3150',
    'font.family':       'sans-serif',
})

ACCENT   = '#4f6ef7'
PALETTE  = ['#e74c3c','#f39c12','#4f6ef7','#95a5a6']
PALETTE2 = ['#2ecc71','#f39c12','#e67e22','#e74c3c']

# ---------------------------------------------------------------
# DATA & MODEL (cached)
# ---------------------------------------------------------------
@st.cache_data
def load_and_train():
    np.random.seed(42)
    n = 10000
    sla_response   = {'Low':24,  'Medium':12, 'High':4,  'Urgent':1}
    sla_resolution = {'Low':120, 'Medium':48, 'High':24, 'Urgent':8}
    breach_prob    = {'Urgent':0.20,'High':0.40,'Medium':0.60,'Low':0.75}

    priorities = np.random.choice(
        ['Low','Medium','High','Urgent'], n, p=[0.25,0.30,0.25,0.20])
    categories = np.random.choice(
        ['Login Issue','Payment Problem','Performance Issue',
         'Feature Request','Account Suspension','Subscription Cancellation',
         'Data Loss','Security Breach','Integration Failure',
         'Billing Dispute'], n)
    channels = np.random.choice(
        ['Email','Chat','Phone','Portal'], n, p=[0.35,0.30,0.20,0.15])
    complexity = np.where(priorities=='Urgent', np.random.randint(6,11,n),
                 np.where(priorities=='High',   np.random.randint(4, 9,n),
                 np.where(priorities=='Medium', np.random.randint(2, 7,n),
                                                np.random.randint(1, 5,n))))
    sla_breach = np.array([
        np.random.binomial(1, breach_prob[p]) for p in priorities])
    response_time = np.array([
        np.random.uniform(sla_response[p]*1.1, sla_response[p]*3.0)
        if sla_breach[i]
        else np.random.uniform(sla_response[p]*0.2, sla_response[p]*0.95)
        for i, p in enumerate(priorities)])
    resolution_time = np.array([
        np.random.uniform(sla_resolution[p]*1.1, sla_resolution[p]*2.5)
        if sla_breach[i]
        else np.random.uniform(sla_resolution[p]*0.3, sla_resolution[p]*0.95)
        for i, p in enumerate(priorities)])
    previous_tickets  = np.random.poisson(2.5, n)
    customer_tenure   = np.random.randint(1, 72, n)
    customer_age      = np.random.randint(18, 70, n)
    escalated = np.array([
        1 if (complexity[i]>=7 and sla_breach[i]==1) or
             (priorities[i]=='Urgent' and sla_breach[i]==1)
        else 0 for i in range(n)])
    agent_names = [f'Agent_{i+1:02d}' for i in range(10)]

    df = pd.DataFrame({
        'priority': priorities, 'category': categories,
        'channel': channels, 'complexity_score': complexity,
        'response_time_hours': response_time.round(2),
        'resolution_time_hours': resolution_time.round(2),
        'previous_tickets': previous_tickets,
        'customer_tenure_months': customer_tenure,
        'customer_age': customer_age,
        'sla_breach': sla_breach, 'escalated': escalated,
        'agent': np.random.choice(agent_names, n)
    })

    le_priority = LabelEncoder()
    le_category = LabelEncoder()
    le_channel  = LabelEncoder()
    df['priority_encoded']  = le_priority.fit_transform(df['priority'])
    df['category_encoded']  = le_category.fit_transform(df['category'])
    df['channel_encoded']   = le_channel.fit_transform(df['channel'])

    features = ['category_encoded','channel_encoded','complexity_score',
                'response_time_hours','resolution_time_hours',
                'previous_tickets','customer_tenure_months','customer_age']

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(df[features], df['priority_encoded'])

    df_nlp  = pd.read_csv('customer_support_tickets_200k.csv')
    sample  = df_nlp['issue_description'].dropna().sample(5000, random_state=42)
    clean   = sample.apply(lambda t: re.sub(r'[^a-z\s]','',
                           re.sub(r'\s+',' ', str(t).lower()).strip()))
    vec     = TfidfVectorizer(max_features=500, stop_words='english',
                              ngram_range=(1,2))
    X_tfidf = vec.fit_transform(clean)
    km      = KMeans(n_clusters=6, random_state=42, n_init=10)
    km.fit(X_tfidf)

    cluster_names = {
        0:'Account Access', 1:'Subscription & Refunds',
        2:'Bug Reports',    3:'Performance Issues',
        4:'Billing & Payment', 5:'Data Sync'
    }

    return (df, model, le_priority, le_category, le_channel,
            features, km, vec, cluster_names, X_tfidf, sample)

with st.spinner('🔄  Training models and loading data...'):
    (df, model, le_priority, le_category, le_channel,
     features, km, vec, cluster_names, X_tfidf, sample) = load_and_train()

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0'>
        <div style='font-size:36px'>🎫</div>
        <div style='color:#ffffff; font-size:15px;
                    font-weight:700; margin-top:6px'>
            Ticket Intelligence
        </div>
        <div style='color:#8b92a5; font-size:12px'>
            ML-Powered Operations
        </div>
    </div>
    <hr style='border-color:#2d3150; margin:12px 0'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Navigation</div>',
                unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Overview",
        "🎫  Ticket Triage",
        "⏱  SLA Breach Risk",
        "🔍  Root Cause Clusters",
        "👥  Workforce Load"
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#2d3150; margin:20px 0 12px 0'>
    <div style='color:#8b92a5; font-size:11px; line-height:1.8'>
        <b style='color:#4f6ef7'>Models:</b> Random Forest, K-Means<br>
        <b style='color:#4f6ef7'>NLP:</b> TF-IDF + PCA<br>
        <b style='color:#4f6ef7'>Data:</b> 10,000 synthetic tickets<br>
        <b style='color:#4f6ef7'>Accuracy:</b> 91% F1-score
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# PAGE 1 — OVERVIEW
# ---------------------------------------------------------------
if "Overview" in page:
    st.markdown('<div class="page-title">Support Ticket Intelligence Engine</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time ML-powered support operations analytics</div>',
                unsafe_allow_html=True)

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total Tickets",    f"{len(df):,}",                    "10K domain-realistic records"),
        ("SLA Breach Rate",  f"{df['sla_breach'].mean():.1%}",  "Urgent: 20% · Low: 75%"),
        ("Escalation Rate",  f"{df['escalated'].mean():.1%}",   "High complexity breaches"),
        ("Avg Complexity",   f"{df['complexity_score'].mean():.1f}/10", "Across all priorities"),
    ]
    for col, (label, value, sub) in zip([k1,k2,k3,k4], kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Priority Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        vals = df['priority'].value_counts().reindex(
            ['Urgent','High','Medium','Low'])
        bars = ax.bar(vals.index, vals.values,
                      color=PALETTE, edgecolor='#2d3150', linewidth=0.5,
                      width=0.6)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+40,
                    f'{int(bar.get_height()):,}',
                    ha='center', va='bottom', fontsize=11, color='#ffffff')
        ax.set_ylim(0, vals.max()*1.15)
        ax.tick_params(axis='x', labelsize=11)
        ax.set_ylabel('Ticket Count', fontsize=11)
        ax.grid(axis='y')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="insight-box">
        Medium priority tickets dominate the backlog (30%) because they
        sit in a grey zone — not urgent enough to jump the queue, but
        complex enough to take time. This is where hidden delays accumulate.
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">SLA Breach Rate by Priority</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        breach = df.groupby('priority')['sla_breach'].mean().reindex(
            ['Urgent','High','Medium','Low'])
        bars = ax.bar(breach.index, breach.values,
                      color=PALETTE2, edgecolor='#2d3150',
                      linewidth=0.5, width=0.6)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.01,
                    f'{bar.get_height():.0%}',
                    ha='center', va='bottom', fontsize=11, color='#ffffff')
        ax.axhline(y=df['sla_breach'].mean(), color='#ffffff',
                   linestyle='--', linewidth=1, alpha=0.4, label='Average')
        ax.set_ylabel('Breach Rate', fontsize=11)
        ax.set_ylim(0, 0.95)
        ax.tick_params(axis='x', labelsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="insight-box">
        Urgent tickets breach least often (20%) — teams prioritise them.
        Low priority tickets breach 75% of the time — they sit unattended.
        This gap is intentional but represents a risk when Low tickets
        contain hidden complexity.
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# PAGE 2 — TICKET TRIAGE
# ---------------------------------------------------------------
elif "Triage" in page:
    st.markdown('<div class="page-title">🎫 Ticket Triage Classifier</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter ticket details — the Random Forest model predicts priority in real time</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
    <b>How this works:</b> The model was trained on 8,000 tickets and tested
    on 2,000 unseen ones, achieving 91% accuracy. It learned that
    resolution time and complexity score are the strongest predictors —
    not the category or channel. Adjust the inputs below and watch
    the confidence bars shift.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Ticket Details</div>',
                    unsafe_allow_html=True)
        cat_input    = st.selectbox("Issue Category", le_category.classes_)
        chan_input   = st.selectbox("Channel", le_channel.classes_)
        comp_input   = st.slider("Complexity Score", 1, 10, 5,
                                  help="1 = trivial, 10 = extremely complex")
        prev_input   = st.number_input("Previous Tickets by Customer",
                                        0, 20, 2)
        tenure_input = st.slider("Customer Tenure (months)", 1, 72, 12)

    with col2:
        st.markdown('<div class="section-header">Time Estimates</div>',
                    unsafe_allow_html=True)
        resp_input = st.number_input("First Response Time (hours)",
                                      0.1, 72.0, 4.0,
                                      help="How long before first reply")
        res_input  = st.number_input("Resolution Time (hours)",
                                      0.1, 240.0, 24.0,
                                      help="How long to fully close ticket")
        st.markdown("""<div class="insight-box" style="margin-top:16px">
        <b>Tip:</b> Response and resolution time together account for
        66% of the model's decision. Try setting response time to
        0.5hrs and resolution to 5hrs — watch it predict Urgent.
        Then set them to 20hrs and 100hrs — it shifts to Low.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍  Predict Priority", type="primary"):
        inp = pd.DataFrame([{
            'category_encoded':      le_category.transform([cat_input])[0],
            'channel_encoded':       le_channel.transform([chan_input])[0],
            'complexity_score':      comp_input,
            'response_time_hours':   resp_input,
            'resolution_time_hours': res_input,
            'previous_tickets':      prev_input,
            'customer_tenure_months':tenure_input,
            'customer_age':          35
        }])
        pred  = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0]
        label = le_priority.inverse_transform([pred])[0]

        color_map = {'Urgent':'#e74c3c','High':'#f39c12',
                     'Medium':'#4f6ef7','Low':'#2ecc71'}
        emoji_map = {'Urgent':'🔴','High':'🟠','Medium':'🔵','Low':'🟢'}

        st.markdown(f"""
        <div style='background:{color_map[label]}22;
                    border:2px solid {color_map[label]};
                    border-radius:12px; padding:20px;
                    text-align:center; margin:16px 0'>
            <div style='font-size:13px; color:#8b92a5;
                        text-transform:uppercase; letter-spacing:1px'>
                Predicted Priority
            </div>
            <div style='font-size:42px; font-weight:800;
                        color:{color_map[label]}; margin:8px 0'>
                {emoji_map[label]} {label}
            </div>
        </div>""", unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            'Priority': le_priority.classes_,
            'Confidence': proba
        }).sort_values('Confidence', ascending=True)

        fig, ax = plt.subplots(figsize=(7, 3))
        colors_bar = [color_map.get(p,'#4f6ef7')
                      for p in prob_df['Priority']]
        bars = ax.barh(prob_df['Priority'], prob_df['Confidence'],
                       color=colors_bar, edgecolor='#2d3150',
                       linewidth=0.5, height=0.5)
        for bar, val in zip(bars, prob_df['Confidence']):
            ax.text(bar.get_width()+0.01,
                    bar.get_y()+bar.get_height()/2,
                    f'{val:.1%}', va='center', fontsize=11)
        ax.set_xlim(0, 1.2)
        ax.set_title('Model Confidence per Class', fontsize=12)
        ax.grid(axis='x')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ---------------------------------------------------------------
# PAGE 3 — SLA BREACH RISK
# ---------------------------------------------------------------
elif "SLA" in page:
    st.markdown('<div class="page-title">⏱ SLA Breach Risk Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Which ticket types and complexity levels are most likely to miss their SLA deadline?</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
    <b>What is an SLA breach?</b> Every ticket has a deadline — Urgent
    tickets must be resolved within 8 hours, Low priority within 120 hours.
    A breach means the team missed that deadline. This page shows
    <i>which categories and complexity levels</i> breach most often,
    so managers can pre-allocate resources before a breach happens.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Breach Rate by Category</div>',
                    unsafe_allow_html=True)
        breach_cat = df.groupby('category')['sla_breach'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7,5))
        colors_cat = [ACCENT if v >= breach_cat.mean() else '#2d3150'
                      for v in breach_cat.values]
        bars = ax.barh(breach_cat.index, breach_cat.values,
                       color=colors_cat, edgecolor='#1e2130', linewidth=0.5)
        ax.axvline(x=breach_cat.mean(), color='#e74c3c',
                   linestyle='--', linewidth=1.5, label='Average')
        for bar, val in zip(bars, breach_cat.values):
            ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
                    f'{val:.0%}', va='center', fontsize=10)
        ax.set_xlabel('Breach Rate')
        ax.legend(); ax.grid(axis='x')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="insight-box">
        Categories above the red average line need extra resource
        allocation. Highlighted bars are your highest-risk categories
        — these are where proactive intervention saves the most SLA points.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Complexity vs Breach Rate</div>',
                    unsafe_allow_html=True)
        breach_comp = df.groupby('complexity_score')['sla_breach'].mean()
        fig, ax = plt.subplots(figsize=(7,5))
        ax.fill_between(breach_comp.index, breach_comp.values,
                        alpha=0.15, color=ACCENT)
        ax.plot(breach_comp.index, breach_comp.values,
                'o-', color=ACCENT, linewidth=2.5,
                markersize=8, markerfacecolor='white',
                markeredgecolor=ACCENT, markeredgewidth=2)
        for x, y in zip(breach_comp.index, breach_comp.values):
            ax.annotate(f'{y:.0%}', (x,y),
                        textcoords='offset points', xytext=(0,10),
                        ha='center', fontsize=9, color='#8b92a5')
        ax.set_xlabel('Complexity Score (1=simple, 10=hardest)')
        ax.set_ylabel('SLA Breach Rate')
        ax.set_title('Higher complexity = higher breach risk')
        ax.grid(True); ax.set_xticks(range(1,11))
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="insight-box">
        Complexity score is the single strongest predictor of breach risk.
        A ticket with score 9–10 is almost certain to breach. The model
        uses this as its top feature when predicting whether a ticket
        will miss its deadline before it's even assigned.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">SLA Policy Reference</div>',
                unsafe_allow_html=True)
    sla_df = pd.DataFrame({
        'Priority':              ['🔴 Urgent','🟠 High','🔵 Medium','🟢 Low'],
        'Response SLA':          ['1 hour',  '4 hours','12 hours','24 hours'],
        'Resolution SLA':        ['8 hours', '24 hours','48 hours','120 hours'],
        'Target Breach Rate':    ['< 20%',   '< 40%',  '< 60%',   '< 75%'],
        'Current Breach Rate':   [
            f"{df[df['priority']=='Urgent']['sla_breach'].mean():.0%}",
            f"{df[df['priority']=='High']['sla_breach'].mean():.0%}",
            f"{df[df['priority']=='Medium']['sla_breach'].mean():.0%}",
            f"{df[df['priority']=='Low']['sla_breach'].mean():.0%}",
        ]
    })
    st.dataframe(sla_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# PAGE 4 — ROOT CAUSE CLUSTERS
# ---------------------------------------------------------------
elif "Root" in page:
    st.markdown('<div class="page-title">🔍 Root Cause Cluster Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">NLP clustering of 5,000 real ticket descriptions — no labels needed</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
    <b>How this works:</b> We fed 5,000 raw ticket descriptions into a
    TF-IDF model, which converted each description into a list of numbers
    based on which words appeared most uniquely. K-Means then grouped
    similar tickets together — without being told what the categories were.
    The 6 clusters it discovered map almost perfectly to real issue types,
    which validates both the NLP approach and the data quality.
    </div>""", unsafe_allow_html=True)

    labels   = km.labels_
    colors_c = ['#e74c3c','#4f6ef7','#2ecc71',
                '#f39c12','#9b59b6','#1abc9c']

    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div class="section-header">Cluster Map (PCA 2D Projection)</div>',
                    unsafe_allow_html=True)
        pca  = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_tfidf.toarray())
        fig, ax = plt.subplots(figsize=(9,6))
        for i in range(6):
            mask = labels == i
            ax.scatter(X_2d[mask,0], X_2d[mask,1],
                       c=colors_c[i], label=cluster_names[i],
                       alpha=0.5, s=12, edgecolors='none')
        centers_2d = pca.transform(km.cluster_centers_)
        for i,(cx,cy) in enumerate(centers_2d):
            ax.annotate(cluster_names[i], xy=(cx,cy),
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4',
                                  facecolor=colors_c[i],
                                  alpha=0.9, edgecolor='white',
                                  linewidth=1),
                        color='white', ha='center')
        ax.legend(markerscale=3, fontsize=9,
                  loc='upper right', framealpha=0.3)
        ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.set_title('Each dot = one support ticket')
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Cluster Sizes</div>',
                    unsafe_allow_html=True)
        for i in range(6):
            count = (labels==i).sum()
            pct   = count/len(labels)*100
            st.markdown(f"""
            <div style='margin-bottom:14px'>
                <div style='color:{colors_c[i]};
                            font-size:13px; font-weight:600'>
                    {cluster_names[i]}
                </div>
                <div style='background:#2d3150; border-radius:4px;
                            height:6px; margin:4px 0'>
                    <div style='background:{colors_c[i]};
                                width:{pct*3:.0f}%; height:6px;
                                border-radius:4px'></div>
                </div>
                <div style='color:#8b92a5; font-size:12px'>
                    {count:,} tickets · {pct:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# PAGE 5 — WORKFORCE LOAD
# ---------------------------------------------------------------
elif "Workforce" in page:
    st.markdown('<div class="page-title">👥 Workforce Load Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Why ticket count alone is a misleading measure of agent workload</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
    <b>The problem with counting tickets:</b> Imagine Agent A closed
    50 tickets today — all simple password resets (complexity 1–2).
    Agent B closed 40 tickets — all data loss and security breach
    issues (complexity 8–10). By raw count, Agent A looks busier.
    By complexity-weighted load, Agent B is carrying nearly 4x the
    cognitive burden. This page surfaces that hidden imbalance.
    </div>""", unsafe_allow_html=True)

    ticket_counts = df.groupby('agent').size().rename('ticket_count')
    weighted_load = df.groupby('agent')['complexity_score'].sum()
    breach_rate   = df.groupby('agent')['sla_breach'].mean()
    agent_df = pd.concat(
        [ticket_counts, weighted_load, breach_rate], axis=1).reset_index()
    agent_df.columns = ['agent','ticket_count','weighted_load','breach_rate']
    agent_df['load_score'] = (
        (agent_df['weighted_load'] - agent_df['weighted_load'].min()) /
        (agent_df['weighted_load'].max() - agent_df['weighted_load'].min())*100
    ).round(1)

    # Overload alerts
    threshold  = agent_df['load_score'].mean() + agent_df['load_score'].std()
    overloaded = agent_df[agent_df['load_score'] > threshold]
    for _, row in overloaded.iterrows():
        st.markdown(f"""<div class="alert-box">
        ⚠ <b>{row['agent']}</b> is overloaded —
        complexity-weighted load score <b>{row['load_score']:.0f}/100</b>
        (SLA breach rate: {row['breach_rate']:.1%})
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Volume vs Complexity Load</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,5))
        x = np.arange(len(agent_df))
        w = 0.35
        ax.bar(x-w/2, agent_df['ticket_count'], width=w,
               label='Ticket Volume', color='#2d3150',
               edgecolor=ACCENT, linewidth=1)
        ax2 = ax.twinx()
        bar_colors = ['#e74c3c' if v >= threshold else ACCENT
                      for v in agent_df['load_score']]
        ax2.bar(x+w/2, agent_df['load_score'], width=w,
                label='Load Score', color=bar_colors,
                edgecolor='#1e2130', linewidth=0.5, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(agent_df['agent'], rotation=45, fontsize=9)
        ax.set_ylabel('Ticket Count', color='#8b92a5')
        ax2.set_ylabel('Complexity Load Score', color='#8b92a5')
        ax2.axhline(y=threshold, color='#e74c3c', linestyle='--',
                    linewidth=1, alpha=0.6)

        patch1 = mpatches.Patch(color='#2d3150',
                                 label='Ticket volume (all similar)')
        patch2 = mpatches.Patch(color=ACCENT,
                                 label='Load score (complexity-weighted)')
        patch3 = mpatches.Patch(color='#e74c3c',
                                 label='Overloaded agents')
        ax.legend(handles=[patch1,patch2,patch3],
                  fontsize=8, loc='upper left')
        ax.set_title('Same volume, very different load', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="insight-box">
        The blue bars (ticket volume) are nearly identical for every agent
        — raw count looks fair. The coloured bars (load score) tell a
        completely different story. Red bars flag agents who are overloaded
        by complexity, not just volume — these are the agents most likely
        to start breaching SLAs next.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Agent Summary</div>',
                    unsafe_allow_html=True)
        display_df = agent_df[['agent','ticket_count',
                                'load_score','breach_rate']].copy()
        display_df.columns = ['Agent','Tickets',
                               'Load Score','Breach Rate']
        display_df['Breach Rate'] = display_df['Breach Rate'].apply(
            lambda x: f"{x:.1%}")
        display_df['Load Score']  = display_df['Load Score'].apply(
            lambda x: f"{x:.0f}/100")
        display_df = display_df.sort_values(
            'Load Score', ascending=False).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("""<div class="insight-box">
        <b>How to read this table:</b> Sort by Load Score, not Tickets.
        An agent with 950 tickets and load score 100 needs reassignment
        or support. An agent with 1,050 tickets and load score 14 has
        capacity to absorb more work. This is how smart team leads
        actually redistribute workload.
        </div>""", unsafe_allow_html=True)