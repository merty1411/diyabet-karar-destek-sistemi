import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import dice_ml
from dice_ml import Dice

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Diyabet Karar Destek Sistemi", layout="wide")

# --- TASARIM (Dinamik Renkler ve Karanlık Mod Koruması) ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; }
    h1, h2, h3, h4, p, li, span, label, div { color: #1A202C !important; }
    [data-testid="stSidebar"] { background-color: #F7FAFC !important; }
    [data-testid="stSidebar"] * { color: #2D3748 !important; }

    /* Genel Kart Yapısı */
    .health-card {
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        margin-top: 25px;
        color: #1A202C !important;
    }
    /* Riskli Durum (Kırmızı) */
    .risk-high { border: 3px solid #E53E3E; background-color: #FFF5F5 !important; }
    /* Güvenli Durum (Yeşil) */
    .risk-low { border: 3px solid #38A169; background-color: #F0FFF4 !important; }
    
    .step-box {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 8px solid;
    }
    .step-red { border-left-color: #E53E3E; background-color: #FFFFFF !important; }
    .step-green { border-left-color: #38A169; background-color: #FFFFFF !important; }
    
    .diet-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .diet-table td, .diet-table th { border: 1px solid #CBD5E0; padding: 10px; color: #1A202C !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    file_name = "diabetes.csv"
    if not os.path.exists(file_name): return None, None, None, None
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Pedigree", "Age", "Outcome"]
    df = pd.read_csv(file_name, names=cols)
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, df[df[col] != 0][col].median())
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_sc, y)
    return df, model, scaler, X

df, model, scaler, X_features = load_and_train()

# --- SIDEBAR ---
st.sidebar.header("📋 Hasta Bilgileri")
with st.sidebar:
    boy = st.number_input('Boyunuz (cm)', 100, 220, 175)
    kilo = st.number_input('Kilonuz (kg)', 30, 200, 105)
    bmi_mevcut = kilo / ((boy/100)**2)
    st.info(f"Mevcut BMI: {bmi_mevcut:.1f}")
    st.divider()
    glucose = st.slider('Açlık Şekeri', 50, 200, 140)
    insulin = st.slider('İnsülin Değeri', 15, 800, 120)
    age = st.slider('Yaşınız', 18, 90, 50)
    preg = st.number_input('Hamilelik Sayısı', 0, 20, 1)
    pedigree = st.slider('Genetik Skor', 0.0, 2.5, 0.5)

user_df = pd.DataFrame({'Pregnancies':[preg], 'Glucose':[glucose], 'BloodPressure':[80], 'SkinThickness':[20], 'Insulin':[insulin], 'BMI':[bmi_mevcut], 'Pedigree':[pedigree], 'Age':[age]})

# --- ANA PANEL ---
st.title("🛡️ Akıllı Diyabet Yönetim Portalı")

if df is not None:
    input_sc = scaler.transform(user_df)
    prob = model.predict_proba(input_sc)[0][1]
    is_high_risk = prob > 0.5

    if is_high_risk:
        st.error(f"⚠️ DİKKAT: YÜKSEK RİSK GRUBUNDASINIZ (Risk Oranı: %{prob*100:.1f})")
    else:
        st.success(f"✅ TEBRİKLER: DÜŞÜK RİSK GRUBUNDASINIZ (Risk Oranı: %{prob*100:.1f})")
    
    st.divider()

    # --- DiCE ANALİZİ VE REHBER ---
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('scaler', scaler), ('model', model)])
    d = dice_ml.Data(dataframe=df, continuous_features=list(X_features.columns), outcome_name='Outcome')
    m = dice_ml.Model(model=pipe, backend="sklearn")
    exp = Dice(d, m, method="random")

    with st.spinner('Verileriniz analiz ediliyor...'):
        dice_exp = exp.generate_counterfactuals(user_df, total_CFs=3, desired_class=0, features_to_vary=["BMI", "Glucose", "Insulin"])
        cf_results = dice_exp.cf_examples_list[0].final_cfs_df
        target_bmi = cf_results['BMI'].mean()
        target_gl = int(cf_results['Glucose'].mean())

    h_kilo = round(target_bmi * ((boy/100)**2), 1)
    v_kilo = round(kilo - h_kilo, 1)

    # Renk Sınıfı Belirleme
    card_style = "risk-high" if is_high_risk else "risk-low"
    step_style = "step-red" if is_high_risk else "step-green"

    st.subheader("📊 Sağlık Hedefleriniz")
    curr = user_df[["Glucose", "BMI", "Insulin"]].copy(); curr.index = ["Şu Anki Durum"]
    targ = pd.DataFrame({'Glucose':[target_gl], 'BMI':[target_bmi], 'Insulin':[int(cf_results['Insulin'].mean())]}, index=["AI Hedefleri"])
    st.table(pd.concat([curr, targ]))

    # --- DİNAMİK REHBER KARTI ---
    st.markdown(f'<div class="health-card {card_style}"><h3>🩺 Sağlık Raporu ve Tavsiyeler</h3>', unsafe_allow_html=True)
    st.write(f"Yaşınız ({age}) ve genetik geçmişiniz ({pedigree}) baz alınarak hazırlanan plan:")
    
    # Kilo Adımı
    if v_kilo > 0:
        st.markdown(f'<div class="step-box {step_style}"><b>1. Kilo Kontrolü:</b> İdeal kilonuz <b>{h_kilo} kg</b>. Yaklaşık <b>{v_kilo} kg</b> vermeniz önerilir.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="step-box {step_style}"><b>1. Kilo Kontrolü:</b> Kilonuz şu an ideal seviyededir. Mevcut formunuzu koruyun.</div>', unsafe_allow_html=True)
    
    # Şeker Adımı
    st.markdown(f'<div class="step-box {step_style}"><b>2. Şeker Yönetimi:</b> Kan şekerinizi <b>{target_gl}</b> seviyesine çekmek riskinizi azaltacaktır.</div>', unsafe_allow_html=True)
    
    # Sabit Beslenme Tablosu
    st.markdown(f"""
    <div class="step-box {step_style}">
        <b>3. Yaşam Tarzı Planı:</b>
        <table class="diet-table">
            <tr><th>Kategori</th><th>Öneri</th></tr>
            <tr><td>Egzersiz</td><td>Haftada 150 dk tempolu yürüyüş.</td></tr>
            <tr><td>Beslenme</td><td>Protein ağırlıklı ve düşük karbonhidratlı diyet.</td></tr>
        </table>
    </div></div>
    """, unsafe_allow_html=True)

    if not is_high_risk:
        st.balloons()

    st.warning("⚠️ Bu sistem bir tahmin aracıdır. Kesin tanı için doktorunuza danışın.")

else:
    st.error("Veri dosyası bulunamadı.")