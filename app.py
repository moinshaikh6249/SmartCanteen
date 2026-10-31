import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

st.set_page_config(page_title="SmartCanteen", page_icon="🍔", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    return joblib.load("demand_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("canteen_data.csv")

model = load_model()
df = load_data()
FOOD_ITEMS = sorted(df["Food_Item"].dropna().unique().tolist())
WEATHER_OPTIONS = sorted(df["Weather"].dropna().unique().tolist())

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 10% 0%,rgba(56,189,248,.08),transparent 28%),radial-gradient(circle at 90% 10%,rgba(139,92,246,.08),transparent 25%),#0b0f17}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#111827 0%,#0b0f17 100%);border-right:1px solid rgba(255,255,255,.08)}
.hero{padding:28px 30px;border:1px solid rgba(255,255,255,.09);border-radius:22px;background:linear-gradient(135deg,rgba(30,41,59,.92),rgba(15,23,42,.78));box-shadow:0 18px 55px rgba(0,0,0,.28);margin-bottom:24px}
.hero-title{font-size:42px;font-weight:800;letter-spacing:-1px;margin:0}.hero-subtitle{color:#aeb8c7;font-size:16px;margin-top:7px}.eyebrow{color:#60a5fa;font-size:12px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px}
.section-title{font-size:27px;font-weight:750;margin-top:8px;margin-bottom:5px}.section-subtitle{color:#9ca8b8;margin-bottom:18px}
.glass-card{padding:20px;border-radius:18px;border:1px solid rgba(255,255,255,.08);background:rgba(17,24,39,.72);box-shadow:0 12px 35px rgba(0,0,0,.18);min-height:120px}
.card-label{color:#9ca8b8;font-size:13px;font-weight:650;margin-bottom:7px}.card-value{font-size:28px;font-weight:800;line-height:1.1}.card-note{color:#7f8da1;font-size:12px;margin-top:7px}
.action-card{padding:18px 20px;border-radius:16px;background:linear-gradient(135deg,rgba(37,99,235,.16),rgba(14,165,233,.06));border:1px solid rgba(96,165,250,.18);margin:8px 0 18px}
.action-title{font-size:17px;font-weight:750;margin-bottom:4px}.action-text{color:#b9c3d0;font-size:14px}.status-good{color:#86efac;font-weight:700}.status-warn{color:#fcd34d;font-weight:700}.status-high{color:#fca5a5;font-weight:700}
div[data-testid="stMetric"]{background:rgba(17,24,39,.68);border:1px solid rgba(255,255,255,.07);padding:15px 16px;border-radius:16px}
div[data-testid="stButton"]>button{border-radius:12px;min-height:45px;font-weight:750}footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

def money(x): return f"₹{x:,.0f}"

def status(d):
    if d >= 200: return "High Demand", "status-high"
    if d >= 100: return "Moderate Demand", "status-warn"
    return "Lower Demand", "status-good"

def price_for(item):
    return float(df.loc[df["Food_Item"] == item, "Price"].mean())

def predict(day_name, weather, temp, rain, exam, holiday, students, item, price):
    x = pd.DataFrame({"Day":[day_name],"Weather":[weather],"Temperature":[temp],"Rainfall_mm":[rain],"Exam_Day":[exam],"Holiday":[holiday],"Students":[students],"Food_Item":[item],"Price":[price]})
    return max(0, int(round(model.predict(x)[0])))

def historical_waste_rate(item):
    item_df = df[df["Food_Item"] == item]
    prepared = float(item_df["Quantity_Prepared"].sum())
    waste = float(item_df["Waste_Quantity"].sum())
    return (waste / prepared * 100) if prepared else 0.0

def daily_plan(d, students, weather, temp, rain, exam, holiday, stock_map):
    day = d.strftime("%A")
    rows=[]
    for item in FOOD_ITEMS:
        price=price_for(item)
        demand=predict(day,weather,temp,rain,exam,holiday,students,item,price)
        safety=int(np.ceil(demand*0.05))
        target=demand+safety
        stock=int(stock_map.get(item,0))
        prep=max(0,target-stock)
        available=stock+prep
        excess=max(0,stock-target)
        waste_rate=historical_waste_rate(item)
        waste_risk="HIGH" if waste_rate>=12 else ("MEDIUM" if waste_rate>=7 else "LOW")
        if stock > target*1.20:
            status="OVERSTOCK"; action="DO NOT PREPARE — Use existing stock first"
        elif stock < demand and waste_risk=="HIGH":
            status="SHORTAGE"; action="PREPARE CAREFULLY — High waste risk"
        elif stock < demand:
            status="SHORTAGE"; action="PREPARE — Stock below forecast"
        elif stock < target:
            status="BUFFER LOW"; action="SMALL PREPARATION — Buffer is low"
        else:
            status="READY"; action="READY — No extra preparation"
        demand_level="HIGH" if demand>=200 else ("MEDIUM" if demand>=100 else "LOW")
        rows.append({"Food Item":item,"Predicted Demand":demand,"Current Stock":stock,"Prepare Now":prep,"Available After Prep":available,"Safety Stock":safety,"Demand Level":demand_level,"Waste Risk":waste_risk,"Historical Waste %":waste_rate,"Excess Units":excess,"Price":price,"Expected Revenue":demand*price,"Status":status,"Action":action})
    order={"SHORTAGE":0,"BUFFER LOW":1,"OVERSTOCK":2,"READY":3}
    result=pd.DataFrame(rows)
    result["_order"]=result["Status"].map(order)
    return result.sort_values(["_order","Predicted Demand"],ascending=[True,False]).drop(columns="_order").reset_index(drop=True)

def hero():
    st.markdown('<div class="hero"><div class="eyebrow">DAILY OPERATIONS • MACHINE LEARNING</div><div class="hero-title">🍔 SmartCanteen</div><div class="hero-subtitle">Predict demand, plan preparation, reduce overproduction and make better day-to-day canteen decisions using data.</div></div>',unsafe_allow_html=True)

def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>',unsafe_allow_html=True)
    if subtitle: st.markdown(f'<div class="section-subtitle">{subtitle}</div>',unsafe_allow_html=True)

st.sidebar.markdown('<div style="padding:8px 0 18px"><div style="font-size:27px;font-weight:800">🍔 SmartCanteen</div><div style="color:#8fa0b5;font-size:12px;margin-top:5px">Daily Canteen Decision System</div></div>',unsafe_allow_html=True)
page=st.sidebar.radio("Navigation",["🏠 Daily Planner","🔮 Demand Prediction","📊 Analytics","🤖 Model Information"])
st.sidebar.divider(); st.sidebar.markdown("### ⚡ What SmartCanteen does")
for t in ["Predicts food demand","Suggests preparation quantity","Estimates expected revenue","Highlights high-demand items","Helps reduce overproduction","Supports daily planning"]: st.sidebar.caption("• "+t)

if page=="🏠 Daily Planner":
    hero(); section("🏠 Daily Canteen Planner","Plan today's preparation using ML forecast + current stock + waste risk.")
    c1,c2,c3,c4=st.columns(4)
    with c1: d=st.date_input("📅 Planning Date",value=date.today())
    with c2: students=st.number_input("👨‍🎓 Expected Students",50,2000,800,10)
    with c3: weather=st.selectbox("🌦️ Weather",WEATHER_OPTIONS)
    with c4: temp=st.number_input("🌡️ Temperature (°C)",10.0,50.0,28.0,.5)
    c5,c6,c7=st.columns(3)
    with c5: rain=st.number_input("🌧️ Rainfall (mm)",0.0,300.0,0.0,1.0)
    with c6: exam=st.selectbox("📚 Exam Day",["No","Yes"])
    with c7: holiday=st.selectbox("🏖️ Holiday",["No","Yes"])
    st.divider()
    section("📦 Step 2 — Current Stock","Enter units already available. SmartCanteen will recommend only the additional quantity needed.")
    st.info("💡 Example: If 100 Tea are already available, enter 100. The system will not ask you to prepare the full forecast again.")
    stock_map={}
    cols=st.columns(3)
    for i,item in enumerate(FOOD_ITEMS):
        with cols[i%3]:
            stock_map[item]=st.number_input(f"📦 {item}",0,5000,0,10,key=f"stock_{item}")
    if st.button("🚀 GENERATE TODAY'S SMART CANTEEN PLAN",use_container_width=True,type="primary"):
        with st.spinner("Running ML forecast, stock check and waste-risk analysis..."):
            st.session_state["plan"]=daily_plan(d,students,weather,temp,rain,1 if exam=="Yes" else 0,1 if holiday=="Yes" else 0,stock_map)
    if "plan" in st.session_state:
        plan=st.session_state["plan"]; st.divider(); section("📊 Today's Decision Dashboard","The important question: what should the canteen actually do today?")
        total_d=int(plan["Predicted Demand"].sum()); total_stock=int(plan["Current Stock"].sum()); total_p=int(plan["Prepare Now"].sum()); total_s=int(plan["Safety Stock"].sum()); total_r=float(plan["Expected Revenue"].sum())
        shortage=int((plan["Status"]=="SHORTAGE").sum()); ready=int((plan["Status"]=="READY").sum()); overstock=int((plan["Status"]=="OVERSTOCK").sum()); highwaste=int((plan["Waste Risk"]=="HIGH").sum())
        a,b,c,e,f=st.columns(5); a.metric("🎯 Predicted Demand",f"{total_d:,}"); b.metric("📦 Current Stock",f"{total_stock:,}"); c.metric("🔥 Prepare Now",f"{total_p:,}"); e.metric("🛡️ Safety Stock",f"{total_s:,}"); f.metric("💰 Expected Revenue",money(total_r))
        if shortage:
            row=plan[plan["Status"]=="SHORTAGE"].iloc[0]
            msg=f"<span class='danger'>🔥 {shortage} item(s) need preparation.</span><br><b>{row['Food Item']}</b> is the first priority: forecast <b>{int(row['Predicted Demand'])}</b>, current stock <b>{int(row['Current Stock'])}</b>, so prepare <b>{int(row['Prepare Now'])}</b> more."
        elif overstock:
            row=plan[plan["Status"]=="OVERSTOCK"].iloc[0]
            msg=f"<span class='warn'>⚠️ Overstock detected.</span><br><b>{row['Food Item']}</b> already has <b>{int(row['Current Stock'])}</b> units against forecast <b>{int(row['Predicted Demand'])}</b>. <span class='good'>Do not prepare more.</span>"
        else:
            msg="<span class='good'>✅ Stock position is healthy.</span><br>Current inventory covers forecast demand. Avoid unnecessary preparation and use the existing stock first."
        st.markdown(f"<div class='action-card'><div class='action-title'>🧠 SmartCanteen's Main Decision</div><div class='action-text'>{msg}<br><br>The decision combines ML demand forecasting, current stock, safety buffer and historical waste behaviour.</div></div>",unsafe_allow_html=True)
        h1,h2,h3,h4=st.columns(4); h1.metric("🔥 Need Preparation",shortage); h2.metric("✅ Already Ready",ready); h3.metric("⚠️ Overstock",overstock); h4.metric("🗑️ High Waste Risk",highwaste)
        st.divider(); st.subheader("📋 Item-Level Action Plan"); st.caption("Every row gives a practical action for today's operations.")
        view=plan.copy(); view["Historical Waste %"]=view["Historical Waste %"].map(lambda x:f"{x:.2f}%")
        view["Price"]=view["Price"].map(money); view["Expected Revenue"]=view["Expected Revenue"].map(money)
        view=view[["Food Item","Predicted Demand","Current Stock","Prepare Now","Available After Prep","Safety Stock","Demand Level","Waste Risk","Historical Waste %","Excess Units","Action"]]
        st.dataframe(view,use_container_width=True,hide_index=True)
        p1,p2=st.columns(2)
        with p1:
            st.subheader("🔥 What to Prepare")
            need=plan[plan["Prepare Now"]>0]
            if len(need)==0: st.success("No additional preparation is required.")
            else:
                for _,r in need.iterrows(): st.write(f"**{r['Food Item']}** → prepare **{int(r['Prepare Now'])}** units")
        with p2:
            st.subheader("🗑️ Waste Prevention Alerts")
            alerts=plan[(plan["Excess Units"]>0)|(plan["Waste Risk"]=="HIGH")]
            if len(alerts)==0: st.success("No major waste warning for today's plan.")
            else:
                for _,r in alerts.iterrows():
                    if r["Excess Units"]>0: st.warning(f"{r['Food Item']}: {int(r['Excess Units'])} excess units above target.")
                    else: st.warning(f"{r['Food Item']}: historical waste risk is HIGH ({r['Historical Waste %']:.2f}%). Prepare carefully.")
        st.divider(); st.subheader("📈 Forecast vs Current Stock")
        st.bar_chart(plan.set_index("Food Item")[["Predicted Demand","Current Stock","Prepare Now"]])

elif page=="🔮 Demand Prediction":
    hero(); section("🔮 Food Demand Prediction","Predict one food item using the same conditions a real canteen manager would consider.")
    c1,c2,c3=st.columns(3)
    with c1: selected_date=st.date_input("📅 Prediction Date",value=date.today()); day=selected_date.strftime("%A"); st.caption(f"Day: {day}")
    with c2: students=st.number_input("👨‍🎓 Expected Students",50,2000,800,10)
    with c3: item=st.selectbox("🍔 Food Item",FOOD_ITEMS)
    c4,c5,c6=st.columns(3)
    with c4: weather=st.selectbox("🌦️ Weather",WEATHER_OPTIONS)
    with c5: temp=st.number_input("🌡️ Temperature (°C)",10.0,50.0,28.0,.5)
    with c6: rain=st.number_input("🌧️ Rainfall (mm)",0.0,300.0,0.0,1.0)
    c7,c8,c9=st.columns(3)
    with c7: exam=st.selectbox("📚 Exam Day",["No","Yes"])
    with c8: holiday=st.selectbox("🏖️ Holiday",["No","Yes"])
    with c9: price=st.number_input("💰 Price (₹)",1.0,500.0,round(price_for(item),2),1.0)
    if st.button("🔮 PREDICT FOOD DEMAND",use_container_width=True,type="primary"):
        with st.spinner("Analyzing conditions..."):
            demand=predict(day,weather,temp,rain,1 if exam=="Yes" else 0,1 if holiday=="Yes" else 0,students,item,price)
        prep=int(np.ceil(demand*1.05)); revenue=demand*price; safety=prep-demand
        st.session_state["result"]={"date":selected_date,"day":day,"item":item,"students":students,"weather":weather,"temp":temp,"rain":rain,"exam":exam,"holiday":holiday,"price":price,"demand":demand,"prep":prep,"revenue":revenue,"safety":safety}
    if "result" in st.session_state:
        r=st.session_state["result"]; st.divider(); st.success("✅ Prediction completed successfully"); section("📊 Prediction Result")
        a,b,c,e=st.columns(4); a.metric("🍔 Predicted Demand",f'{r["demand"]:,} items'); b.metric("📦 Recommended Preparation",f'{r["prep"]:,} items'); c.metric("💰 Expected Revenue",money(r["revenue"])); e.metric("🛡️ Safety Stock",f'{r["safety"]:,} items')
        s,css=status(r["demand"])
        st.markdown(f'<div class="action-card"><div class="action-title">💡 SmartCanteen Recommendation</div><div class="action-text"><span class="{css}">{s}</span> expected for <b>{r["item"]}</b>. Prepare approximately <b>{r["prep"]}</b> items and keep <b>{r["safety"]}</b> as a small buffer.</div></div>',unsafe_allow_html=True)
        st.subheader("📝 Prediction Summary"); x,y=st.columns(2)
        with x:
            st.write(f'**Date:** {r["date"]}'); st.write(f'**Day:** {r["day"]}'); st.write(f'**Food Item:** {r["item"]}'); st.write(f'**Expected Students:** {r["students"]:,}'); st.write(f'**Price:** {money(r["price"])}')
        with y:
            st.write(f'**Weather:** {r["weather"]}'); st.write(f'**Temperature:** {r["temp"]} °C'); st.write(f'**Rainfall:** {r["rain"]} mm'); st.write(f'**Exam Day:** {r["exam"]}'); st.write(f'**Holiday:** {r["holiday"]}')

elif page=="📊 Analytics":
    hero(); section("📊 Canteen Analytics","Understand historical demand, revenue, waste and operational patterns.")
    f1,f2,f3=st.columns(3)
    with f1: sf=st.selectbox("🍔 Food Item",["All"]+FOOD_ITEMS)
    with f2: sw=st.selectbox("🌦️ Weather",["All"]+WEATHER_OPTIONS)
    with f3: se=st.selectbox("📚 Day Type",["All","Normal Day","Exam Day"])
    x=df.copy()
    if sf!="All": x=x[x["Food_Item"]==sf]
    if sw!="All": x=x[x["Weather"]==sw]
    if se!="All": x=x[x["Exam_Day"]==(1 if se=="Exam Day" else 0)]
    sales=int(x["Quantity_Sold"].sum()); revenue=float(x["Revenue"].sum()); waste=int(x["Waste_Quantity"].sum()); prepared=int(x["Quantity_Prepared"].sum()); wp=waste/prepared*100 if prepared else 0
    st.divider(); a,b,c,e=st.columns(4); a.metric("🍔 Total Items Sold",f"{sales:,}"); b.metric("💰 Total Revenue",money(revenue)); c.metric("🗑️ Total Waste",f"{waste:,}"); e.metric("📉 Waste Percentage",f"{wp:.2f}%")
    if len(x):
        best=x.groupby("Food_Item")["Quantity_Sold"].sum().sort_values(ascending=False).index[0]; bad=x.groupby("Food_Item")["Waste_Quantity"].sum().sort_values(ascending=False).index[0]
        st.markdown(f'<div class="action-card"><div class="action-title">🧠 Data-Driven Operational Insight</div><div class="action-text"><b>{best}</b> has the highest historical sales in the current filter. <b>{bad}</b> has the highest recorded waste and deserves closer preparation monitoring.</div></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a: st.subheader("🍔 Food Item Demand"); st.bar_chart(x.groupby("Food_Item")["Quantity_Sold"].sum().sort_values(ascending=False))
    with b: st.subheader("💰 Revenue by Food Item"); st.bar_chart(x.groupby("Food_Item")["Revenue"].sum().sort_values(ascending=False))
    a,b=st.columns(2)
    with a: st.subheader("🌦️ Average Demand by Weather"); st.bar_chart(x.groupby("Weather")["Quantity_Sold"].mean().round(2))
    with b:
        st.subheader("📚 Exam Day vs Normal Day"); q=x.groupby("Exam_Day")["Quantity_Sold"].mean(); q.index=["Normal Day" if v==0 else "Exam Day" for v in q.index]; st.bar_chart(q)
    st.subheader("🗑️ Waste by Food Item"); st.bar_chart(x.groupby("Food_Item")["Waste_Quantity"].sum().sort_values(ascending=False))
    st.divider(); st.subheader("📋 Filtered Dataset"); st.dataframe(x.head(50),use_container_width=True,hide_index=True)

else:
    hero(); section("🤖 Machine Learning Model","Technical transparency behind SmartCanteen's daily decisions.")
    st.subheader("🏆 Selected Model"); st.success("Random Forest Regressor is the selected model because it achieved the lowest Mean Absolute Error.")
    a,b,c=st.columns(3); a.metric("MAE","7.79"); b.metric("RMSE","9.84"); c.metric("R² Score","0.9771")
    st.caption("Test-set performance: MAE ≈ 7.79 units and R² = 0.9771.")
    st.divider(); st.subheader("📊 Model Performance Comparison")
    perf=pd.DataFrame({"Model":["Random Forest","Decision Tree","Linear Regression"],"MAE":[7.791613,9.893033,19.659095],"RMSE":[9.836202,12.438533,27.887483],"R² Score":[.977139,.963442,.816235]})
    perf["MAE"]=perf["MAE"].round(2); perf["RMSE"]=perf["RMSE"].round(2); perf["R² Score"]=perf["R² Score"].round(4); st.dataframe(perf,use_container_width=True,hide_index=True)
    st.divider(); st.subheader("📚 Training Information"); a,b,c=st.columns(3); a.metric("Dataset Records","2,190"); b.metric("Training Records","1,752"); c.metric("Testing Records","438")
    st.divider(); st.subheader("🎯 Prediction Target"); st.info("Quantity_Sold — the number of units expected to be sold for a selected food item.")
    st.subheader("📥 Model Features")
    features=pd.DataFrame({"Feature":["Day","Weather","Temperature","Rainfall","Exam Day","Holiday","Students","Food Item","Price"],"Type":["Categorical","Categorical","Numeric","Numeric","Binary","Binary","Numeric","Categorical","Numeric"]})
    st.dataframe(features,use_container_width=True,hide_index=True)
    st.divider(); st.subheader("⚙️ Machine Learning Algorithms"); a,b,c=st.columns(3)
    a.markdown("**Linear Regression**\n\nBaseline regression model for comparison."); b.markdown("**Decision Tree**\n\nCaptures non-linear relationships between features."); c.markdown("**Random Forest**\n\nCombines multiple trees to improve prediction performance and generalization.")
    st.divider(); st.subheader("📐 Evaluation Metrics"); st.dataframe(pd.DataFrame({"Metric":["MAE","RMSE","R² Score"],"Purpose":["Average prediction error","Prediction error with greater penalty for large errors","How well the model explains variation in demand"]}),use_container_width=True,hide_index=True)
    st.divider(); st.success("🤖 SmartCanteen uses the trained Random Forest model stored in demand_model.pkl for demand prediction.")

st.divider(); st.markdown('<div style="text-align:center;color:#64748b;padding:8px 0 20px"><b>SmartCanteen</b> • Data-Driven Daily Canteen Planning<br><span style="font-size:12px">Machine Learning • Demand Forecasting • Analytics • Operational Decision Support</span></div>',unsafe_allow_html=True)
