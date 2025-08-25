from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from datetime import date,datetime
from collections import Counter
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np



def how_many_days(df2):
  ca_pa_closed = []
  closed_days = []
  my_ca_pa = []
  for inx,row in enumerate(df2['level']):
    try:
      if row == 0 or row == 1:
        days = datetime.strptime(df2['root_cause_closed_date'].iloc[inx], "%m/%d/%Y")-df2['raise_date'].iloc[inx]
        ca_pa_closed.append(df2['ca_pa'].iloc[inx])
        closed_days.append(days.days)
      elif row == 2:
        days = datetime.strptime(df2['corrective_action_closed_date'].iloc[inx], "%m/%d/%Y")-df2['raise_date'].iloc[inx]
        ca_pa_closed.append(df2['ca_pa'].iloc[inx])
        closed_days.append(days.days)
      elif row == 3:
        days = datetime.strptime(df2['preventive_action_closed_date'].iloc[inx], "%m/%d/%Y")-df2['raise_date'].iloc[inx]
        ca_pa_closed.append(df2['ca_pa'].iloc[inx])
        closed_days.append(days.days)
      elif row == 4:
        days = datetime.strptime(df2['implementing_action_closed_date'].iloc[inx], "%m/%d/%Y")-df2['raise_date'].iloc[inx]
        ca_pa_closed.append(df2['ca_pa'].iloc[inx])
        closed_days.append(days.days)
      elif row == 5:
        days = datetime.strptime(df2['accesptance_action_closed_date'].iloc[inx], "%m/%d/%Y")-df2['raise_date'].iloc[inx]
        ca_pa_closed.append(df2['ca_pa'].iloc[inx])
        closed_days.append(days.days)
      elif row == 6:
        days = datetime.strptime(df2['follow_up_closed_date'].iloc[inx], "%m/%d/%Y")-df2['raise_date'].iloc[inx]
        ca_pa_closed.append(df2['ca_pa'].iloc[inx])
        closed_days.append(days.days)
        if (datetime.today() - datetime.strptime(df2['follow_up_closed_date'].iloc[inx], "%m/%d/%Y")).days<8:
           my_ca_pa.append(df2['ca_pa'].iloc[inx])

    except:
      None
  return ca_pa_closed,closed_days,my_ca_pa

def audt_more(df):
  df2 = df[df['status']=="Open"]
  list_of_audit = df2['audit'].unique()
  df_my = df[df['audit'].isin(list_of_audit)]
  results = []

  for audit_id, group in df_my.groupby('audit'):
      open_capa = group[group['status'] == 'Open']
      closed_capa = group[group['status'] == 'Closed']

      result = {
          'Audit ID': audit_id,
          'Audit ID number': audit_id.split("-")[0],
          'Audit ID descr': ",".join(audit_id.split("-")[1:]),
          'Open Count': len(open_capa),
          'Closed Count': len(closed_capa),
          'Open CAPAs': ",".join(list(open_capa['ca_pa'])),
          'Closed CAPAs': ",".join(list(closed_capa['ca_pa'])),
      }
      results.append(result)

  # Convert results to DataFrame
  summary_df = pd.DataFrame(results)

  return summary_df


#######################

def capa_individual_progress(df):
  df = df[df['status']=="Open"]
  df['root_cause_target_date'] = pd.to_datetime(df['root_cause_target_date'])
  df['target_date'] = pd.to_datetime(df['target_date'])
  df['corrective_action_target_date'] = pd.to_datetime(df['corrective_action_target_date'])
  df['preventive_action_target_date'] = pd.to_datetime(df['preventive_action_target_date'])
  df['accesptance_action_target_date'] = pd.to_datetime(df['accesptance_action_target_date'])
  df['follow_up_target_date'] = pd.to_datetime(df['follow_up_target_date'])

  overdue = []
  les_3 = []
  les_7 = []
  les_14 = []
  plus_14 = []

  for index,each_level in enumerate(df['level']):

    if each_level == 0:
      day = df['root_cause_target_date'].iloc[index] - datetime.today()
      if day.days < 0:
        overdue.append("Root Cause")
        overdue.append("Corrective Action Plan")
        overdue.append("Preventive Action")
        overdue.append("Acceptance of CAP")
        overdue.append("Implementing of CAP")
        overdue.append("Follow Up")
      elif day.days >= 0 and day.days <= 3:
        les_3.append("Root Cause")
        les_3.append("Corrective Action Plan")
        les_3.append("Preventive Action")
        les_3.append("Acceptance of CAP")
        les_3.append("Implementing of CAP")
        les_3.append("Follow Up")
      elif day.days > 3 and day.days <= 7:
        les_7.append("Root Cause")
        les_7.append("Corrective Action Plan")
        les_7.append("Preventive Action")
        les_7.append("Acceptance of CAP")
        les_7.append("Implementing of CAP")
        les_7.append("Follow Up")
      elif day.days > 7 and day.days <= 14:
        les_14.append("Root Cause")
        les_14.append("Corrective Action Plan")
        les_14.append("Preventive Action")
        les_14.append("Acceptance of CAP")
        les_14.append("Implementing of CAP")
        les_14.append("Follow Up")
      else:
        plus_14.append("Root Cause")
        plus_14.append("Corrective Action Plan")
        plus_14.append("Preventive Action")
        plus_14.append("Acceptance of CAP")
        plus_14.append("Implementing of CAP")
        plus_14.append("Follow Up")
    if each_level == 1:
      day = df['corrective_action_target_date'].iloc[index] - datetime.today()
      if day.days < 0:
        overdue.append("Corrective Action Plan")
        overdue.append("Preventive Action")
        overdue.append("Acceptance of CAP")
        overdue.append("Implementing of CAP")
        overdue.append("Follow Up")
      elif day.days >= 0 and day.days <= 3:
        les_3.append("Corrective Action Plan")
        les_3.append("Preventive Action")
        les_3.append("Acceptance of CAP")
        les_3.append("Implementing of CAP")
        les_3.append("Follow Up")
      elif day.days > 3 and day.days <= 7:
        les_7.append("Corrective Action Plan")
        les_7.append("Preventive Action")
        les_7.append("Acceptance of CAP")
        les_7.append("Implementing of CAP")
        les_7.append("Follow Up")
      elif day.days > 7 and day.days <= 14:
        les_14.append("Corrective Action Plan")
        les_14.append("Preventive Action")
        les_14.append("Acceptance of CAP")
        les_14.append("Implementing of CAP")
        les_14.append("Follow Up")
      else:
        plus_14.append("Corrective Action Plan")
        plus_14.append("Preventive Action")
        plus_14.append("Acceptance of CAP")
        plus_14.append("Implementing of CAP")
        plus_14.append("Follow Up")
    if each_level == 2:
      day = df['preventive_action_target_date'].iloc[index] - datetime.today()
      if day.days < 0:
        overdue.append("Preventive Action")
        overdue.append("Acceptance of CAP")
        overdue.append("Implementing of CAP")
        overdue.append("Follow Up")
      elif day.days >= 0 and day.days <= 3:
        les_3.append("Preventive Action")
        les_3.append("Acceptance of CAP")
        les_3.append("Implementing of CAP")
        les_3.append("Follow Up")
      elif day.days > 3 and day.days <= 7:
        les_7.append("Preventive Action")
        les_7.append("Acceptance of CAP")
        les_7.append("Implementing of CAP")
        les_7.append("Follow Up")
      elif day.days > 7 and day.days <= 14:
        les_14.append("Preventive Action")
        les_14.append("Acceptance of CAP")
        les_14.append("Implementing of CAP")
        les_14.append("Follow Up")
      else:
        plus_14.append("Preventive Action")
        plus_14.append("Acceptance of CAP")
        plus_14.append("Implementing of CAP")
        plus_14.append("Follow Up")
    if each_level == 3:
      day = df['accesptance_action_target_date'].iloc[index] - datetime.today()
      if day.days < 0:
        overdue.append("Acceptance of CAP")
        overdue.append("Implementing of CAP")
        overdue.append("Follow Up")
      elif day.days >= 0 and day.days <= 3:
        les_3.append("Acceptance of CAP")
        les_3.append("Implementing of CAP")
        les_3.append("Follow Up")
      elif day.days > 3 and day.days <= 7:
        les_7.append("Acceptance of CAP")
        les_7.append("Implementing of CAP")
        les_7.append("Follow Up")
      elif day.days > 7 and day.days <= 14:
        les_14.append("Acceptance of CAP")
        les_14.append("Implementing of CAP")
        les_14.append("Follow Up")
      else:
        plus_14.append("Acceptance of CAP")
        plus_14.append("Implementing of CAP")
        plus_14.append("Follow Up")
    if each_level == 4:
      try:
        day = df['implementing_action_target_date'].iloc[index] - datetime.today()
      except:
        day = df['target_date'].iloc[index] - datetime.today()
      if day.days < 0:
        overdue.append("Implementing of CAP")
        overdue.append("Follow Up")
      elif day.days >= 0 and day.days <= 3:
        les_3.append("Implementing of CAP")
        les_3.append("Follow Up")
      elif day.days > 3 and day.days <= 7:
        les_7.append("Implementing of CAP")
        les_7.append("Follow Up")
      elif day.days > 7 and day.days <= 14:
        les_14.append("Implementing of CAP")
        les_14.append("Follow Up")
      else:
        plus_14.append("Implementing of CAP")
        plus_14.append("Follow Up")
    if each_level == 5 or each_level == 6:
      try:
        day =  df['follow_up_target_date'].iloc[index] - datetime.today()
      except:
        day = df['target_date'].iloc[index] - datetime.today()
      if day.days < 0:
        overdue.append("Follow Up")
      elif day.days >= 0 and day.days <= 3:
        les_3.append("Follow Up")
      elif day.days > 3 and day.days <= 7:
        les_7.append("Follow Up")
      elif day.days > 7 and day.days <= 14:
        les_14.append("Follow Up")
      else:
        plus_14.append("Follow Up")
  return (overdue,les_3,les_7,les_14,plus_14)


def build_progress_dataframe(overdue, les_3, les_7, les_14, plus_14):
    stages = ["Root Cause", "Corrective Action Plan", "Preventive Action", "Acceptance of CAP", "Implementing of CAP", "Follow Up"]

    # Count occurrences of each stage in each time bucket
    overdue_count = Counter(overdue)
    les_3_count = Counter(les_3)
    les_7_count = Counter(les_7)
    les_14_count = Counter(les_14)
    plus_14_count = Counter(plus_14)

    data = []

    for stage in stages:
        over = overdue_count.get(stage, 0)
        l3 = les_3_count.get(stage, 0)
        b3_7 = les_7_count.get(stage, 0)
        b7_14 = les_14_count.get(stage, 0)
        m14 = plus_14_count.get(stage, 0)
        total = over + l3 + b3_7 + b7_14 + m14

        row = {
            "CA/PA Stages": stage,
            "Overdue": over,
            "Overdue \n≤ 3 days": l3,
            "Overdue \n4-7 days": b3_7,
            "Overdue \n8-14 days": b7_14,
            "Overdue \n> 14 days": m14,
            "Total": total
        }
        data.append(row)

    df_result = pd.DataFrame(data)
    return df_result

#######################
def required_informations(df):
  df = df[df['status']=="Open"]
  df['raise_date'] = pd.to_datetime(df['raise_date'])
  df['target_date'] = pd.to_datetime(df['target_date'])
  # son 7 gunluk CAPA
  last_7_days_rised_capa = []
  for day in df["raise_date"]:
    days = datetime.today() - day
    if days.days <= 7:
      last_7_days_rised_capa.append(day)
  # CAPA ahy side or SCAA side
  ahy_capa = []
  scaa_capa = []
  for index, each_level in enumerate(df["level"]):
    if each_level < 3:
      ahy_capa.append(each_level)
    elif each_level > 2 and df['status'].iloc[index]=="Open":
      scaa_capa.append(each_level)
  # CAPA status deadline 
  capa_ovedue = []
  capa_less_3_days = []
  capa_3_7_days = []
  capa_more_7_days = []
  for index, capa_due in enumerate(df["target_date"]):
    days = capa_due - datetime.today()
    if days.days <0:
      try:
        capa_ovedue.append(df["ca_pa"].iloc[index].split('-')[1])
      except:
        capa_ovedue.append(df["ca_pa"].iloc[index])
    elif days.days>= 0 and days.days <= 3:
      try:
        capa_less_3_days.append(df["ca_pa"].iloc[index].split('-')[1])
      except:
         capa_less_3_days.append(df["ca_pa"].iloc[index])
    elif days.days > 3 and days.days < 7:
      try:
        capa_3_7_days.append(df["ca_pa"].iloc[index].split('-')[1])
      except:
         capa_3_7_days.append(df["ca_pa"].iloc[index])
    else:
      try:
        number = df["ca_pa"].iloc[index].split('-')[1]
        capa_more_7_days.append(number)
      except:
         capa_more_7_days.append(df["ca_pa"].iloc[index])



  return len(last_7_days_rised_capa), len(ahy_capa), len(scaa_capa), len(capa_ovedue), len(capa_less_3_days), len(capa_3_7_days), len(capa_more_7_days),",".join(capa_ovedue),",".join(capa_less_3_days),",".join(capa_3_7_days),",".join(capa_more_7_days)











app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    df = pd.read_csv("ca_pa_data.csv")
    df['raise_date1'] = df['raise_date']
    df['raise_date'] = pd.to_datetime(df['raise_date'])
    df_status = df.copy()
    df_always_open = df[df['status']=="Open"]

    stage_mapping = {
        'root_cause_details': 1,
        'corrective_action_details': 2,
        'preventive_action_details': 3,
        'accesptance_action_details': 4,
        'implementing_action_details': 5,
        'follow_up_details': 6
    }


    # Prepare date options (optional: use unique dates or convert to strings)
    data_type_options = sorted(df['raise_date'].dropna().dt.date.unique())
    status_options = sorted(df['status'].dropna().unique())
    audit_options = sorted(df_always_open['audit'].dropna().unique())
    source_options = sorted(df_always_open['source'].dropna().unique())
    source_options = sorted(df_always_open['source'].dropna().unique())
    ca_pa_options = sorted(df_always_open['ca_pa'].dropna().unique())

    selected_data_types = []
    start_date = ""
    end_date = ""
    selected_audits = []
    selected_sources = []
    selected_ca_pa = []


    if request.method == "POST":
        selected_statuses = request.form.getlist('Status')
        selected_data_types = request.form.getlist('DataType')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        selected_audits = request.form.getlist('Audit')
        selected_sources = request.form.getlist('Source')
        selected_ca_pa = request.form.getlist('CA_PA')

        if selected_data_types:
            selected_dates = pd.to_datetime(selected_data_types)
            df = df[df['raise_date'].dt.date.isin(selected_dates)]
            df_status = df_status[df_status['raise_date'].dt.date.isin(selected_dates)]
        if start_date:
            df = df[df['raise_date'] >= pd.to_datetime(start_date)]
            df_status = df_status[df_status['raise_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['raise_date'] <= pd.to_datetime(end_date)]
            df_status = df_status[df_status['raise_date'] <= pd.to_datetime(end_date)]
        if selected_statuses:
            df = df[df['status'].isin(selected_statuses)]
        if selected_data_types:
            df = df[df['raise_date'].isin(selected_data_types)]
            df_status = df_status[df_status['raise_date'].isin(selected_data_types)]
        if selected_audits:
            df = df[df['audit'].isin(selected_audits)]
            df_status = df_status[df_status['audit'].isin(selected_audits)]
        if selected_sources:
            df = df[df['source'].isin(selected_sources)]
            df_status = df_status[df_status['source'].isin(selected_sources)]
        if selected_ca_pa:
            df = df[df['ca_pa'].isin(selected_ca_pa)]
            df_status = df_status[df_status['ca_pa'].isin(selected_ca_pa)]

    total_full = int(df_status.shape[0])
    total_open = int(df_status[df_status['status']=='Open'].shape[0])
    total_closed = int(df_status[df_status['status']=='Closed'].shape[0])
    try:
        total_pending = int(df_status[df_status['status']=='Pending'].shape[0])
    except:
       total_pending=int(0)


# Create an empty list to store levels
    levels = []

    # Reset bar data
    bars = {col: [] for col in stage_mapping}
    skipped_combined = []

    # Row-by-row processing
    for _, row in df_status.iterrows():
        max_level = 0
        filled = []

        # Detect filled stages and max level
        for col in stage_mapping:
            if pd.notna(row[col]) and str(row[col]).strip() != "":
                filled.append(col)
                max_level = max(max_level, stage_mapping[col])

        levels.append(max_level)  # Store the level

        # Fill bars
        for col in stage_mapping:
            bars[col].append(1 if col in filled else 0)

        # Detect skipped stages (including follow_up_details)
        skipped = 0
        for col in stage_mapping:
            if col not in filled and stage_mapping[col] < max_level:
                skipped = 1
                break
        skipped_combined.append(skipped)

    # Add the "level" column to your DataFrame
    df_status["level"] = levels

    df_open = df.copy()
    df_open = df_open.sort_values(by="ca_pa")
    x_labels = df_open['ca_pa']

    stage_mapping = {
        'root_cause_details': 1,
        'corrective_action_details': 2,
        'preventive_action_details': 3,
        'accesptance_action_details': 4,
        'implementing_action_details': 5,
        'follow_up_details': 6
    }


    # Bar data preparation
    bars = {col: [] for col in stage_mapping}
    skipped_combined = []

    for _, row in df_open.iterrows():
        filled_stages = []
        max_level = 0

        for col in stage_mapping:
            if pd.notna(row[col]) and str(row[col]).strip() != "":
                filled_stages.append(col)
                if stage_mapping[col] > max_level:
                    max_level = stage_mapping[col]

        # Bar values
        for col in stage_mapping:
            bars[col].append(1 if col in filled_stages else 0)

        # Skipped check
        skipped = 0
        for col in list(stage_mapping.keys())[:-1]:
            if stage_mapping[col] < max_level and col not in filled_stages:
                skipped = 1
                break
        skipped_combined.append(skipped)
######################################################  Results ###################################################3






    total = int(df_status[df_status['status']=='Open'].shape[0])
    ca_pa_closed,closed_days,ca_pa_list = how_many_days(df_status)
    new, ahy_capa, scaa_capa, overdue, due_3, due_3_7, due_7_more,capa_ovedue,capa_less_3_days,capa_3_7_days,capa_more_7_days= required_informations(df_status)
    overdue_n,les_3,les_7,les_14,plus_14 = capa_individual_progress(df_status)
    df_summary = build_progress_dataframe(overdue_n,les_3,les_7,les_14,plus_14)
    data = df_summary.values.tolist()
    columns = df_summary.columns.tolist()



######################################################################################################################
    # Plotly figure
    fig = go.Figure()
    color_map = {
        'root_cause_details': "#FF0000",
        'corrective_action_details': '#FF6600',
        'preventive_action_details': '#FFC000',
        'accesptance_action_details': '#92D050',
        'implementing_action_details': '#00B050',
        'follow_up_details': '#548235'
    }

    for col in stage_mapping:
        fig.add_trace(go.Bar(
            name=col.replace('_details', '').replace('_', ' ').title(),
            x=x_labels,
            y=bars[col],
            marker_color=color_map[col],
            text=["" for _ in bars[col]],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color="white")
        ))


    fig.update_layout(
        title="Findings Status in SCAA Q-Pulse",
        xaxis_title="CA/PA ID",
        yaxis_title="Details Count",
        barmode='stack',
        template="seaborn",
        legend_title="Stage Name",
        title_font=dict(size=24, family='Arial Black', color='darkblue'),
        xaxis=dict(tickangle=-90),
        yaxis=dict(range=[0, 6]),
        margin=dict(l=50, r=50, t=80, b=150),
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(240,240,240,0.95)',
        font=dict(size=14),
        height=600
    )

    chart_html = pio.to_html(fig, full_html=False)


    my_df = audt_more(df_status)
    df_tt = my_df.drop(columns=['Audit ID','Open Count','Closed Count'])



    columns1 = df_tt.columns.tolist()
    data1 = df_tt.values.tolist()

    df_melted = my_df.melt(id_vars="Audit ID number", 
                    value_vars=["Open Count", "Closed Count"],
                    var_name="Status", value_name="Count")
    
    fig_audit = px.bar(df_melted, 
             x="Audit ID number", 
             y="Count", 
             color="Status", 
             barmode="stack",
             title="Audit open closed status",
             color_discrete_map={"Open Count": "red", "Closed Count": "green"})
    
    chart__audit = pio.to_html(fig_audit, full_html=False)


    




    totals = {
        "Status": ["Closed","Open"],
        "Count": [df[df['status']=="Closed"].shape[0], df[df['status']=="Open"].shape[0]]
    }
    df_totals = pd.DataFrame(totals)

    pie = go.Pie(
        labels=df_totals["Status"],
        values=df_totals["Count"],
        marker=dict(colors=["red", "green"]),
        textinfo="label+percent+value"
    )

    scatter = go.Scatter(
        x=ca_pa_closed,
        y=closed_days,
        mode="markers+text",
        textposition="top center",
        marker=dict(size=4, color="red"),
        name="Closed Time"
    )




    avg_value = np.mean(closed_days)



    avg_line = go.Scatter(
        x=ca_pa_closed,
        y=[avg_value] * len(ca_pa_closed),
        mode="lines",
        line=dict(color="blue", dash="dash"),
        name=f"Avg = {avg_value:.2f}"
    )



    fig_last_full = make_subplots(rows=1, cols=2, subplot_titles=(" ", "Audit Counts with Avg Line"),
                        specs=[[{"type": "domain"}, {"type": "xy"}]],column_widths=[0.25, 0.75])



    fig_last_full.add_trace(pie, row=1, col=1)

    fig_last_full.add_trace(scatter, row=1, col=2)
    fig_last_full.add_trace(avg_line, row=1, col=2)


    fig_last_full.update_layout(
        xaxis2=dict(
            rangeslider=dict(visible=True),
            tickmode="linear",
            dtick=1
        )
    )


    chart_the_last_full = pio.to_html(fig_last_full, full_html=False)


    return render_template("index.html",
                           plot=chart_html,
                           data_types=data_type_options,
                           selected_data_types=selected_data_types,
                           start_date=start_date,
                           end_date=end_date,
                           status_options=status_options,
                           total=total,
                           new=new,
                           ahy_capa=ahy_capa,
                           scaa_capa=scaa_capa,
                           overdue=overdue,
                           due_3=due_3,
                           due_3_7=due_3_7,
                           due_7_more=due_7_more,
                           data=data, columns=columns,data1=data1,columns1=columns1,
                           total_full=total_full,
                           total_open=total_open,
                           total_closed=total_closed,
                           total_pending=total_pending,
                           capa_ovedue= capa_ovedue,
                           capa_less_3_days=capa_less_3_days,
                           capa_3_7_days=capa_3_7_days,
                           capa_more_7_days=capa_more_7_days,                        
                           source_options=source_options,
                           selected_audits=selected_audits,
                           selected_sources=selected_sources,
                           audit_options=audit_options,
                           selected_ca_pa=selected_ca_pa,
                           ca_pa_options=ca_pa_options,
                           plot_audit_count = chart__audit,
                           chart_the_last_full=chart_the_last_full,
                           ca_pa_list_name = ",".join(ca_pa_list),
                           count_capa = len(ca_pa_list)
) 

if __name__ == '__main__':
    app.run(debug=True)

