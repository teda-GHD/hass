import time
import paho.mqtt.client as mqtt
import json
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# === INPUTS ===

# User timezone
timezone = "SET TIMEZONE"

# Local HA instance details
ha_url = "INSERT HA URL"
auth_token = "INSERT AUTH TOKEN"

# Sensor entity IDs
sensor_total_consumed_energy = "sensor.total_consumed_energy"
sensor_solcast_forecast = "sensor.solcast_pv_forecast_forecast_" #exclude today/tomorrow
sensor_battery_charge = "sensor.battery_charge_nominal"
sensor_battery_capacity = "sensor.battery_capacity"

# MQQT Broker details
broker_id="BROKER ID FOR MQTT"
broker_address = "MQTT URL"
broker_port = "MQQT PORT"
username = "MQTT USER
password = "MQTT PASSWORD"

# === Define time range for data retrieval ===
now = datetime.now(ZoneInfo(timezone))
start_time = (now - timedelta(days=7)).replace(
    hour=0, minute=0, second=0, microsecond=0
)

# === Prepare API request ===
start_string = start_time.isoformat()
end_string = now.isoformat()

headers = {
    "Authorization": f"Bearer {auth_token}",
    "Content-Type": "application/json",
}

url = f"{ha_url}api/history/period/{start_string}"

energy_params = {
    "end_time": end_string,
    "filter_entity_id": sensor_total_consumed_energy,
    "minimal_response": "true",
    "no_attributes": "true",
    "significant_changes_only": "false",
}

# === Fetch historical state data ===
response = requests.get(url, headers=headers, params=energy_params)

# === Load data into DataFrame ===
energy_df = pd.json_normalize(response.json()[0]).drop(
    columns=["entity_id", "last_updated"]
)

# === Calculate consumed energy ===
energy_df["last_changed"] = pd.to_datetime(
    energy_df["last_changed"], format="ISO8601", utc=False
)
energy_df["state"] = pd.to_numeric(energy_df["state"], errors="coerce")
energy_df.dropna(subset=["state"], inplace=True)
energy_df = energy_df.set_index("last_changed")

regular_index = pd.date_range(start=start_string, end=end_string, freq="30min")

energy_df = (
    energy_df.reindex(energy_df.index.union(regular_index))
    .sort_index()
    .interpolate(method="linear", limit_direction="both")
    .loc[regular_index]
)

energy_df.index = pd.to_datetime(energy_df.index).tz_convert(timezone)

daily_baseline = energy_df.groupby(energy_df.index.date)["state"].transform("first")

energy_df["daily"] = energy_df["state"] - daily_baseline
energy_df["time"] = energy_df.index.time
energy_df["date"] = energy_df.index.date

energy_df = energy_df.pivot(index="time", columns="date", values="daily")
energy_df = energy_df.diff()
energy_df[energy_df < 0] = 0
energy_df = energy_df.interpolate(limit_direction="both")
energy_df["time"] = energy_df.index

energy_df.to_csv("energy.csv", index=True)

# === Define time range for data retrieval ===

url_today = f"{ha_url}api/states/{sensor_solcast_forecast}today"
url_tomor = f"{ha_url}api/states/{sensor_solcast_forecast}tomorrow"
url_3d = f"{ha_url}api/states/{sensor_solcast_forecast}day_3" # NEED TO HAVE ACTIVE IN SOLCAST INTEGRATION
url_4d = f"{ha_url}api/states/{sensor_solcast_forecast}day_4" # AS ABOVE

solar_params = {
    "minimal_response": "false",
    "no_attributes": "false",
    "significant_changes_only": "true",
}
# === Fetch historical state data ===
solar_response_today = requests.get(url_today, headers=headers)
solar_response_tomor = requests.get(url_tomor, headers=headers)
solar_response_day_3 = requests.get(url_3d, headers=headers)
solar_response_day_4 = requests.get(url_4d, headers=headers)

solar_data_today = pd.json_normalize(solar_response_today.json())
solar_data_tomor = pd.json_normalize(solar_response_tomor.json())
solar_data_day_3 = pd.json_normalize(solar_response_day_3.json())
solar_data_day_4 = pd.json_normalize(solar_response_day_4.json())

solar_forecast_today = pd.json_normalize(
    solar_data_today["attributes.detailedForecast"][0]
)
solar_forecast_tomor = pd.json_normalize(
    solar_data_tomor["attributes.detailedForecast"][0]
)

solar_forecast_day_3 = pd.json_normalize(
    solar_data_day_3["attributes.detailedForecast"][0]
)

solar_forecast_day_4 = pd.json_normalize(
    solar_data_day_4["attributes.detailedForecast"][0]
)

solar_forecast = pd.concat(
    [solar_forecast_today, solar_forecast_tomor, solar_forecast_day_3, solar_forecast_day_4], ignore_index=True
)
solar_forecast.index = pd.to_datetime(solar_forecast["period_start"], utc=False)
solar_forecast["time"] = solar_forecast.index.time
solar_forecast = solar_forecast.drop(columns=["period_start"])
solar_forecast[["pv_estimate10", "pv_estimate", "pv_estimate90"]] *= 0.5

# === Prepare API request ===
url_soc = f"{ha_url}api/states/{sensor_battery_charge}"

soc_params = {
    "minimal_response": "false",
    "no_attributes": "false",
    "significant_changes_only": "true",
}
# === Fetch historical state data ===
soc_current = requests.get(url_soc, headers=headers, params=soc_params)
soc_current = pd.json_normalize(soc_current.json())
soc_current = float(soc_current["state"][0])

# === Prepare API request ===
url_cap = f"{ha_url}api/states/{sensor_battery_capacity}"

# === Fetch historical state data ===
soc_cap = requests.get(url_cap, headers=headers, params=soc_params)
soc_cap = pd.json_normalize(soc_cap.json())
soc_cap = float(soc_cap["state"][0])

# === Monte Carlo Simulation ===
nsim = 20000
pct = [2.5, 50, 97.5] # CUSTOMISE 2.5/95TH PROVIDES 90% LIEKLIHOOD RANGE, 50 IS MEDIAN

soc_now = np.full(nsim, soc_current)
cum_ene = np.full(nsim, 0.0)
cum_sol = np.full(nsim, 0.0)
tod_cha, tod_exp, tod_cse, tod_dis, tod_imp = np.full(nsim, 0.0), np.full(nsim, 0.0), np.full(nsim, 0.0), np.full(nsim, 0.0), np.full(nsim, 0.0)
tom_cha, tom_exp, tom_cse, tom_dis, tom_imp = np.full(nsim, 0.0), np.full(nsim, 0.0), np.full(nsim, 0.0), np.full(nsim, 0.0), np.full(nsim, 0.0)

net_forecast = []

for index, row in solar_forecast.iterrows():

    full_datetime = index
    time_value = row["time"]

    # Find matching energy consumption profile
    match = energy_df[energy_df["time"] == time_value]
    match = pd.to_numeric(match.iloc[0, 1:7], errors="coerce").dropna().values

    sample_energy = np.random.normal(match.mean(), match.std(), size=nsim)

    # Solar generation sampling
    solar_low = row["pv_estimate10"]
    solar_med = row["pv_estimate"]
    solar_high = row["pv_estimate90"]

    u = pd.DataFrame({"ran": np.random.triangular(left = 0, mode = 0.5, right = 1, size = nsim)})
    u["x"] = np.where(
      u["ran"] < 0.1,
      solar_low * u["ran"] / 0.1,
      np.where(
          u["ran"] < 0.5,
          solar_low + (solar_med - solar_low) * (u["ran"] - 0.1) / 0.4,
          np.where(
              u["ran"] < 0.9,
              solar_med + (solar_high - solar_med) * (u["ran"] - 0.5) / 0.4,
              solar_high * (1 + (u["ran"] - 0.9) / 0.1)
          )
      )
    )
    sample_solar = u["x"].to_numpy()

    # Ensure no negative values
    sample_energy = np.maximum(sample_energy, 0)
    sample_solar = np.maximum(sample_solar, 0)

    # Cumulative sums
    cum_ene += sample_energy
    cum_sol += sample_solar

    # Net energy calculation
    sample_net = sample_solar - sample_energy

    # Percentiles calculation
    sol_low, sol_med, sol_high = np.percentile(sample_solar, pct)
    ene_low, ene_med, ene_high = np.percentile(sample_energy, pct)
    net_low, net_med, net_high = np.percentile(sample_net, pct)
    cene_low, cene_med, cene_high = np.percentile(cum_ene, pct)
    csol_low, csol_med, csol_high = np.percentile(cum_sol, pct)

    # Today's and tomorrow's totals
    if now <= full_datetime:
        if now.date() == full_datetime.date():
            tod_cha += np.minimum(np.maximum(0,sample_solar - sample_energy), (soc_cap - sample_bat))
            tod_exp += np.maximum(0,sample_solar - sample_energy - np.minimum(np.maximum(0,sample_solar - sample_energy), (soc_cap - sample_bat)))
            tod_cse += np.minimum(sample_solar,sample_energy) 
            tod_dis += np.minimum(np.maximum(0, sample_energy - np.minimum(sample_solar, sample_energy)), sample_bat)
            tod_imp += np.maximum(0, sample_energy - np.minimum(sample_solar, sample_energy) - np.minimum(np.maximum(0, sample_energy - np.minimum(sample_solar, sample_energy)), sample_bat))
        elif (now + timedelta(days=1)).date() == full_datetime.date():
            tom_cha += np.minimum(np.maximum(0,sample_solar - sample_energy), (soc_cap - sample_bat))
            tom_exp += np.maximum(0,sample_solar - sample_energy - np.minimum(np.maximum(0,sample_solar - sample_energy), (soc_cap - sample_bat)))
            tom_cse += np.minimum(sample_solar,sample_energy) 
            tom_dis += np.minimum(np.maximum(0, sample_energy - np.minimum(sample_solar, sample_energy)), sample_bat)
            tom_imp += np.maximum(0, sample_energy - np.minimum(sample_solar, sample_energy) - np.minimum(np.maximum(0, sample_energy - np.minimum(sample_solar, sample_energy)), sample_bat))

    # Update battery state of charge
    if now <= full_datetime:
        sample_bat = np.clip(soc_now + sample_net, 0, soc_cap)
    else:
        sample_bat = np.full(nsim, soc_current)

    # Battery percentiles
    bat_low, bat_med, bat_high = np.percentile(sample_bat, pct)

    # Store results
    net_forecast.append(
        {
            "index": full_datetime.isoformat(),
            "net_low": round(net_low, 2),
            "net_med": round(net_med, 2),
            "net_high": round(net_high, 2),
            "sol_low": round(sol_low, 2),
            "sol_med": round(sol_med, 2),
            "sol_high": round(sol_high, 2),
            "ene_low": round(ene_low, 2),
            "ene_med": round(ene_med, 2),
            "ene_high": round(ene_high, 2),
            "bat_low": round(bat_low, 2),
            "bat_med": round(bat_med, 2),
            "bat_high": round(bat_high, 2),
            "cene_low": round(cene_low, 2),
            "cene_med": round(cene_med, 2),
            "cene_high": round(cene_high, 2),
            "csol_low": round(csol_low, 2),
            "csol_med": round(csol_med, 2),
            "csol_high": round(csol_high, 2),
        }
    )

    # Update current battery state of charge for next iteration
    soc_now = sample_bat

# === Prepare MQTT Payload ===
net_forecast_df = pd.DataFrame(net_forecast)

forecast_payload = net_forecast_df.to_dict(orient="records")

client = mqtt.Client(
    client_id="broker_id", callback_api_version=mqtt.CallbackAPIVersion.VERSION2
)
client.username_pw_set(username, password)
client.connect(broker_address, broker_port, 60)
client.loop_start()

config_topic = "homeassistant/sensor/energy_forecast_py/config"
state_topic = "homeassistant/sensor/energy_forecast_py/state"
json_attr_topic = "homeassistant/sensor/energy_forecast_py/attributes"

config_payload = {
    "name": "Net Energy Forecast",
    "state_topic": state_topic,
    "json_attributes_topic": json_attr_topic,
    "unit_of_measurement": "kWh",
}
client.publish(config_topic, json.dumps(config_payload), retain=True)
client.publish(state_topic, "0", retain=True)

exp_payload = {"energy_forecast": forecast_payload,
               "today_charge": round(np.percentile(tod_cha,50),2),
               "today_export": round(np.percentile(tod_exp,50),2),
               "today_consumed": round(np.percentile(tod_cse,50),2),
               "today_discharge": round(np.percentile(tod_dis,50),2),
               "today_import": round(np.percentile(tod_imp,50),2),
               "tomorrow_charge": round(np.percentile(tom_cha,50),2),
               "tomorrow_export": round(np.percentile(tom_exp,50),2),
               "tomorrow_consumed": round(np.percentile(tom_cse,50),2),
               "tomorrow_discharge": round(np.percentile(tom_dis,50),2),
               "tomorrow_import": round(np.percentile(tom_imp,50),2)
              }

client.publish(json_attr_topic, json.dumps(exp_payload), retain=True)

time.sleep(2)

client.loop_stop()
client.disconnect()

print('Executed: ',now)
