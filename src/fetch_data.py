import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe.entsoe import EntsoePandasClient
from fmiopendata.wfs import download_stored_query

load_dotenv()
ENTSOE_KEY = os.getenv("ENTSOE_API_KEY")
FINGRID_KEY = os.getenv("FINGRID_API_KEY")
TZ = "Europe/Helsinki"


class DataFetcher:
    def __init__(self):
        self.check_env_vars()

    def check_env_vars(self):
        if not ENTSOE_KEY or not FINGRID_KEY:
            raise ValueError("Missing API keys in .env")

    def get_time_window(self):
        """Determines the target window for prediction (past 7 days + tomorrow"""
        now = pd.Timestamp.now(tz=TZ)
        start_history = now.normalize() - pd.Timedelta(days=7)
        # start_tomorrow = now.normalize() + pd.Timedelta(days=1)
        # end_tomorrow = start_tomorrow + pd.Timedelta(hours=23)
        end_horizon = now.normalize() + pd.Timedelta(days=3, hours=23)

        return now, start_history, end_horizon  # start_tomorrow, end_tomorrow

    def fetch_fmi_forecast(self, start_time, end_time):
        print("Fetching FMI weather forecast")

        start_str = (
            start_time.astimezone(dt.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        end_str = (
            end_time.astimezone(dt.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

        locations = ["Helsinki", "Pori"]
        params = ["Air temperature", "Wind speed"]

        weather_data = []

        stored_query_id = "fmi::forecast::harmonie::surface::point::multipointcoverage"

        for loc in locations:
            try:
                obs = download_stored_query(
                    stored_query_id,
                    args=[
                        f"place={loc}",
                        f"starttime={start_str}",
                        f"endtime={end_str}",
                    ],
                )

                # Parse FMI object structure
                for t, data in obs.data.items():
                    row = {"timestamp": t}
                    for param in params:
                        val_obj = data[loc].get(param)
                        value = (
                            val_obj["value"]
                            if isinstance(val_obj, dict)
                            else val_obj.value
                        )

                        clean_param = (
                            "temp" if "temperature" in param.lower() else "wind"
                        )
                        col_name = f"{clean_param}_{loc.lower()}"
                        row[col_name] = value

                    weather_data.append(row)

            except Exception as e:
                print(f"Error fetching FMI for {loc}: {e}")

        df = pd.DataFrame(weather_data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
                TZ
            )
            df = df.groupby("timestamp").first().reset_index()
            df.set_index("timestamp", inplace=True)
            df = df.resample("1h").mean().interpolate()

        return df

    def fetch_fingrid_data(self, start_time, end_time):
        print("Fetching findgrid forecast and nuclear")

        base_url = "https://data.fingrid.fi/api/datasets/{}/data"
        headers = {"x-api-key": FINGRID_KEY}

        buffer_start_time = start_time - pd.Timedelta(hours=24)
        start_str = buffer_start_time.astimezone(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        end_str = end_time.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        datasets = {
            245: "wind_forecast_72h_mw",
            166: "consumption_forecast_72h_mw",
            188: "nuclear_mw",
            248: "solar_forecast_72h_mw",
            # 242: "total_production_forecast_72h_mw",
        }
        dfs = []

        for ds_id, col_name in datasets.items():
            # Sleep to prevent 429 errors (Too many requests)
            time.sleep(1.6)

            try:
                url = base_url.format(ds_id)

                params = {
                    "startTime": start_str,
                    "endTime": end_str,
                    "format": "json",
                    "pageSize": 20000,
                }

                # CUT

                resp = requests.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json().get("data", [])

                if data:
                    temp_df = pd.DataFrame(data)
                    temp_df["startTime"] = pd.to_datetime(
                        temp_df["startTime"]
                    ).dt.tz_convert(TZ)
                    temp_df.rename(
                        columns={"value": col_name, "startTime": "timestamp"},
                        inplace=True,
                    )
                    temp_df.set_index("timestamp", inplace=True)
                    temp_df = temp_df[[col_name]]
                    temp_df = temp_df.resample("1h").mean()

                    dfs.append(temp_df)
                else:
                    print(f"Warning: No data returned for ID {ds_id}")

            except Exception as e:
                print(f"Error fetching fingrid ID {ds_id}: {e}")

        if dfs:
            # Join all fingrid data
            full_df = pd.concat(dfs, axis=1)

            # Fill nuclear current data to the future (persistence forecast)
            if "nuclear_mw" in full_df.columns:
                full_df["nuclear_mw"] = full_df["nuclear_mw"].ffill()

            full_df.to_csv("fingrid_full_df.csv")
            return full_df

        print("Warning: No fingrid data fetched")
        return pd.DataFrame(columns=datasets.values())

    def fetch_entsoe_history(self, end_time):
        print("Fetching Entsoe-E price history")
        client = EntsoePandasClient(api_key=ENTSOE_KEY)

        # Get history data to create lag features
        start = end_time - pd.Timedelta(days=17)

        try:
            prices = client.query_day_ahead_prices("FI", start=start, end=end_time)
            df = prices.to_frame(name="price")
            df.index.name = "timestamp"
            df.index = df.index.tz_convert(TZ)
            df = df.resample("1h").mean()
            return df
        except Exception as e:
            print(f"Error fetching Entso-E: {e}")
            return pd.DataFrame()

    # Main pipeline
    def run(self):
        now, start_history, end_horizon = self.get_time_window()
        print(f"Target prediction window: {now} to {end_horizon}")

        df_fmi = self.fetch_fmi_forecast(start_history, end_horizon)
        df_fingrid = self.fetch_fingrid_data(start_history, end_horizon)
        df_price = self.fetch_entsoe_history(end_horizon)

        master_index = pd.date_range(
            start=start_history, end=end_horizon, freq="1h", tz=TZ
        )
        df_master = pd.DataFrame(index=master_index)
        df_master = df_master.join(df_fmi, how="left")
        df_master = df_master.join(df_fingrid, how="left")

        full_dataset = df_master.combine_first(df_price)

        full_dataset = full_dataset.sort_index()

        if "nuclear_mw" in full_dataset.columns:
            full_dataset["nuclear_mw"] = full_dataset["nuclear_mw"].ffill()

        print("\nData pipeline complete")
        print(f"Latest official price timestamp: {df_price.index[-1]}")
        print(f"Forecast horizon end: {df_master.index[-1]}")

        # Extra save
        full_dataset.to_csv("fetch_data_df.csv")

        return full_dataset


if __name__ == "__main__":
    fetcher = DataFetcher()
    df = fetcher.run()

    print(df.head())
    print(df.tail())

    df.to_csv("fetch_data_df.csv")
