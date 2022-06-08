from cftime import Datetime360Day as DT360
from dash import html
import io
import numpy as np
import pandas as pd
from collections import OrderedDict

import fbfmaproom

def test_year_label_oneyear():
    assert fbfmaproom.year_label(
        DT360(1961, 6, 1),
        1
    ) == "1961"

def test_year_label_straddle():
    assert fbfmaproom.year_label(
        DT360(1960, 12, 16),
        3
    ) == "1960/61"

def test_from_month_since_360Day():
    assert fbfmaproom.from_month_since_360Day(735.5) == DT360(2021, 4, 16)

def test_table_cb():
    table, prob_thresh = fbfmaproom.table_cb.__wrapped__(
        issue_month0 = 1,
        freq=30,
        mode='0',
        geom_key='ET05',
        pathname='/fbfmaproom/ethiopia',
        severity=0,
        obs_keys=['rain', 'ndvi'],
        trigger_key="pnep",
        bad_years_key="bad-years",
        season='season1',
    )
    assert np.isclose(prob_thresh, 36.930862)

    thead, tbody = table.children
    assert len(thead.children) == 6
    row = thead.children[1]
    assert len(row.children) == 6

    cell = row.children[0]
    div, tooltip = row.children[0].children
    assert div.children == 'Act-in-vain:'

    assert row.children[1].children == 5
    assert thead.children[4].children[1].children == "66.67%"
    assert thead.children[5].children[3].children[0].children == 'Rain (mm/month)'

    assert len(tbody.children) == 40 # will break when we add a new year

    row = tbody.children[3]
    cell = row.children[0]
    assert row.children[0].children == '2019'
    assert row.children[0].className == ''
    assert row.children[1].children == 'El Niño'
    assert row.children[1].className == 'cell-el-nino'
    assert row.children[2].children == '34.28'
    assert row.children[2].className == ''
    assert row.children[3].children == '43.36'
    assert row.children[3].className == ''
    assert row.children[4].children == '0.24'
    assert row.children[4].className == 'cell-severity-0'
    assert row.children[5].children == ''
    assert row.children[5].className == 'cell-good-year'

    assert tbody.children[5].children[5].className == 'cell-bad-year'


# overlaps with test_generate_tables, but this one uses synthetic
# data. Merge them?
def test_augment_table_data():
    # year       2022 2021 2020 2019 2018 2017
    # bad_year   T    F    T    F    F    T
    # enso_state           T    F    T    F
    # enso_summ            tp   tn   fp   fn
    # worst_obs  F    F    F    F    F    T
    # obs_summ             fn   tn   tn   tp
    # worst_pnep      F    F    T    F    F
    # pnep_summ       tn   fn   fp   tn   fn
    time = [DT360(y, 1, 16) for y in range(2022, 2016, -1)]
    main_df = pd.DataFrame(
        index=time,
        data={
            "bad-years": [1, 0, 1, 0, 0, 1],
            "enso_state": [np.nan, np.nan, "El Niño", "La Niña", "El Niño", "Neutral"],
            "pnep": [np.nan, 19.606438, 29.270180, 33.800949, 12.312943, 1.],
            "rain": [np.nan, np.nan, 200., 400., 300., 100.],
            "time": time,
        }
    )
    freq = 34
    table_columns = {
        "bad-years": {
            "lower_is_worse": False,
            "type": fbfmaproom.ColType.OBS,
        },
        "pnep": {
            "lower_is_worse": False,
            "type": fbfmaproom.ColType.FORECAST,
        },
        "rain": {
            "lower_is_worse": True,
            "type": fbfmaproom.ColType.OBS,
        },
    }

    aug, summ, prob = fbfmaproom.augment_table_data(main_df, freq, table_columns, "pnep", "bad-years")

    expected_aug = main_df.copy()
    expected_aug["worst_bad-years"] = [1, 0, 1, 0, 0, 1]
    expected_aug["worst_pnep"] = [np.nan, 0, 0, 1, 0, 0]
    expected_aug["worst_rain"] = [np.nan, np.nan, 0, 0, 0, 1]
    pd.testing.assert_frame_equal(expected_aug, aug, check_column_type=True)

    expected_summ = pd.DataFrame(dict(
        # [tp, fp, fn, tn, accuracy]
        enso_state=[1, 1, 1, 1, "50.00%"],
        pnep=[0, 1, 2, 2, "40.00%"],
        rain=[1, 0, 1, 2, "75.00%"],
    ))
    pd.testing.assert_frame_equal(expected_summ, summ)

    assert np.isclose(prob, 33.800949)

def test_forecast_tile_url_callback_yesdata():
    url, is_alert, colormap = fbfmaproom.tile_url_callback.__wrapped__(
        2021, 2, 30, '/fbfmaproom/ethiopia', 'pnep', 'season1'
    )
    assert url == '/fbfmaproom-tiles/forecast/pnep/{z}/{x}/{y}/ethiopia/season1/2021/2/30'
    assert not is_alert
    assert type(colormap) == list

def test_forecast_tile_url_callback_nodata():
    url, is_alert, colormap = fbfmaproom.tile_url_callback.__wrapped__(
        3333, 2, 30, '/fbfmaproom/ethiopia', 'pnep', 'season1'
    )
    assert url == ''
    assert is_alert
    assert type(colormap) == list

def test_forecast_tile():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom-tiles/forecast/pnep/6/40/27/ethiopia/season1/2021/2/30')
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"

def test_obs_tile():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom-tiles/obs/rain/6/40/27/ethiopia/season1/2021')
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"

def test_vuln_tile():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get("/fbfmaproom-tiles/vuln/6/39/30/ethiopia/0/2019")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"

def test_pnep_percentile_pixel_trigger():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/pnep_percentile?country_key=ethiopia"
            "&mode=pixel"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&prob_thresh=10"
            "&bounds=[[6.75, 43.75], [7, 44]]"
        )
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["probability"], 13.415)
    assert d["triggered"] is True

def test_pnep_percentile_pixel_notrigger():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/pnep_percentile?country_key=ethiopia"
            "&mode=pixel"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&prob_thresh=20"
            "&bounds=[[6.75, 43.75], [7, 44]]"
        )
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["probability"], 13.415)
    assert d["triggered"] is False

def test_pnep_percentile_region():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/pnep_percentile?country_key=ethiopia"
            "&mode=2"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&prob_thresh=20"
            "&region=(ET05,ET0505,ET050501)"
        )
    print(r.data)
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["probability"], 14.6804)
    assert d["triggered"] is False

def test_pnep_percentile_straddle():
    "Lead time spans Jan 1"
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/pnep_percentile?country_key=malawi"
            "&mode=0"
            "&season=season1"
            "&issue_month=10"
            "&season_year=2021"
            "&freq=30.0"
            "&prob_thresh=30.31437"
            "&region=152"
        )
    print(r.data)
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["probability"], 33.10532)
    assert d["triggered"] is True


def test_stats():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom-admin/stats')
        print(resp.data)
        assert resp.status_code == 200

def test_update_selected_region_pixel():
    positions, key = fbfmaproom.update_selected_region.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        'pixel'
    )
    assert positions == [[[[6.75, 43.75], [7.0, 43.75], [7.0, 44.0], [6.75, 44.0]]]]
    assert key == "[[6.75, 43.75], [7.0, 44.0]]"

def test_update_selected_region_level0():
    positions, key = fbfmaproom.update_selected_region.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        '0'
    )
    assert len(positions[0][0]) == 1323
    assert key == "ET05"

def test_update_selected_region_level1():
    positions, key = fbfmaproom.update_selected_region.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        '1'
    )
    assert len(positions[0][0]) == 143
    assert key == "(ET05,ET0505)"

def test_update_popup_pixel():
    title, content = fbfmaproom.update_popup.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        'pixel',
        2017,
    )
    print(repr(content))
    assert isinstance(title, html.H3)
    assert title.children == '6.87500° N 43.87500° E'
    assert content.children == []

def test_update_popup_level0():
    title, content = fbfmaproom.update_popup.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        '0',
        2017,
    )
    assert isinstance(title, html.H3)
    assert title.children == 'Somali'
    assert len(content.children) == 12
    assert content.children[0].children == 'Vulnerability: '
    assert content.children[1].strip() == '31'

def test_hits_and_misses():
    # year       1960 1961 1962 1963 1964 1965 1966 1967
    # prediction T    F    T    F    F    T
    # truth                T    F    T    F    T    F
    # true_pos             1
    # false_pos                           1
    # false_neg                      1
    # true_neg                  1
    prediction = pd.Series(
        data=[True, False, True, False, False, True],
        index=[DT360(1960 + x, 1, 1) for x in range(6)]
    )
    truth = pd.Series(
        data=[True, False, True, False, True, False],
        index=[DT360(1962 + x, 1, 1) for x in range(6)]
    )
    true_pos, false_pos, false_neg, true_neg, pct = fbfmaproom.hits_and_misses(prediction, truth)
    assert true_pos == 1
    assert false_pos == 1
    assert false_neg == 1
    assert true_neg == 1
    assert pct == "50.00%"
