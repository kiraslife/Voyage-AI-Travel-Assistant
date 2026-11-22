# ui_components.py
import re
import html
import logging
from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.graph_objects as go

logger = logging.getLogger("VoyageAI.UI")

def _pick_attraction_image(name: str, category: str, idx: int) -> str:
    text = f"{name} {category}".lower()
    if "fort" in text or "palace" in text or "monument" in text:
        return "https://images.unsplash.com/photo-1599661046289-e31897846e41?auto=format&fit=crop&w=500&q=70"
    if "temple" in text or "heritage" in text:
        return "https://images.unsplash.com/photo-1548013146-72479768bada?auto=format&fit=crop&w=500&q=70"
    if "lake" in text or "waterfall" in text:
        return "https://images.unsplash.com/photo-1544735716-392fe2489ffa?auto=format&fit=crop&w=500&q=70"
    if "park" in text or "garden" in text or "nature" in text:
        return "https://images.unsplash.com/photo-1587474260584-136574528ed5?auto=format&fit=crop&w=500&q=70"
    if "beach" in text or "island" in text or "coast" in text:
        return "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=500&q=70"
    defaults = [
        "https://images.unsplash.com/photo-1524492412937-b28074a5d7da?auto=format&fit=crop&w=500&q=70",
        "https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=500&q=70",
        "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=500&q=70",
    ]
    return defaults[idx % len(defaults)]


def _html(raw_html: str) -> None:
    """
    Strips leading indentation on every line and blank lines so Streamlit's
    CommonMark parser NEVER mistakes nested HTML tags for a <pre><code> block.
    """
    cleaned = "\n".join(line.lstrip() for line in raw_html.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_dribbble_css() -> None:
    """Injects the dark travel SaaS theme and compact 3-column bento styling."""
    _html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg-main: #08090B;
            --bg-sidebar: #101216;
            --bg-card: #17191F;
            --bg-card-sec: #20232B;
            --accent-cyan: #35BDF4;
            --accent-purple: #A78BFA;
            --accent-green: #34D399;
            --text-primary: #F5F5F5;
            --text-secondary: #9298A4;
            --border-subtle: #30343D;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
        }

        .stApp {
            background-color: var(--bg-main) !important;
        }
        .block-container {
            padding-top: 0.75rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1.15rem !important;
            padding-right: 1.15rem !important;
            max-width: 100% !important;
        }
        header[data-testid="stHeader"] {
            display: none !important;
        }

        section[data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-subtle) !important;
            width: 220px !important;
            min-width: 220px !important;
            max-width: 220px !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding: 0.65rem 0.65rem !important;
        }

        .v-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 14px;
            margin-bottom: 10px;
            transition: border-color 0.18s ease;
            position: relative;
            overflow: hidden;
        }
        .v-card:hover {
            border-color: #434956;
        }
        .v-pill-btn {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-card-sec);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 500;
        }

        .v-featured-img-wrap {
            position: relative;
            width: 100%;
            height: 195px;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 12px;
        }
        .v-featured-img-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .v-img-counter {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(8, 9, 11, 0.75);
            backdrop-filter: blur(6px);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 999px;
        }
        .v-route-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(8, 9, 11, 0.78);
            backdrop-filter: blur(6px);
            border: 1px solid var(--border-subtle);
            color: var(--accent-cyan);
            font-size: 11px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 999px;
        }
        .v-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border-subtle);
            font-size: 11.5px;
            color: var(--text-secondary);
            align-items: center;
        }
        .v-status-dot {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            color: var(--accent-green);
            font-weight: 600;
        }

        .v-sched-purple {
            background: rgba(167, 139, 250, 0.14);
            border-left: 3px solid var(--accent-purple);
            border-radius: 6px;
            padding: 6px 9px;
            margin-bottom: 7px;
        }
        .v-sched-green {
            background: rgba(52, 211, 153, 0.14);
            border-left: 3px solid var(--accent-green);
            border-radius: 6px;
            padding: 6px 9px;
            margin-bottom: 7px;
        }
        .v-sched-cyan {
            background: rgba(53, 189, 244, 0.14);
            border-left: 3px solid var(--accent-cyan);
            border-radius: 6px;
            padding: 6px 9px;
            margin-bottom: 6px;
        }

        .v-weather-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 6px;
            margin-top: 10px;
            text-align: center;
        }
        .v-weather-col {
            background: var(--bg-card-sec);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 8px 4px;
        }

        .v-hotel-grid {
            display: grid;
            grid-template-columns: 44% 56%;
            gap: 12px;
            align-items: center;
        }
        .v-hotel-img {
            width: 100%;
            height: 138px;
            border-radius: 11px;
            object-fit: cover;
        }

        .v-budget-wrap {
            display: grid;
            grid-template-columns: 36% 64%;
            gap: 12px;
            align-items: center;
        }
        .v-prog-track {
            width: 100%;
            height: 5px;
            background: #262A34;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 3px;
            margin-bottom: 7px;
        }

        .v-attractions-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .v-attr-card {
            position: relative;
            height: 132px;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-subtle);
        }
        .v-attr-card img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .v-attr-overlay {
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(8,9,11,0.1) 20%, rgba(8,9,11,0.88) 100%);
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            padding: 8px;
        }

        .v-saved-card {
            position: relative;
            height: 78px;
            border-radius: 11px;
            overflow: hidden;
            border: 1px solid var(--border-subtle);
            margin-bottom: 6px;
        }
        .v-saved-card img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .v-saved-overlay {
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(8,9,11,0.15) 0%, rgba(8,9,11,0.86) 100%);
            padding: 7px 9px;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        }

        .v-ai-dock {
            background: rgba(23, 25, 31, 0.95);
            border: 1px solid #35BDF455;
            box-shadow: 0 0 24px rgba(53, 189, 244, 0.12);
            border-radius: 16px;
            padding: 12px 14px;
            margin-top: 4px;
        }

        div.stButton > button {
            background-color: var(--bg-card-sec) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 999px !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            padding: 0.28rem 0.75rem !important;
            min-height: 30px !important;
            transition: all 0.15s ease !important;
        }
        div.stButton > button:hover {
            border-color: var(--accent-cyan) !important;
            color: var(--accent-cyan) !important;
        }
        div.stButton > button[kind="primary"] {
            background-color: #35BDF4 !important;
            color: #08090B !important;
            border-color: #35BDF4 !important;
            font-weight: 600 !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #F5F5F5 !important;
            border-color: #F5F5F5 !important;
            color: #08090B !important;
        }
        </style>
        """
    )


def render_error_banner(error_info: Optional[Dict[str, Any]]) -> None:
    """Renders a compact production error notification without raw Python tracebacks."""
    if not error_info:
        return

    provider = html.escape(str(error_info.get("provider", "AI Provider")))
    model = html.escape(str(error_info.get("model", "gemini-2.5-flash")))
    category = html.escape(str(error_info.get("category", "Request Quota Exceeded")))
    action = html.escape(str(error_info.get("suggested_action", "Please try again later or switch providers in Settings.")))
    fb_status = html.escape(str(error_info.get("fallback_status", "No secondary fallback provider configured.")))

    _html(
        f"""
        <div style="background: rgba(239, 68, 68, 0.10); border: 1px solid rgba(239, 68, 68, 0.45); border-radius: 12px; padding: 10px 14px; margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="color:#F87171; font-weight:600; font-size:12.5px;">
                    ⚠️ Voyage AI is temporarily unable to generate a new AI response because the selected provider reached its request quota.
                </span>
                <span style="background:#20232B; border:1px solid #30343D; color:#F5F5F5; font-size:11px; padding:2px 8px; border-radius:999px;">
                    {provider} • {model}
                </span>
            </div>
            <div style="font-size:11.5px; color:#9298A4; line-height:1.45;">
                <strong style="color:#F5F5F5;">Category:</strong> {category} &nbsp;|&nbsp;
                <strong style="color:#F5F5F5;">Fallback Status:</strong> {fb_status}<br/>
                <strong style="color:#35BDF4;">Suggested Action:</strong> {action}
            </div>
        </div>
        """
    )


def render_map_card(
    destination_name: str,
    dest_coords: Optional[Dict[str, float]],
    places: List[Dict[str, Any]],
) -> None:
    """Renders the MapLibre go.Scattermap card safely."""
    _html(
        """
        <div style="background:#17191F; border:1px solid #30343D; border-bottom:none; border-radius:16px 16px 0 0; padding:10px 12px 4px 12px; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:12px; font-weight:600; color:#F5F5F5;">🗺️ Destination Map</span>
            <span style="font-size:10.5px; color:#35BDF4; background:#20232B; padding:2px 7px; border-radius:999px; border:1px solid #30343D;">
                MapLibre Live
            </span>
        </div>
        """
    )

    try:
        valid_points: List[Dict[str, Any]] = []
        for p in places or []:
            lat = p.get("lat") if p.get("lat") is not None else p.get("latitude")
            lon = p.get("lon") if p.get("lon") is not None else p.get("longitude")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                if -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180:
                    valid_points.append({
                        "name": str(p.get("name", "Attraction")),
                        "lat": float(lat),
                        "lon": float(lon),
                        "category": str(p.get("category", "Point of Interest")),
                        "rating": p.get("rating", "N/A"),
                    })

        center_lat, center_lon = None, None
        if dest_coords and isinstance(dest_coords.get("lat"), (int, float)) and isinstance(dest_coords.get("lon"), (int, float)):
            center_lat = float(dest_coords["lat"])
            center_lon = float(dest_coords["lon"])
        elif valid_points:
            center_lat = sum(pt["lat"] for pt in valid_points) / len(valid_points)
            center_lon = sum(pt["lon"] for pt in valid_points) / len(valid_points)

        if center_lat is None or center_lon is None:
            _html(
                f"""
                <div class="v-card" style="border-radius:0 0 16px 16px; height:158px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
                    <div style="font-size:18px; margin-bottom:4px;">📍</div>
                    <div style="font-size:12px; font-weight:600; color:#F5F5F5;">Map Coordinates Unavailable</div>
                    <div style="font-size:11px; color:#9298A4; max-width:210px; margin-top:2px;">
                        No verified geographic coordinates found in dataset for {html.escape(destination_name or 'this destination')}.
                    </div>
                </div>
                """
            )
            return

        if not valid_points:
            valid_points.append({
                "name": destination_name,
                "lat": center_lat,
                "lon": center_lon,
                "category": "Destination Hub",
                "rating": "Hub",
            })

        lats = [pt["lat"] for pt in valid_points]
        lons = [pt["lon"] for pt in valid_points]
        labels = [pt["name"] for pt in valid_points]
        hover_texts = [
            f"<b>{pt['name']}</b><br>Category: {pt['category']}<br>Rating: ⭐ {pt['rating']}"
            for pt in valid_points
        ]

        fig = go.Figure()
        if hasattr(go, "Scattermap"):
            fig.add_trace(
                go.Scattermap(
                    lat=lats,
                    lon=lons,
                    mode="markers+text",
                    marker=dict(size=11, color="#35BDF4"),
                    text=labels,
                    textposition="top right",
                    textfont=dict(size=10, color="#F5F5F5"),
                    hovertext=hover_texts,
                    hoverinfo="text",
                )
            )
            fig.update_layout(
                map=dict(
                    style="carto-darkmatter",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=10.2,
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=158,
                paper_bgcolor="#17191F",
                plot_bgcolor="#17191F",
                showlegend=False,
            )
        else:
            raise RuntimeError("Installed Plotly version does not expose go.Scattermap.")

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})

    except Exception as exc:
        logger.error("Map rendering failed safely: %s", exc)
        _html(
            """
            <div class="v-card" style="border-radius:0 0 16px 16px; height:158px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
                <div style="font-size:12px; font-weight:600; color:#F5F5F5;">Map Preview Temporarily Unavailable</div>
                <div style="font-size:11px; color:#9298A4; margin-top:4px;">
                    Interactive map could not be initialized. Itinerary data remains active.
                </div>
            </div>
            """
        )


def render_featured_trip_card(trip: Dict[str, Any]) -> None:
    """Renders the Left-Upper Featured Trip Card."""
    dest = html.escape(str(trip.get("destination", "Goa")))
    source = html.escape(str(trip.get("source", "Delhi")))
    title = html.escape(str(trip.get("title", f"{dest} Signature Trip 2026")))
    dates = html.escape(str(trip.get("dates", "21 Nov – 25 Nov 2026")))
    days = int(trip.get("days", 4))
    travelers = int(trip.get("travelers", 2))
    status = html.escape(str(trip.get("status", "On Track")))
    summary = html.escape(str(trip.get("summary", "")))
    image_url = html.escape(str(trip.get("image_url", "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=900&q=80")))

    flight = trip.get("flight") or {}
    flight_str = (
        f"{flight.get('airline', 'IndiGo')} ({flight.get('flight_no', flight.get('flight_id', '6E-204'))}) • ₹{int(float(flight.get('price', 5400))):,}"
        if flight else "Flight selection available in dataset"
    )

    _html(
        f"""
        <div class="v-card">
            <div class="v-featured-img-wrap">
                <img src="{image_url}" alt="{dest}"/>
                <div class="v-img-counter">1 / {max(len(trip.get('places', [])), 4)}</div>
                <div class="v-route-badge">✈️ {source} → {dest}</div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div style="font-size:17px; font-weight:700; color:#F5F5F5; letter-spacing:-0.01em;">
                    {title}
                </div>
                <div style="font-size:11px; color:#9298A4;">{dates}</div>
            </div>
            <div style="font-size:12px; color:#9298A4; margin-top:6px; line-height:1.45; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">
                {summary}
            </div>
            <div style="margin-top:8px; background:#20232B; border:1px solid #30343D; border-radius:8px; padding:5px 9px; font-size:11px; color:#35BDF4;">
                🛫 <strong>Recommended Flight:</strong> {html.escape(flight_str)}
            </div>
            <div class="v-meta-row">
                <span>🕒 {days} Days</span>
                <span>•</span>
                <span>👥 {travelers} Travelers</span>
                <span>•</span>
                <span class="v-status-dot">● {status}</span>
            </div>
        </div>
        """
    )


def render_arrival_schedule_card(trip: Dict[str, Any]) -> None:
    """Interactive day-by-day schedule card with Day tabs and scrollable timeline."""
    import html as _hl
    days_count = max(1, int(trip.get("days", 3) or 3))
    day_plans = trip.get("day_plans") or []

    # Build multi-day fallback from existing schedule/places if preset trip has no day_plans yet
    if not day_plans:
        base_sched = trip.get("schedule") or []
        places = trip.get("places") or []
        hotel_name = (trip.get("hotel") or {}).get("name", "Local Hotel")
        for d in range(1, days_count + 1):
            items = []
            if d == 1:
                items.append({"time": "10:00 AM", "activity": f"Check-in: {hotel_name}", "category": "Stay", "note": "Freshen up & drop luggage"})
                for s in base_sched[:2]:
                    items.append({"time": s.get("time", "02:00 PM"), "activity": s.get("activity", "City Walk"), "category": "Sightseeing", "note": "Curated stop"})
            else:
                p_idx = (d - 2) % max(1, len(places))
                p_name = places[p_idx].get("name", f"Explore {trip.get('destination', 'City')}") if places else f"Explore {trip.get('destination', 'City')}"
                items.append({"time": "09:30 AM", "activity": f"Visit {p_name}", "category": "Sightseeing", "note": "Beat the midday crowd"})
                items.append({"time": "01:30 PM", "activity": f"Local Cuisine Lunch in {trip.get('destination', 'City')}", "category": "Food", "note": "Try signature regional dishes"})
                items.append({"time": "05:30 PM", "activity": "Evening Market & Sunset Walk", "category": "Leisure", "note": "Golden hour photos"})
            day_plans.append({"day": d, "theme": f"Day {d} Highlights", "items": items})

    _html(
        f'<div class="bento-card" style="padding:12px 14px 6px 14px; margin-bottom:6px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">'
        f'<span style="font-size:12px; font-weight:700; color:#F5F5F5;">📅 Day-by-Day Schedule</span>'
        f'<span style="font-size:10px; color:#60A5FA; background:#1E293B; padding:2px 8px; border-radius:999px;">{len(day_plans)} Days • Scrollable</span>'
        f'</div>'
        f'<div style="font-size:11px; color:#9CA3AF;">{_hl.escape(str(trip.get("dates", f"{days_count}-Day Plan")))}</div>'
        f'</div>'
    )

    tab_labels = [f"Day {dp.get('day', i + 1)}" for i, dp in enumerate(day_plans)] + ["All"]
    tabs = st.tabs(tab_labels)
    cat_colors = {"Sightseeing": "#3B82F6", "Food": "#F59E0B", "Leisure": "#10B981", "Stay": "#8B5CF6", "Travel": "#EC4899"}

    def _render_slot(item: Dict[str, Any]) -> None:
        cat = str(item.get("category", "Sightseeing"))
        accent = cat_colors.get(cat, "#38BDF8")
        t_str = _hl.escape(str(item.get("time", "")))
        act = _hl.escape(str(item.get("activity", "")))
        note = _hl.escape(str(item.get("note", "")))
        _html(
            f'<div style="background:#171A23; border:1px solid #262A36; border-left:3px solid {accent}; '
            f'border-radius:8px; padding:7px 10px; margin-bottom:6px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; gap:6px;">'
            f'<span style="font-size:12px; font-weight:600; color:#F3F4F6;">{act}</span>'
            f'<span style="font-size:10px; color:{accent}; white-space:nowrap;">{t_str}</span>'
            f'</div>'
            f'{f"<div style=\"font-size:10.5px; color:#9CA3AF; margin-top:2px;\">{note}</div>" if note else ""}'
            f'</div>'
        )

    for idx, dp in enumerate(day_plans):
        with tabs[idx]:
            with st.container(height=185, border=False):
                theme = _hl.escape(str(dp.get("theme", f"Day {idx + 1}")))
                _html(f'<div style="font-size:11px; font-weight:600; color:#93C5FD; margin-bottom:6px;">✨ {theme}</div>')
                for it in (dp.get("items") or []):
                    _render_slot(it)

    with tabs[-1]:
        with st.container(height=185, border=False):
            for idx, dp in enumerate(day_plans):
                theme = _hl.escape(str(dp.get("theme", f"Day {idx + 1}")))
                _html(f'<div style="font-size:11px; font-weight:700; color:#60A5FA; margin:4px 0;">Day {dp.get("day", idx + 1)} — {theme}</div>')
                for it in (dp.get("items") or []):
                    _render_slot(it)


def render_weather_card(weather_data: List[Dict[str, Any]], destination: str) -> bool:
    """Renders the Right-Upper 5-Day Forecast card without raw HTML code blocks."""
    cols_html = ""
    if weather_data:
        for day in weather_data[:5]:
            d_name = html.escape(str(day.get("day", "Day")))
            d_date = html.escape(str(day.get("date_short", "")))
            icon = html.escape(str(day.get("icon", "☀️")))
            cond = html.escape(str(day.get("condition", "Clear")))
            temp_max = html.escape(str(day.get("temp_max", "--")))
            temp_min = html.escape(str(day.get("temp_min", "--")))
            cols_html += (
                f'<div class="v-weather-col">'
                f'<div style="font-size:11px; font-weight:600; color:#F5F5F5;">{d_name}</div>'
                f'<div style="font-size:9.5px; color:#9298A4; margin-bottom:2px;">{d_date}</div>'
                f'<div style="font-size:17px; margin:2px 0;">{icon}</div>'
                f'<div style="font-size:9px; color:#9298A4; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{cond}</div>'
                f'<div style="font-size:11.5px; font-weight:700; color:#F5F5F5; margin-top:3px;">{temp_max}°C</div>'
                f'<div style="font-size:9.5px; color:#9298A4;">{temp_min}°C</div>'
                f'</div>'
            )
    else:
        cols_html = (
            '<div style="grid-column: span 5; background:#20232B; border:1px solid #30343D; border-radius:10px; padding:18px; text-align:center; color:#9298A4; font-size:11.5px;">'
            'Live Open-Meteo weather forecast is unavailable right now. Click Refresh to retry.'
            '</div>'
        )

    _html(
        f"""
        <div class="v-card" style="height:192px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:13px; font-weight:600; color:#F5F5F5;">5 Days forecast • {html.escape(destination)}</span>
                <span style="font-size:10.5px; color:#34D399;">Open-Meteo API</span>
            </div>
            <div class="v-weather-grid">
                {cols_html}
            </div>
        </div>
        """
    )
    return st.button("↻ Refresh Weather", key="refresh_weather_btn")


def render_hotel_card(hotel: Optional[Dict[str, Any]], destination: str) -> bool:
    """Renders the Center-Middle Recommended Hotel Card."""
    if not hotel:
        _html(
            f"""
            <div class="v-card" style="height:168px; display:flex; align-items:center; justify-content:center; color:#9298A4; font-size:12px;">
                No matching hotel found in dataset for {html.escape(destination)} within selected filters.
            </div>
            """
        )
        return False

    name = html.escape(str(hotel.get("name", hotel.get("hotel_name", "Recommended Resort & Spa"))))
    rating = html.escape(str(hotel.get("rating", hotel.get("stars", "4.7"))))
    location = html.escape(str(hotel.get("location", hotel.get("city", destination))))
    price = int(float(hotel.get("price_per_night", hotel.get("price", 4200))))
    desc = html.escape(str(hotel.get("description", "Curated stay selected from verified local hospitality dataset.")))
    img = html.escape(str(hotel.get("image_url", "https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&w=700&q=80")))

    _html(
        f"""
        <div class="v-card" style="height:168px;">
            <div class="v-hotel-grid">
                <div>
                    <img class="v-hotel-img" src="{img}" alt="{name}"/>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:10.5px; color:#35BDF4; font-weight:600; text-transform:uppercase;">Recommended Hotel</span>
                        <span style="font-size:11px; color:#FBBF24; font-weight:600;">★ {rating}</span>
                    </div>
                    <div style="font-size:14.5px; font-weight:700; color:#F5F5F5; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                        {name}
                    </div>
                    <div style="font-size:11px; color:#9298A4; margin-bottom:4px;">
                        📍 {location} • <strong style="color:#34D399;">₹{price:,}/night</strong>
                    </div>
                    <div style="font-size:11px; color:#9298A4; line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin-bottom:8px;">
                        {desc}
                    </div>
                    <div style="display:inline-block; background:#20232B; border:1px solid #30343D; border-radius:999px; padding:4px 12px; font-size:11px; color:#F5F5F5;">
                        Recommended Stay ↗
                    </div>
                </div>
            </div>
        </div>
        """
    )
    return False


def render_ai_insight_card(trip: Dict[str, Any], agent_meta: Dict[str, Any]) -> None:
    """Renders the Right-Middle AI Travel Insight Card."""
    insight = html.escape(str(trip.get("ai_insight", "Itinerary optimized using deterministic local datasets for flights, hotels, and weather windows.")))
    prefs = trip.get("preferences", ["Culture & Heritage", "Scenic Views", "Budget Optimized"])
    prefs_html = "".join(
        f'<span style="background:#20232B; border:1px solid #30343D; border-radius:999px; padding:2px 8px; font-size:10px; color:#A78BFA; margin-right:5px;">{html.escape(str(p))}</span>'
        for p in prefs[:3]
    )
    provider_badge = html.escape(f"{agent_meta.get('provider', 'Gemini')} • {agent_meta.get('model', 'gemini-2.5-flash')}")

    _html(
        f"""
        <div class="v-card" style="height:168px; display:grid; grid-template-columns: 34% 66%; gap:12px; align-items:center;">
            <div style="background:#20232B; border:1px solid #30343D; border-radius:12px; padding:10px; transform:rotate(-3deg); text-align:center;">
                <div style="font-size:22px;">🧭</div>
                <div style="font-size:11px; font-weight:700; color:#F5F5F5; margin-top:4px;">Voyage AI</div>
                <div style="font-size:9.5px; color:#34D399; margin-top:2px;">● Agent Verified</div>
                <div style="font-size:9px; color:#9298A4; margin-top:4px;">{provider_badge}</div>
            </div>
            <div>
                <div style="font-size:11px; color:#35BDF4; font-weight:600; margin-bottom:3px;">AI TRAVEL INSIGHT</div>
                <div style="font-size:12px; color:#F5F5F5; line-height:1.4; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; margin-bottom:8px;">
                    "{insight}"
                </div>
                <div style="margin-bottom:4px;">{prefs_html}</div>
            </div>
        </div>
        """
    )


def render_budget_card(budget: Dict[str, Any]) -> None:
    """Renders the Left-Lower Budget Breakdown Card cleanly with zero raw HTML leaks."""
    flights = float(budget.get("flights", 10800))
    hotels = float(budget.get("hotels", 16800))
    food = float(budget.get("food", 6400))
    transport = float(budget.get("transport", 3600))
    total = float(budget.get("total", flights + hotels + food + transport)) or 1.0

    f_pct = int(round((flights / total) * 100))
    h_pct = int(round((hotels / total) * 100))
    fd_pct = int(round((food / total) * 100))
    t_pct = max(0, 100 - (f_pct + h_pct + fd_pct))

    _html(
        f"""
        <div class="v-card">
            <div class="v-budget-wrap">
                <div style="background:#20232B; border:1px solid #30343D; border-radius:12px; padding:14px 10px; text-align:center;">
                    <div style="font-size:10.5px; color:#9298A4;">Total Est. Budget</div>
                    <div style="font-size:21px; font-weight:700; color:#F5F5F5; margin:6px 0 10px 0;">
                        ₹{int(total):,}
                    </div>
                    <div style="background:#17191F; border:1px solid #30343D; border-radius:999px; padding:4px 10px; font-size:11px; color:#9298A4; display:inline-block;">
                        Details
                    </div>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:11px;">
                        <span style="color:#9298A4;">✈️ Flights</span>
                        <span style="color:#F5F5F5; font-weight:600;">₹{int(flights):,} <span style="color:#9298A4;">({f_pct}%)</span></span>
                    </div>
                    <div class="v-prog-track"><div style="width:{f_pct}%; height:100%; background:#35BDF4;"></div></div>
                    <div style="display:flex; justify-content:space-between; font-size:11px;">
                        <span style="color:#9298A4;">🏨 Hotels</span>
                        <span style="color:#F5F5F5; font-weight:600;">₹{int(hotels):,} <span style="color:#9298A4;">({h_pct}%)</span></span>
                    </div>
                    <div class="v-prog-track"><div style="width:{h_pct}%; height:100%; background:#34D399;"></div></div>
                    <div style="display:flex; justify-content:space-between; font-size:11px;">
                        <span style="color:#9298A4;">🍽️ Food & Dining</span>
                        <span style="color:#F5F5F5; font-weight:600;">₹{int(food):,} <span style="color:#9298A4;">({fd_pct}%)</span></span>
                    </div>
                    <div class="v-prog-track"><div style="width:{fd_pct}%; height:100%; background:#A78BFA;"></div></div>
                    <div style="display:flex; justify-content:space-between; font-size:11px;">
                        <span style="color:#9298A4;">🚕 Local Transport</span>
                        <span style="color:#F5F5F5; font-weight:600;">₹{int(transport):,} <span style="color:#9298A4;">({t_pct}%)</span></span>
                    </div>
                    <div class="v-prog-track" style="margin-bottom:0;"><div style="width:{t_pct}%; height:100%; background:#FBBF24;"></div></div>
                </div>
            </div>
        </div>
        """
    )


def render_attraction_cards(places: List[Dict[str, Any]]) -> None:
    """Scrollable real attractions gallery displaying all discovered destination landmarks."""
    import html as _hl
    places = places or []
    _html(
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin:8px 0 6px 0;">'
        f'<span style="font-size:13px; font-weight:700; color:#F5F5F5;">Top Discovered Attractions</span>'
        f'<span style="font-size:11px; color:#9CA3AF;">{len(places)} Verified Places • Scrollable</span>'
        f'</div>'
    )
    if not places:
        st.info("Search any destination above to discover top attractions.")
        return

    with st.container(height=230, border=False):
        for row_start in range(0, len(places), 3):
            row_items = places[row_start : row_start + 3]
            cols = st.columns(3, gap="small")
            for c_idx, place in enumerate(row_items):
                g_idx = row_start + c_idx
                name = str(place.get("name", "Landmark"))
                cat = str(place.get("category", place.get("type", "Landmark")))
                rating = place.get("rating", 4.5)
                desc = str(place.get("description", ""))
                img_url = _pick_attraction_image(name, cat, g_idx)
                with cols[c_idx]:
                    _html(
                        f'<div style="background:#161922; border:1px solid #262A36; border-radius:12px; overflow:hidden; margin-bottom:8px;">'
                        f'<div style="position:relative; height:98px; background:url('{img_url}') center/cover no-repeat;">'
                        f'<div style="position:absolute; inset:0; background:linear-gradient(to top, rgba(15,17,23,0.92) 8%, rgba(15,17,23,0.12) 70%);"></div>'
                        f'<span style="position:absolute; top:6px; left:6px; font-size:9.5px; font-weight:600; color:#E5E7EB; background:rgba(15,17,23,0.78); padding:2px 7px; border-radius:999px;">{_hl.escape(cat)}</span>'
                        f'<span style="position:absolute; top:6px; right:6px; font-size:10px; font-weight:700; color:#FBBF24; background:rgba(15,17,23,0.78); padding:2px 6px; border-radius:999px;">★ {rating}</span>'
                        f'<div style="position:absolute; bottom:6px; left:8px; right:8px; font-size:12px; font-weight:700; color:#FFFFFF; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{_hl.escape(name)}</div>'
                        f'</div>'
                        f'{f"<div style=\"padding:6px 8px; font-size:10.5px; color:#9CA3AF; line-height:1.3; min-height:34px;\">{_hl.escape(desc[:72])}</div>" if desc else ""}'
                        f'</div>'
                    )


def render_packing_checklist(packing_state: Dict[str, Dict[str, bool]]) -> None:
    """Renders the Right-Lower Interactive Mandatory Packing Checklist."""
    total_items = sum(len(items) for items in packing_state.values())
    checked_items = sum(1 for items in packing_state.values() for v in items.values() if v)

    _html(
        f"""
        <div style="background:#17191F; border:1px solid #30343D; border-radius:16px; padding:12px 14px 6px 14px; margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:13px; font-weight:600; color:#F5F5F5;">Mandatory Packing Essential</span>
                <span style="font-size:11px; color:#34D399; font-weight:600;">{checked_items}/{total_items} Packed</span>
            </div>
        </div>
        """
    )

    with st.container():
        for category, items in packing_state.items():
            with st.expander(f"📦 {category}", expanded=(category in ["Travel Documents", "Clothing"])):
                for item_label, is_checked in items.items():
                    state_key = f"pack_chk_{category}_{item_label}"
                    new_val = st.checkbox(item_label, value=is_checked, key=state_key)
                    packing_state[category][item_label] = new_val